import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_geometric_features(ply_path: Path) -> np.ndarray:
    """Extract geometric features from a PLY chunk."""
    try:
        plydata = PlyData.read(ply_path)
        xyz = np.stack((plydata['vertex']['x'], 
                       plydata['vertex']['y'], 
                       plydata['vertex']['z']), axis=1)
        
        if len(xyz) == 0:
            return np.zeros(16)  # Return zero vector for empty chunks
        
        # Compute features
        centroid = np.mean(xyz, axis=0)  # 3D
        bbox_min = np.min(xyz, axis=0)   # 3D
        bbox_max = np.max(xyz, axis=0)   # 3D
        bbox_size = bbox_max - bbox_min  # 3D
        
        # Density (points per unit volume)
        volume = np.prod(bbox_size) if np.all(bbox_size > 0) else 1.0
        density = len(xyz) / volume  # 1D
        
        # PCA for shape orientation (using eigenvalues)
        centered = xyz - centroid
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)  # 3D
        
        # Concatenate: 3 + 3 + 3 + 1 + 3 = 13D (pad to 16 for alignment)
        features = np.concatenate([
            centroid,           # [0:3]
            bbox_size,          # [3:6]
            [density],          # [6]
            eigenvalues,        # [7:10]
            [len(xyz)],         # [10] point count
            np.zeros(5)         # [11:16] padding
        ])
        
        return features.astype(np.float32)
    
    except Exception as e:
        logger.warning(f"Failed to extract geometric features from {ply_path}: {e}")
        return np.zeros(16, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Hybrid Spatial-Semantic Embedder")
    parser.add_argument("--chunks_dir", required=True, help="Directory with PLY chunks")
    parser.add_argument("--renders_dir", required=True, help="Directory with rendered images")
    parser.add_argument("--manifest_path", required=True, help="Path to manifest.json")
    parser.add_argument("--db_path", default="hybrid_spatial_db", help="ChromaDB path")
    parser.add_argument("--collection_name", default="hybrid_chunks", help="Collection name")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32", help="CLIP model")
    parser.add_argument("--batch_size", type=int, default=16, help="CLIP batch size")
    parser.add_argument("--skip_semantic", action="store_true", help="Skip CLIP, use only geometric")
    args = parser.parse_args()

    # ====================================================================
    # PATH VALIDATION (Critical Gap #3 Fix)
    # ====================================================================
    
    # Convert to absolute paths
    chunks_dir = Path(args.chunks_dir).resolve()
    renders_dir = Path(args.renders_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    db_path = Path(args.db_path).resolve()
    
    # Validate input paths exist
    if not chunks_dir.exists():
        logger.error(f"Chunks directory does not exist: {chunks_dir}")
        sys.exit(1)
    
    if not chunks_dir.is_dir():
        logger.error(f"Chunks path is not a directory: {chunks_dir}")
        sys.exit(1)
    
    if not manifest_path.exists():
        logger.error(f"Manifest file does not exist: {manifest_path}")
        sys.exit(1)
    
    if not manifest_path.is_file():
        logger.error(f"Manifest path is not a file: {manifest_path}")
        sys.exit(1)
    
    # Validate renders directory
    if not renders_dir.exists():
        logger.warning(f"Renders directory does not exist: {renders_dir}")
        if not args.skip_semantic:
            logger.error("Renders required for semantic embedding. Either create renders or use --skip_semantic")
            sys.exit(1)
    
    # Create db_path parent if needed
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create database directory {db_path.parent}: {e}")
        sys.exit(1)
    
    logger.info("✅ Path validation passed")
    logger.info(f"   Chunks dir: {chunks_dir}")
    logger.info(f"   Renders dir: {renders_dir}")
    logger.info(f"   Manifest: {manifest_path}")
    logger.info(f"   Database: {db_path}")
    
    # Load dependencies
    try:
        import chromadb
        if not args.skip_semantic:
            from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        logger.error("Install: pip install chromadb transformers")
        sys.exit(1)

    # Load and validate manifest
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in manifest file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to read manifest: {e}")
        sys.exit(1)
    
    # Validate manifest structure
    if "chunks" not in manifest:
        logger.error("Manifest missing 'chunks' field")
        sys.exit(1)
    
    chunks_info = manifest["chunks"]
    
    if not chunks_info:
        logger.warning("Manifest contains no chunks")
        sys.exit(0)  # Exit gracefully
    
    logger.info(f"Processing {len(chunks_info)} chunks")
    
    # Validate chunk files exist
    missing_chunks = []
    for chunk_id, info in chunks_info.items():
        if "file" not in info:
            logger.warning(f"Chunk {chunk_id} missing 'file' field in manifest")
            continue
        chunk_path = chunks_dir / info["file"]
        if not chunk_path.exists():
            missing_chunks.append(chunk_id)
    
    if missing_chunks:
        logger.warning(f"Found {len(missing_chunks)} chunks in manifest but files missing: {missing_chunks[:5]}...")
        if len(missing_chunks) == len(chunks_info):
            logger.error("All chunk files are missing!")
            sys.exit(1)

    # Initialize CLIP (if needed)
    clip_model = None
    clip_processor = None
    
    if not args.skip_semantic:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                if free_mem < 2.0:
                    logger.warning(f"Low VRAM ({free_mem:.1f}GB), using CPU")
                    device = "cpu"
            except:
                pass
        
        logger.info(f"Loading CLIP on {device}...")
        clip_model = CLIPModel.from_pretrained(args.model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(args.model_name)

    # Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at {args.db_path}")
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection_name)

    # Process chunks
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    # Collect image paths for batching
    if not args.skip_semantic:
        image_tasks: List[Tuple[str, Path]] = []
        for chunk_id in chunks_info.keys():
            render_dir = renders_dir / chunk_id
            if render_dir.exists():
                for img in render_dir.glob("*.png"):
                    image_tasks.append((chunk_id, img))
        
        logger.info(f"Found {len(image_tasks)} images")
        
        # Batch process CLIP
        chunk_clip_embeds: Dict[str, List[np.ndarray]] = {}
        num_batches = (len(image_tasks) + args.batch_size - 1) // args.batch_size
        
        for i in tqdm(range(num_batches), desc="CLIP Embedding"):
            start = i * args.batch_size
            end = min((i + 1) * args.batch_size, len(image_tasks))
            batch = image_tasks[start:end]
            
            images = []
            valid_idx = []
            for idx, (cid, p) in enumerate(batch):
                try:
                    images.append(Image.open(p))
                    valid_idx.append(idx)
                except:
                    pass
            
            if not images:
                continue
            
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy()
            
            for local_idx, global_idx in enumerate(valid_idx):
                chunk_id = batch[global_idx][0]
                if chunk_id not in chunk_clip_embeds:
                    chunk_clip_embeds[chunk_id] = []
                chunk_clip_embeds[chunk_id].append(feats_np[local_idx])
    
    # Combine geometric + semantic
    logger.info("Combining features...")
    for chunk_id, info in tqdm(chunks_info.items(), desc="Processing chunks"):
        # 1. Geometric features
        ply_path = chunks_dir / info["file"]
        geo_feats = extract_geometric_features(ply_path)  # 16D
        
        # 2. Semantic features (if available)
        if args.skip_semantic or chunk_id not in chunk_clip_embeds:
            # Use zeros if no CLIP
            sem_feats = np.zeros(512, dtype=np.float32)  # CLIP default is 512D
        else:
            # Average CLIP embeddings
            clip_list = chunk_clip_embeds[chunk_id]
            sem_feats = np.mean(np.stack(clip_list), axis=0)
            sem_feats = sem_feats / np.linalg.norm(sem_feats)
        
        # 3. Concatenate: [16D geo | 512D semantic] = 528D hybrid
        hybrid_embedding = np.concatenate([geo_feats, sem_feats])
        
        # 4. Metadata
        bbox = info.get("bbox", {"min": [0,0,0], "max": [0,0,0]})
        meta = {
            "file": info.get("file", ""),
            "points": info.get("points", 0),
            "depth": info.get("depth", -1),
            "centroid_x": float(geo_feats[0]),
            "centroid_y": float(geo_feats[1]),
            "centroid_z": float(geo_feats[2]),
            "density": float(geo_feats[6]),
            "bbox_min_x": bbox["min"][0],
            "bbox_min_y": bbox["min"][1],
            "bbox_min_z": bbox["min"][2],
            "bbox_max_x": bbox["max"][0],
            "bbox_max_y": bbox["max"][1],
            "bbox_max_z": bbox["max"][2],
        }
        
        ids.append(chunk_id)
        embeddings.append(hybrid_embedding.tolist())
        metadatas.append(meta)
        documents.append(f"3D chunk {chunk_id} at ({meta['centroid_x']:.1f}, {meta['centroid_y']:.1f}, {meta['centroid_z']:.1f})")
    
    # Upsert
    if ids:
        logger.info(f"Upserting {len(ids)} hybrid embeddings...")
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc="Upserting"):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                documents=documents[i:end]
            )
        logger.info("✅ Hybrid indexing complete!")
        logger.info(f"   Embedding dim: {len(embeddings[0])} (16 geo + 512 semantic)")

if __name__ == "__main__":
    main()
