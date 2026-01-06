import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Embed rendered chunks using CLIP and store in ChromaDB (Optimized).")
    parser.add_argument("--renders_dir", required=True, help="Directory containing rendered images (subdirs by chunk_id)")
    parser.add_argument("--manifest_path", required=True, help="Path to the manifest.json of the chunks")
    parser.add_argument("--db_path", default="spatial_db", help="Path to store ChromaDB")
    parser.add_argument("--collection_name", default="spatial_chunks", help="ChromaDB collection name")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CLIP inference")
    args = parser.parse_args()

    renders_dir = Path(args.renders_dir)
    manifest_path = Path(args.manifest_path)
    
    # 1. Load Dependencies
    try:
        import chromadb
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        logger.error("Please install chromadb and transformers: pip install chromadb transformers")
        sys.exit(1)

    # 2. Load Manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    chunks_info = manifest["chunks"]
    logger.info(f"Loaded manifest with {len(chunks_info)} chunks.")

    # 3. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"Free VRAM: {free_mem:.2f} GB")
            if free_mem < 2.0:
                logger.warning("Low VRAM detected (<2GB). Switching to CPU.")
                device = "cpu"
        except:
            pass

    logger.info(f"Loading CLIP model {args.model_name} on {device}...")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    # 4. Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at {args.db_path}...")
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection_name)

    # 5. Collect all image paths
    # We flatten the structure: list of (chunk_id, image_path)
    logger.info("Scanning for images...")
    all_tasks: List[Tuple[str, Path]] = []
    
    # Vectorized scan? Not really possible with pathlib glob, but fast enough.
    # We can iterate over chunks_info keys which is safer than globbing everything
    for chunk_id in chunks_info.keys():
        chunk_dir = renders_dir / chunk_id
        if chunk_dir.exists():
            for img_path in chunk_dir.glob("*.png"):
                all_tasks.append((chunk_id, img_path))
    
    logger.info(f"Found {len(all_tasks)} images to process.")
    if not all_tasks:
        return

    # 6. Batch Processing
    # We will accumulate embeddings per chunk
    chunk_embeddings: Dict[str, List[np.ndarray]] = {}
    
    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(all_tasks) + batch_size - 1) // batch_size
    
    logger.info(f"Processing in {num_batches} batches (Batch Size: {batch_size})...")
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_tasks))
        batch_tasks = all_tasks[start_idx:end_idx]
        
        # Load images
        batch_images = []
        valid_indices = []
        for idx, (cid, p) in enumerate(batch_tasks):
            try:
                batch_images.append(Image.open(p))
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")

        if not batch_images:
            continue

        # Inference
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        
        # Normalize
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        features_np = features.cpu().numpy()
        
        # Distribute back to chunks
        for local_idx, global_idx in enumerate(valid_indices):
            chunk_id = batch_tasks[global_idx][0]
            if chunk_id not in chunk_embeddings:
                chunk_embeddings[chunk_id] = []
            chunk_embeddings[chunk_id].append(features_np[local_idx])

    # 7. Aggregate and Upsert
    logger.info("Aggregating and Upserting...")
    
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for chunk_id, feats_list in chunk_embeddings.items():
        if not feats_list:
            continue
        
        # Mean pooling of all views for this chunk
        feats_stack = np.stack(feats_list)
        avg_feat = np.mean(feats_stack, axis=0)
        # Normalize again after averaging
        avg_feat = avg_feat / np.linalg.norm(avg_feat)
        
        info = chunks_info.get(chunk_id, {})
        
        ids.append(chunk_id)
        embeddings.append(avg_feat.tolist())
        
        # Safe metadata access
        bbox = info.get("bbox", {"min": [0,0,0], "max": [0,0,0]})
        meta = {
            "file": info.get("file", ""),
            "points": info.get("points", 0),
            "depth": info.get("depth", -1),
            "bbox_min_x": bbox["min"][0],
            "bbox_min_y": bbox["min"][1],
            "bbox_min_z": bbox["min"][2],
            "bbox_max_x": bbox["max"][0],
            "bbox_max_y": bbox["max"][1],
            "bbox_max_z": bbox["max"][2],
        }
        metadatas.append(meta)
        documents.append(f"Spatial chunk {chunk_id}")

    if ids:
        # Batch upsert
        upsert_batch_size = 100
        total_upserts = (len(ids) + upsert_batch_size - 1) // upsert_batch_size
        for i in tqdm(range(total_upserts), desc="Upserting"):
            start = i * upsert_batch_size
            end = min((i + 1) * upsert_batch_size, len(ids))
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                documents=documents[start:end]
            )
        logger.info("✅ Indexing complete.")

if __name__ == "__main__":
    main()
