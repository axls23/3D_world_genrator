import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Hybrid Spatial-Semantic RAG Query")
    parser.add_argument("--db_path", default="hybrid_spatial_db", help="Path to ChromaDB")
    parser.add_argument("--collection_name", default="hybrid_chunks", help="Collection name")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32", help="CLIP model")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    
    # Query modes
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--text", help="Semantic text query (e.g., 'red car')")
    query_group.add_argument("--spatial", nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                            help="Spatial query by coordinates")
    query_group.add_argument("--hybrid", nargs='+', 
                            help="Hybrid: --hybrid X Y Z 'query text'")
    
    args = parser.parse_args()

    # Load dependencies
    try:
        import chromadb
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        logger.error("Install: pip install chromadb transformers")
        sys.exit(1)

    # Connect to DB
    client = chromadb.PersistentClient(path=args.db_path)
    try:
        collection = client.get_collection(name=args.collection_name)
    except ValueError:
        logger.error(f"Collection '{args.collection_name}' not found. Did you run hybrid_embedder.py?")
        sys.exit(1)

    # Initialize CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CLIP model...")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    # Construct query embedding
    query_embedding = None
    
    if args.text:
        # Pure semantic query
        logger.info(f"📝 Semantic Query: '{args.text}'")
        
        # Zero geometric features, full CLIP features
        geo_feats = np.zeros(16, dtype=np.float32)
        
        inputs = processor(text=[args.text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_feats = model.get_text_features(**inputs)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        sem_feats = text_feats.cpu().numpy()[0]
        
        query_embedding = np.concatenate([geo_feats, sem_feats])
    
    elif args.spatial:
        # Pure spatial query
        x, y, z = args.spatial
        logger.info(f"📍 Spatial Query: ({x:.1f}, {y:.1f}, {z:.1f})")
        
        # Construct geometric features matching centroid
        geo_feats = np.zeros(16, dtype=np.float32)
        geo_feats[0:3] = [x, y, z]  # Centroid
        
        # Zero semantic features
        sem_feats = np.zeros(512, dtype=np.float32)
        
        query_embedding = np.concatenate([geo_feats, sem_feats])
    
    elif args.hybrid:
        # Hybrid query: X Y Z + text
        try:
            x, y, z = float(args.hybrid[0]), float(args.hybrid[1]), float(args.hybrid[2])
            text_query = ' '.join(args.hybrid[3:])
        except:
            logger.error("Hybrid format: --hybrid X Y Z 'query text'")
            sys.exit(1)
        
        logger.info(f"🔀 Hybrid Query: ({x:.1f}, {y:.1f}, {z:.1f}) + '{text_query}'")
        
        # Geometric
        geo_feats = np.zeros(16, dtype=np.float32)
        geo_feats[0:3] = [x, y, z]
        
        # Semantic
        inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_feats = model.get_text_features(**inputs)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        sem_feats = text_feats.cpu().numpy()[0]
        
        query_embedding = np.concatenate([geo_feats, sem_feats])

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=args.k
    )

    # Display results
    logger.info(f"\n{'='*60}")
    logger.info(f"Top {args.k} Results:")
    logger.info(f"{'='*60}\n")
    
    for i, (chunk_id, dist, meta) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        logger.info(f"{i+1}. Chunk: {chunk_id} (Distance: {dist:.4f})")
        logger.info(f"   File: {meta['file']}")
        logger.info(f"   Points: {meta['points']}")
        logger.info(f"   Centroid: ({meta['centroid_x']:.1f}, {meta['centroid_y']:.1f}, {meta['centroid_z']:.1f})")
        logger.info(f"   Density: {meta['density']:.2e}")
        logger.info(f"   BBox: [{meta['bbox_min_x']:.1f}, {meta['bbox_min_y']:.1f}, {meta['bbox_min_z']:.1f}]")
        logger.info(f"         → [{meta['bbox_max_x']:.1f}, {meta['bbox_max_y']:.1f}, {meta['bbox_max_z']:.1f}]")
        logger.info(f"{'-'*60}\n")

if __name__ == "__main__":
    main()
