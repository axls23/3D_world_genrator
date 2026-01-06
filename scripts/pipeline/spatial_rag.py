import argparse
import logging
import sys
from pathlib import Path

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Query the Spatial RAG database.")
    parser.add_argument("query", help="Text query (e.g., 'a red car')")
    parser.add_argument("--db_path", default="spatial_db", help="Path to ChromaDB")
    parser.add_argument("--collection_name", default="spatial_chunks", help="ChromaDB collection name")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--visualize", action="store_true", help="Launch viewer with results")
    parser.add_argument("--chunk_dir", help="Directory containing chunk PLY files (required for visualization)")
    args = parser.parse_args()

    # 1. Load Dependencies
    try:
        import chromadb
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        logger.error("Please install chromadb and transformers.")
        sys.exit(1)

    # 2. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CLIP model...")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    # 3. Connect to DB
    client = chromadb.PersistentClient(path=args.db_path)
    try:
        collection = client.get_collection(name=args.collection_name)
    except ValueError:
        logger.error(f"Collection {args.collection_name} not found. Did you run chunk_embedder.py?")
        sys.exit(1)

    # 4. Embed Query
    logger.info(f"Querying for: '{args.query}'")
    inputs = processor(text=[args.query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    query_embedding = text_features.cpu().numpy().tolist()

    # 5. Retrieve
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=args.k
    )

    # 6. Display Results
    logger.info(f"\nTop {args.k} Results:")
    logger.info("-" * 40)
    
    retrieved_ids = results['ids'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    
    chunk_files = []

    for i, (chunk_id, dist, meta) in enumerate(zip(retrieved_ids, distances, metadatas)):
        logger.info(f"{i+1}. Chunk: {chunk_id} (Dist: {dist:.4f})")
        logger.info(f"   File: {meta['file']}")
        logger.info(f"   Points: {meta['points']}")
        chunk_files.append(meta['file'])
        logger.info("-" * 40)

    # 7. Visualize
    if args.visualize:
        if not args.chunk_dir:
            logger.error("Please provide --chunk_dir to visualize results.")
            return
            
        chunk_dir = Path(args.chunk_dir)
        # Create a temporary manifest for the viewer
        temp_manifest = {
            "chunks": {}
        }
        
        for i, (chunk_id, meta) in enumerate(zip(retrieved_ids, metadatas)):
            temp_manifest["chunks"][chunk_id] = {
                "file": meta['file'],
                "points": meta['points'],
                # Reconstruct bbox for viewer if needed, but viewer mainly needs file
            }
            
        # We can just pass the list of files to the viewer if we modify it, 
        # or we can point the viewer to the original directory and filter?
        # The viewer I wrote supports loading from a directory with manifest.
        # Let's create a temp manifest file in the chunk dir (or temp dir)
        
        temp_manifest_path = chunk_dir / "retrieval_manifest.json"
        import json
        with open(temp_manifest_path, 'w') as f:
            json.dump(temp_manifest, f, indent=2)
            
        logger.info(f"Launching viewer for retrieved chunks...")
        
        # Construct command to run viewer
        # We need to use the same python env
        import subprocess
        
        # Assuming chunk_viewer.py is in the same scripts/pipeline dir
        viewer_script = Path(__file__).parent / "chunk_viewer.py"
        
        cmd = [sys.executable, str(viewer_script), 
               "--input_path", str(chunk_dir), # It will look for manifest.json by default, but we want retrieval_manifest.json
               # Wait, my viewer loads manifest.json by default if dir.
               # I should update viewer to accept manifest filename or just overwrite manifest.json?
               # Overwriting is bad.
               # Let's update viewer to look for specific manifest if provided?
               # Or just pass the list of files?
               # My viewer supports single file or dir.
               # Let's just print the command for now or try to hack it.
               ]
        
        # Actually, let's just tell the user how to view it, or update viewer to take a manifest path.
        # Updating viewer is cleaner.
        logger.info(f"To view results, run:")
        logger.info(f"python {viewer_script} --input_path {chunk_dir} --manifest {temp_manifest_path.name}")

if __name__ == "__main__":
    main()
