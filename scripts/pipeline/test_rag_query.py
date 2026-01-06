#!/usr/bin/env python3
"""
Test script to query the Hybrid RAG system
"""

import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from hybrid_rag import HybridSpatialRAG

def main():
    # Initialize RAG system
    db_path = Path(__file__).parent.parent.parent / "data" / "ADAPTIVE_TEST" / "hybrid_rag_db"
    
    print(f"Initializing Hybrid RAG system from: {db_path}")
    rag = HybridSpatialRAG(str(db_path), "test_chunks")
    
    # Test 1: Spatial query
    print("\n" + "="*70)
    print("TEST 1: Spatial Query (coordinate-based)")
    print("="*70)
    query_point = [0.0, 0.0, 0.0]
    results = rag.query_spatial(query_point, k=3)
    
    print(f"Query: Find 3 chunks near {query_point}")
    print(f"Results: {len(results)} chunks found")
    for i, result in enumerate(results):
        print(f"\n  {i+1}. Chunk: {result['chunk_id']}")
        print(f"     Distance: {result['distance']:.2f}")
        print(f"     Centroid: ({result['centroid_x']:.1f}, {result['centroid_y']:.1f}, {result['centroid_z']:.1f})")
        print(f"     Points: {result['points']}")
        print(f"     Density: {result['density']:.4f}")
    
    # Test 2: Semantic query
    print("\n" + "="*70)
    print("TEST 2: Semantic Query (text-based)")
    print("="*70)
    results = rag.query_semantic("center region", k=3)
    
    print(f"Query: 'center region'")
    print(f"Results: {len(results)} chunks found")
    for i, result in enumerate(results):
        print(f"\n  {i+1}. Chunk: {result['chunk_id']}")
        print(f"     Centroid: ({result['centroid_x']:.1f}, {result['centroid_y']:.1f}, {result['centroid_z']:.1f})")
        print(f"     Points: {result['points']}")
    
    # Test 3: Hybrid query
    print("\n" + "="*70)
    print("TEST 3: Hybrid Query (coordinate + text)")
    print("="*70)
    results = rag.query_hybrid(query_point, "dense area", k=3, spatial_weight=0.6)
    
    print(f"Query: Near {query_point} + 'dense area' (spatial weight: 0.6)")
    print(f"Results: {len(results)} chunks found")
    for i, result in enumerate(results):
        print(f"\n  {i+1}. Chunk: {result['chunk_id']}")
        print(f"     Centroid: ({result['centroid_x']:.1f}, {result['centroid_y']:.1f}, {result['centroid_z']:.1f})")
        print(f"     Points: {result['points']}")
        print(f"     Density: {result['density']:.4f}")
    
    print("\n" + "="*70)
    print("✅ All tests completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
