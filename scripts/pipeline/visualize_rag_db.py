#!/usr/bin/env python3
"""
Visualize ChromaDB Hybrid RAG Database
Creates interactive visualizations of spatial and semantic embeddings
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_chromadb_data(db_path: str, collection_name: str) -> Dict:
    """Load all data from ChromaDB"""
    try:
        import chromadb
    except ImportError:
        print("Error: Install chromadb: pip install chromadb")
        sys.exit(1)
    
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        print(f"Error: Collection '{collection_name}' not found in database")
        print(f"Available collections: {[c.name for c in client.list_collections()]}")
        sys.exit(1)
    
    # Get all data
    results = collection.get(include=['embeddings', 'metadatas', 'documents'])
    
    return {
        'ids': results['ids'],
        'embeddings': np.array(results['embeddings']),
        'metadatas': results['metadatas'],
        'documents': results['documents']
    }

def extract_spatial_data(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Extract centroids and labels from metadata"""
    centroids = []
    labels = []
    densities = []
    point_counts = []
    
    for chunk_id, meta in zip(data['ids'], data['metadatas']):
        centroids.append([
            meta['centroid_x'],
            meta['centroid_y'],
            meta['centroid_z']
        ])
        labels.append(chunk_id)
        densities.append(meta.get('density', 0))
        point_counts.append(meta.get('points', 0))
    
    return np.array(centroids), labels, np.array(densities), np.array(point_counts)

def visualize_spatial_chunks_3d(centroids: np.ndarray, labels: List[str], 
                                densities: np.ndarray, point_counts: np.ndarray,
                                output_path: Path = None):
    """Create 3D visualization of chunk spatial distribution"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D scatter
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        centroids[:, 2],
        c=point_counts,
        s=100,
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    ax1.set_xlabel('X (mm)', fontsize=10)
    ax1.set_ylabel('Y (mm)', fontsize=10)
    ax1.set_zlabel('Z (mm)', fontsize=10)
    ax1.set_title('3D Spatial Distribution of Chunks\n(Color = Point Count)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Points', shrink=0.5)
    
    # Add labels for top 10 densest chunks
    top_indices = np.argsort(point_counts)[-10:]
    for idx in top_indices:
        ax1.text(centroids[idx, 0], centroids[idx, 1], centroids[idx, 2],
                labels[idx], fontsize=6, alpha=0.7)
    
    # XY projection
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(
        centroids[:, 0], 
        centroids[:, 1],
        c=point_counts,
        s=100,
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    ax2.set_xlabel('X (mm)', fontsize=10)
    ax2.set_ylabel('Y (mm)', fontsize=10)
    ax2.set_title('Top View (XY Projection)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Points')
    
    # XZ projection
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(
        centroids[:, 0], 
        centroids[:, 2],
        c=point_counts,
        s=100,
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    ax3.set_xlabel('X (mm)', fontsize=10)
    ax3.set_ylabel('Z (mm)', fontsize=10)
    ax3.set_title('Front View (XZ Projection)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Points')
    
    # YZ projection
    ax4 = fig.add_subplot(224)
    scatter4 = ax4.scatter(
        centroids[:, 1], 
        centroids[:, 2],
        c=point_counts,
        s=100,
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    ax4.set_xlabel('Y (mm)', fontsize=10)
    ax4.set_ylabel('Z (mm)', fontsize=10)
    ax4.set_title('Side View (YZ Projection)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='Points')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'spatial_distribution_3d.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path / 'spatial_distribution_3d.png'}")
    else:
        plt.show()
    
    plt.close()

def visualize_embedding_space(embeddings: np.ndarray, labels: List[str],
                              point_counts: np.ndarray, output_path: Path = None):
    """Visualize embedding space using t-SNE"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Warning: Install scikit-learn for t-SNE: pip install scikit-learn")
        return
    
    print("Computing t-SNE (this may take a moment)...")
    
    # Separate geometric and semantic embeddings
    geo_embeddings = embeddings[:, :16]
    sem_embeddings = embeddings[:, 16:]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # t-SNE on full hybrid embeddings
    tsne_full = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embedded_full = tsne_full.fit_transform(embeddings)
    
    ax = axes[0, 0]
    scatter = ax.scatter(embedded_full[:, 0], embedded_full[:, 1],
                        c=point_counts, s=100, cmap='plasma', alpha=0.7,
                        edgecolors='black', linewidth=0.5)
    ax.set_title('t-SNE: Full Hybrid Embeddings (528D)\n(Color = Point Count)', 
                fontsize=11, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter, ax=ax, label='Points')
    
    # Add labels for outliers
    top_indices = np.argsort(point_counts)[-5:]
    for idx in top_indices:
        ax.annotate(labels[idx], (embedded_full[idx, 0], embedded_full[idx, 1]),
                   fontsize=8, alpha=0.7)
    
    # t-SNE on geometric only
    if len(geo_embeddings) > 1:
        tsne_geo = TSNE(n_components=2, random_state=42, perplexity=min(30, len(geo_embeddings)-1))
        embedded_geo = tsne_geo.fit_transform(geo_embeddings)
        
        ax = axes[0, 1]
        scatter = ax.scatter(embedded_geo[:, 0], embedded_geo[:, 1],
                            c=point_counts, s=100, cmap='viridis', alpha=0.7,
                            edgecolors='black', linewidth=0.5)
        ax.set_title('t-SNE: Geometric Features Only (16D)\n(Color = Point Count)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, ax=ax, label='Points')
    
    # t-SNE on semantic only
    if len(sem_embeddings) > 1 and sem_embeddings.sum() != 0:
        tsne_sem = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sem_embeddings)-1))
        embedded_sem = tsne_sem.fit_transform(sem_embeddings)
        
        ax = axes[1, 0]
        scatter = ax.scatter(embedded_sem[:, 0], embedded_sem[:, 1],
                            c=point_counts, s=100, cmap='coolwarm', alpha=0.7,
                            edgecolors='black', linewidth=0.5)
        ax.set_title('t-SNE: Semantic Features Only (512D CLIP)\n(Color = Point Count)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, ax=ax, label='Points')
    
    # Distribution histogram
    ax = axes[1, 1]
    ax.hist(point_counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Point Count', fontsize=10)
    ax.set_ylabel('Number of Chunks', fontsize=10)
    ax.set_title('Distribution of Point Counts Across Chunks', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'embedding_space_tsne.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path / 'embedding_space_tsne.png'}")
    else:
        plt.show()
    
    plt.close()

def create_statistics_summary(data: Dict, centroids: np.ndarray, 
                              densities: np.ndarray, point_counts: np.ndarray,
                              output_path: Path = None):
    """Create statistical summary visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Point count distribution
    ax = axes[0, 0]
    ax.hist(point_counts, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(point_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(point_counts):.1f}')
    ax.axvline(np.median(point_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(point_counts):.1f}')
    ax.set_xlabel('Points per Chunk', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Point Count Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Density distribution
    ax = axes[0, 1]
    non_zero_densities = densities[densities > 0]
    if len(non_zero_densities) > 0:
        ax.hist(np.log10(non_zero_densities + 1e-10), bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Log10(Density)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Density Distribution (log scale)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No density data', ha='center', va='center', fontsize=12)
    
    # 3. Spatial extent
    ax = axes[0, 2]
    ranges = centroids.max(axis=0) - centroids.min(axis=0)
    ax.bar(['X', 'Y', 'Z'], ranges, color=['red', 'green', 'blue'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Range (mm)', fontsize=10)
    ax.set_title('Spatial Extent Along Each Axis', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Top 10 largest chunks
    ax = axes[1, 0]
    top_10_idx = np.argsort(point_counts)[-10:]
    top_labels = [data['ids'][i] for i in top_10_idx]
    top_counts = point_counts[top_10_idx]
    
    y_pos = np.arange(len(top_labels))
    ax.barh(y_pos, top_counts, color='teal', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=8)
    ax.set_xlabel('Point Count', fontsize=10)
    ax.set_title('Top 10 Largest Chunks', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Embedding statistics
    ax = axes[1, 1]
    embeddings = data['embeddings']
    geo_norms = np.linalg.norm(embeddings[:, :16], axis=1)
    sem_norms = np.linalg.norm(embeddings[:, 16:], axis=1)
    
    ax.scatter(geo_norms, sem_norms, s=100, alpha=0.6, c=point_counts, 
              cmap='viridis', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Geometric Embedding Norm (16D)', fontsize=10)
    ax.set_ylabel('Semantic Embedding Norm (512D)', fontsize=10)
    ax.set_title('Embedding Component Magnitudes', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    stats_text = f"""
DATABASE STATISTICS
{'='*40}

Total Chunks: {len(data['ids'])}
Embedding Dimension: {embeddings.shape[1]}

POINT COUNTS:
  Mean: {np.mean(point_counts):.1f}
  Median: {np.median(point_counts):.1f}
  Max: {np.max(point_counts)}
  Min: {np.min(point_counts)}
  Std Dev: {np.std(point_counts):.1f}

SPATIAL EXTENT:
  X Range: {ranges[0]:.1f} mm
  Y Range: {ranges[1]:.1f} mm
  Z Range: {ranges[2]:.1f} mm
  
CENTROIDS:
  Mean: ({centroids.mean(axis=0)[0]:.1f}, 
         {centroids.mean(axis=0)[1]:.1f}, 
         {centroids.mean(axis=0)[2]:.1f})

EMBEDDINGS:
  Geometric (16D): {embeddings[:, :16].shape}
  Semantic (512D): {embeddings[:, 16:].shape}
  Total: {embeddings.shape}
    """
    
    ax.text(0.1, 0.9, stats_text, fontsize=9, verticalalignment='top',
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'database_statistics.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path / 'database_statistics.png'}")
    else:
        plt.show()
    
    plt.close()

def print_text_summary(data: Dict, centroids: np.ndarray, 
                      densities: np.ndarray, point_counts: np.ndarray):
    """Print text-based summary to console"""
    
    print("\n" + "="*70)
    print("CHROMADB HYBRID RAG DATABASE SUMMARY")
    print("="*70)
    
    print(f"\n📊 GENERAL INFO:")
    print(f"   Total Chunks: {len(data['ids'])}")
    print(f"   Embedding Dimension: {data['embeddings'].shape[1]} (16 geo + 512 semantic)")
    print(f"   Database Size: {data['embeddings'].nbytes / 1024:.1f} KB (embeddings only)")
    
    print(f"\n📍 SPATIAL DISTRIBUTION:")
    print(f"   Centroid Range:")
    print(f"     X: [{centroids[:, 0].min():.1f}, {centroids[:, 0].max():.1f}] mm")
    print(f"     Y: [{centroids[:, 1].min():.1f}, {centroids[:, 1].max():.1f}] mm")
    print(f"     Z: [{centroids[:, 2].min():.1f}, {centroids[:, 2].max():.1f}] mm")
    print(f"   Mean Centroid: ({centroids.mean(axis=0)[0]:.1f}, {centroids.mean(axis=0)[1]:.1f}, {centroids.mean(axis=0)[2]:.1f})")
    
    print(f"\n📦 POINT STATISTICS:")
    print(f"   Total Points: {point_counts.sum()}")
    print(f"   Mean per Chunk: {point_counts.mean():.1f}")
    print(f"   Median per Chunk: {np.median(point_counts):.1f}")
    print(f"   Max per Chunk: {point_counts.max()}")
    print(f"   Min per Chunk: {point_counts.min()}")
    
    print(f"\n🏆 TOP 5 LARGEST CHUNKS:")
    top_5_idx = np.argsort(point_counts)[-5:][::-1]
    for rank, idx in enumerate(top_5_idx, 1):
        print(f"   {rank}. {data['ids'][idx]}: {point_counts[idx]} points")
    
    print(f"\n💾 EMBEDDINGS INFO:")
    geo_mean_norm = np.linalg.norm(data['embeddings'][:, :16], axis=1).mean()
    sem_mean_norm = np.linalg.norm(data['embeddings'][:, 16:], axis=1).mean()
    print(f"   Geometric Mean Norm: {geo_mean_norm:.2f}")
    print(f"   Semantic Mean Norm: {sem_mean_norm:.2f}")
    
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description="Visualize Hybrid RAG ChromaDB")
    parser.add_argument("--db_path", required=True, help="Path to ChromaDB directory")
    parser.add_argument("--collection_name", default="test_chunks", help="Collection name")
    parser.add_argument("--output_dir", help="Directory to save visualizations (optional)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.db_path}")
    print(f"Collection: {args.collection_name}")
    data = load_chromadb_data(args.db_path, args.collection_name)
    
    # Extract spatial data
    centroids, labels, densities, point_counts = extract_spatial_data(data)
    
    # Print text summary
    print_text_summary(data, centroids, densities, point_counts)
    
    # Setup output directory
    output_path = None
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Saving visualizations to: {output_path}")
    
    # Create visualizations
    print("\n🎨 Generating visualizations...")
    
    visualize_spatial_chunks_3d(centroids, labels, densities, point_counts, 
                               None if args.show else output_path)
    
    visualize_embedding_space(data['embeddings'], labels, point_counts,
                             None if args.show else output_path)
    
    create_statistics_summary(data, centroids, densities, point_counts,
                             None if args.show else output_path)
    
    if output_path and not args.show:
        print(f"\n✅ All visualizations saved to: {output_path}")
    
    print("\n✨ Visualization complete!")

if __name__ == "__main__":
    main()
