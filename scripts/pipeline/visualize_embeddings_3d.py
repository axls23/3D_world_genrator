#!/usr/bin/env python3
"""
Interactive 3D Visualization of Hybrid RAG Embedding Space
Visualize 528D vectors in 3D abstract space using dimensionality reduction
"""

import argparse
import sys
from pathlib import Path
import numpy as np

def load_chromadb_data(db_path: str, collection_name: str):
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
        print(f"Error: Collection '{collection_name}' not found")
        sys.exit(1)
    
    results = collection.get(include=['embeddings', 'metadatas', 'documents'])
    
    return {
        'ids': results['ids'],
        'embeddings': np.array(results['embeddings']),
        'metadatas': results['metadatas'],
        'documents': results['documents']
    }

def create_interactive_3d_visualization(data, method='umap', output_path=None):
    """Create interactive 3D visualization using Plotly"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("Error: Install plotly: pip install plotly")
        sys.exit(1)
    
    embeddings = data['embeddings']
    ids = data['ids']
    metadatas = data['metadatas']
    
    # Extract metadata for visualization
    point_counts = np.array([m['points'] for m in metadatas])
    centroids_x = np.array([m['centroid_x'] for m in metadatas])
    centroids_y = np.array([m['centroid_y'] for m in metadatas])
    centroids_z = np.array([m['centroid_z'] for m in metadatas])
    densities = np.array([m.get('density', 0) for m in metadatas])
    
    print(f"\n🔄 Reducing {embeddings.shape[1]}D embeddings to 3D using {method.upper()}...")
    
    # Dimensionality reduction to 3D
    if method == 'pca':
        try:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3, random_state=42)
            embeddings_3d = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_
            print(f"   Explained variance: {explained_var.sum()*100:.1f}%")
        except ImportError:
            print("Error: Install scikit-learn: pip install scikit-learn")
            sys.exit(1)
    
    elif method == 'tsne':
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, len(embeddings) - 1)
            reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity)
            embeddings_3d = reducer.fit_transform(embeddings)
            print(f"   t-SNE complete (perplexity={perplexity})")
        except ImportError:
            print("Error: Install scikit-learn: pip install scikit-learn")
            sys.exit(1)
    
    elif method == 'umap':
        try:
            import umap
            n_neighbors = min(15, len(embeddings) - 1)
            reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=n_neighbors)
            embeddings_3d = reducer.fit_transform(embeddings)
            print(f"   UMAP complete (n_neighbors={n_neighbors})")
        except ImportError:
            print("Warning: UMAP not available. Install with: pip install umap-learn")
            print("Falling back to PCA...")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3, random_state=42)
            embeddings_3d = reducer.fit_transform(embeddings)
    
    else:
        print(f"Unknown method: {method}. Using PCA.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3, random_state=42)
        embeddings_3d = reducer.fit_transform(embeddings)
    
    # Create hover text with detailed information
    hover_texts = []
    for i, chunk_id in enumerate(ids):
        text = (
            f"<b>Chunk: {chunk_id}</b><br>"
            f"Points: {point_counts[i]:,}<br>"
            f"Density: {densities[i]:.2e}<br>"
            f"Centroid: ({centroids_x[i]:.1f}, {centroids_y[i]:.1f}, {centroids_z[i]:.1f})<br>"
            f"File: {metadatas[i].get('file', 'N/A')}"
        )
        hover_texts.append(text)
    
    # Create figure with multiple views
    print("\n🎨 Creating interactive 3D visualization...")
    
    # Main 3D scatter plot colored by point count
    fig = go.Figure()
    
    # Add trace colored by point count
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=point_counts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Point Count", x=1.05),
            line=dict(color='black', width=0.5)
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Chunks'
    ))
    
    # Add annotations for largest chunks
    top_5_idx = np.argsort(point_counts)[-5:]
    for idx in top_5_idx:
        fig.add_trace(go.Scatter3d(
            x=[embeddings_3d[idx, 0]],
            y=[embeddings_3d[idx, 1]],
            z=[embeddings_3d[idx, 2]],
            mode='text',
            text=[ids[idx]],
            textposition='top center',
            textfont=dict(size=10, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    method_name = method.upper()
    fig.update_layout(
        title=dict(
            text=f'<b>3D Embedding Space Visualization ({method_name})</b><br>'
                 f'<sub>528D Hybrid Embeddings (16D Geometric + 512D CLIP Semantic)</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title=f'{method_name} Dimension 1',
            yaxis_title=f'{method_name} Dimension 2',
            zaxis_title=f'{method_name} Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        width=1200,
        height=900,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Save or show
    if output_path:
        html_file = output_path / f'embedding_space_3d_{method}.html'
        fig.write_html(str(html_file))
        print(f"\n✅ Saved interactive visualization: {html_file}")
        print(f"\n🌐 Open in browser: file:///{html_file.resolve()}")
        return str(html_file)
    else:
        fig.show()
        return None

def create_multi_view_visualization(data, output_path=None):
    """Create visualization with all three methods side by side"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        print("Error: Install required packages: pip install plotly scikit-learn")
        sys.exit(1)
    
    embeddings = data['embeddings']
    ids = data['ids']
    metadatas = data['metadatas']
    point_counts = np.array([m['points'] for m in metadatas])
    
    print("\n🔄 Computing all reduction methods...")
    
    # PCA
    print("   Computing PCA...")
    pca = PCA(n_components=3, random_state=42)
    pca_3d = pca.fit_transform(embeddings)
    
    # t-SNE
    print("   Computing t-SNE...")
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    tsne_3d = tsne.fit_transform(embeddings)
    
    # Try UMAP
    try:
        import umap
        print("   Computing UMAP...")
        n_neighbors = min(15, len(embeddings) - 1)
        umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=n_neighbors)
        umap_3d = umap_reducer.fit_transform(embeddings)
        has_umap = True
    except ImportError:
        print("   UMAP not available, skipping...")
        has_umap = False
    
    # Create subplots
    cols = 3 if has_umap else 2
    titles = ['PCA', 't-SNE', 'UMAP'] if has_umap else ['PCA', 't-SNE']
    
    fig = make_subplots(
        rows=1, cols=cols,
        subplot_titles=titles,
        specs=[[{'type': 'scatter3d'}] * cols],
        horizontal_spacing=0.05
    )
    
    # Add PCA
    fig.add_trace(
        go.Scatter3d(
            x=pca_3d[:, 0], y=pca_3d[:, 1], z=pca_3d[:, 2],
            mode='markers',
            marker=dict(size=6, color=point_counts, colorscale='Viridis', showscale=False),
            text=ids,
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add t-SNE
    fig.add_trace(
        go.Scatter3d(
            x=tsne_3d[:, 0], y=tsne_3d[:, 1], z=tsne_3d[:, 2],
            mode='markers',
            marker=dict(size=6, color=point_counts, colorscale='Viridis', showscale=False),
            text=ids,
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add UMAP if available
    if has_umap:
        fig.add_trace(
            go.Scatter3d(
                x=umap_3d[:, 0], y=umap_3d[:, 1], z=umap_3d[:, 2],
                mode='markers',
                marker=dict(size=6, color=point_counts, colorscale='Viridis', showscale=True),
                text=ids,
                hovertemplate='<b>%{text}</b><extra></extra>',
                showlegend=False
            ),
            row=1, col=3
        )
    
    fig.update_layout(
        title=dict(
            text='<b>Embedding Space Comparison</b><br><sub>528D → 3D Reduction Methods</sub>',
            x=0.5,
            xanchor='center'
        ),
        width=1800,
        height=700,
        template='plotly_white'
    )
    
    if output_path:
        html_file = output_path / 'embedding_space_comparison.html'
        fig.write_html(str(html_file))
        print(f"\n✅ Saved comparison visualization: {html_file}")
        return str(html_file)
    else:
        fig.show()
        return None

def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Embedding Space Visualization")
    parser.add_argument("--db_path", required=True, help="Path to ChromaDB directory")
    parser.add_argument("--collection_name", default="test_chunks", help="Collection name")
    parser.add_argument("--method", default="umap", choices=['pca', 'tsne', 'umap'],
                       help="Dimensionality reduction method")
    parser.add_argument("--output_dir", help="Directory to save HTML visualization")
    parser.add_argument("--comparison", action="store_true", 
                       help="Create comparison of all methods")
    
    args = parser.parse_args()
    
    # Load data
    print(f"📚 Loading ChromaDB from: {args.db_path}")
    print(f"   Collection: {args.collection_name}")
    data = load_chromadb_data(args.db_path, args.collection_name)
    
    print(f"\n📊 Loaded {len(data['ids'])} chunks")
    print(f"   Embedding dimension: {data['embeddings'].shape[1]}")
    
    # Setup output directory
    output_path = None
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    if args.comparison:
        html_file = create_multi_view_visualization(data, output_path)
    else:
        html_file = create_interactive_3d_visualization(data, args.method, output_path)
    
    if html_file:
        print("\n" + "="*70)
        print("🎉 INTERACTIVE 3D VISUALIZATION READY!")
        print("="*70)
        print(f"\n📂 File: {html_file}")
        print(f"\n💡 Usage:")
        print(f"   - Rotate: Click and drag")
        print(f"   - Zoom: Scroll wheel")
        print(f"   - Hover: See chunk details")
        print(f"   - Reset: Double-click")
        print("\n" + "="*70)

if __name__ == "__main__":
    main()
