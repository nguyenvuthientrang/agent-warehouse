#!/usr/bin/env python3
"""Perform k-means clustering on thought embeddings."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def load_embeddings(embedding_path: Path) -> tuple[list[dict], np.ndarray]:
    """Load embeddings from JSON file.
    
    Args:
        embedding_path: Path to thought_embedding.json file
        
    Returns:
        Tuple of (embedding_data, embeddings_array) where embeddings_array is shape (n_samples, n_features)
    """
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    with open(embedding_path, 'r') as f:
        embedding_data = json.load(f)
    
    if not embedding_data:
        raise ValueError("No embeddings found in file")
    
    # Extract embeddings as numpy array
    embeddings = np.array([item["embedding"] for item in embedding_data])
    
    return embedding_data, embeddings


def perform_kmeans(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    """Perform k-means clustering on embeddings.
    
    Args:
        embeddings: Embedding vectors, shape (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (fitted KMeans model, cluster labels)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return kmeans, labels


def visualize_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path | None = None,
    method: str = "pca",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Visualize clusters using dimensionality reduction (PCA or UMAP).
    
    Args:
        embeddings: Embedding vectors, shape (n_samples, n_features)
        labels: Cluster labels, shape (n_samples,)
        output_path: Optional path to save the visualization
        method: Dimensionality reduction method ('pca' or 'umap')
        n_neighbors: Number of neighbors for UMAP (default: 15)
        min_dist: Minimum distance for UMAP (default: 0.1)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    # Reduce to 2D
    if method.lower() == "umap":
        if not UMAP_AVAILABLE:
            print("UMAP not available, falling back to PCA")
            method = "pca"
        else:
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
            xlabel = "UMAP 1"
            ylabel = "UMAP 2"
            title = f"K-means Clustering of Thought Embeddings (UMAP visualization, n_neighbors={n_neighbors})"
    else:  # PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        xlabel = f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})'
        ylabel = f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})'
        title = 'K-means Clustering of Thought Embeddings (PCA visualization)'
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to run k-means clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform k-means clustering on thought embeddings")
    parser.add_argument(
        "embedding_file",
        type=str,
        help="Path to thought_embedding.json file",
    )
    parser.add_argument(
        "-k", "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSON file path for cluster assignments (default: clusters.json in same directory)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of clusters (requires matplotlib)",
    )
    parser.add_argument(
        "--viz-output",
        type=str,
        help="Path to save visualization (default: clusters.png in same directory)",
    )
    parser.add_argument(
        "--viz-method",
        type=str,
        choices=["pca", "umap"],
        default="pca",
        help="Dimensionality reduction method for visualization (default: pca)",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP (default: 15)",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="Minimum distance for UMAP (default: 0.1)",
    )
    
    args = parser.parse_args()
    
    embedding_path = Path(args.embedding_file)
    
    print(f"Loading embeddings from: {embedding_path}")
    embedding_data, embeddings = load_embeddings(embedding_path)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    print(f"\nPerforming k-means clustering with k={args.n_clusters}...")
    kmeans, labels = perform_kmeans(embeddings, args.n_clusters, args.random_state)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(embeddings, labels)
    print(f"Silhouette score: {silhouette_avg:.4f}")
    
    # Print cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} thoughts ({count/len(labels)*100:.1f}%)")
    
    # Save cluster assignments
    output_path = Path(args.output) if args.output else embedding_path.parent / "clusters.json"
    cluster_data = [
        {
            "id": item["id"],
            "cluster": int(label),
            "distance_to_center": float(np.linalg.norm(item["embedding"] - kmeans.cluster_centers_[label])),
        }
        for item, label in zip(embedding_data, labels)
    ]
    
    with open(output_path, 'w') as f:
        json.dump({
            "n_clusters": args.n_clusters,
            "n_samples": len(embeddings),
            "silhouette_score": float(silhouette_avg),
            "cluster_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
            "assignments": cluster_data,
        }, f, indent=2)
    
    print(f"\nCluster assignments saved to: {output_path}")
    
    # Visualize if requested
    if args.visualize:
        viz_output = Path(args.viz_output) if args.viz_output else embedding_path.parent / "clusters.png"
        if args.viz_method == "umap" and not UMAP_AVAILABLE:
            print("Warning: UMAP not available. Install with: pip install umap-learn")
            print("Falling back to PCA visualization")
            args.viz_method = "pca"
        visualize_clusters(
            embeddings,
            labels,
            viz_output,
            method=args.viz_method,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )


if __name__ == "__main__":
    main()

