#!/usr/bin/env python3
"""Perform Hierarchical Agglomerative Clustering (HAC) on thought embeddings."""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

try:
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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


def perform_hac(
    embeddings: np.ndarray,
    n_clusters: int | None = None,
    linkage: str = "ward",
    distance_threshold: float | None = None,
    affinity: str = "euclidean",
) -> tuple[AgglomerativeClustering, np.ndarray]:
    """Perform Hierarchical Agglomerative Clustering on embeddings.
    
    Args:
        embeddings: Embedding vectors, shape (n_samples, n_features)
        n_clusters: Number of clusters (if None, use distance_threshold)
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        distance_threshold: Distance threshold for cutting the tree (if None, use n_clusters)
        affinity: Distance metric ('euclidean', 'l1', 'l2', 'manhattan', 'cosine')
        
    Returns:
        Tuple of (fitted AgglomerativeClustering model, cluster labels)
    """
    if linkage == "ward" and affinity != "euclidean":
        raise ValueError("Ward linkage can only be used with euclidean affinity")
    
    # Newer sklearn versions use 'metric' instead of 'affinity'
    clustering_kwargs = {
        "n_clusters": n_clusters,
        "linkage": linkage,
        "distance_threshold": distance_threshold,
    }
    
    # Try 'metric' first (newer sklearn), fall back to 'affinity' (older sklearn)
    try:
        clustering = AgglomerativeClustering(**clustering_kwargs, metric=affinity)
    except TypeError:
        clustering = AgglomerativeClustering(**clustering_kwargs, affinity=affinity)
    
    labels = clustering.fit_predict(embeddings)
    
    return clustering, labels


def plot_dendrogram(
    embeddings: np.ndarray,
    linkage_method: str = "ward",
    affinity: str = "euclidean",
    output_path: Path | None = None,
    max_d: float | None = None,
):
    """Plot dendrogram for hierarchical clustering.
    
    Args:
        embeddings: Embedding vectors, shape (n_samples, n_features)
        linkage_method: Linkage criterion ('ward', 'complete', 'average', 'single')
        affinity: Distance metric ('euclidean', 'l1', 'l2', 'manhattan', 'cosine')
        output_path: Optional path to save the dendrogram
        max_d: Maximum distance to show in dendrogram (for cutting visualization)
    """
    if not SCIPY_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("SciPy or Matplotlib not available, skipping dendrogram")
        return
    
    # Calculate distance matrix
    if affinity == "cosine":
        from sklearn.metrics.pairwise import cosine_distances
        dist_matrix = cosine_distances(embeddings)
        # Convert square distance matrix to condensed form
        condensed_distances = dist_matrix[np.triu_indices(len(embeddings), k=1)]
    else:
        condensed_distances = pdist(embeddings, metric=affinity)
    
    # Perform linkage
    linkage_matrix = linkage(condensed_distances, method=linkage_method)
    
    # Determine color threshold
    if max_d is not None:
        color_threshold = max_d
    else:
        color_threshold = 0.7 * max(linkage_matrix[:, 2])
    
    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram_kwargs = {
        "leaf_rotation": 90,
        "leaf_font_size": 8,
        "color_threshold": color_threshold,
    }
    if len(embeddings) > 30:
        dendrogram_kwargs["truncate_mode"] = "level"
        dendrogram_kwargs["p"] = 10
    
    dendrogram(linkage_matrix, **dendrogram_kwargs)
    plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage, {affinity} distance)')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    if max_d:
        plt.axhline(y=max_d, c='k', linestyle='--', label=f'Distance threshold: {max_d:.3f}')
        plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Dendrogram saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


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
            title = f"HAC Clustering of Thought Embeddings (UMAP visualization, n_neighbors={n_neighbors})"
    else:  # PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        xlabel = f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})'
        ylabel = f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})'
        title = 'HAC Clustering of Thought Embeddings (PCA visualization)'
    
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
    """Main function to run HAC clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform Hierarchical Agglomerative Clustering on thought embeddings")
    parser.add_argument(
        "embedding_file",
        type=str,
        help="Path to thought_embedding.json file",
    )
    parser.add_argument(
        "-k", "--n-clusters",
        type=int,
        help="Number of clusters (if not specified, use --distance-threshold)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        help="Distance threshold for cutting the tree (alternative to --n-clusters)",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        choices=["ward", "complete", "average", "single"],
        default="ward",
        help="Linkage criterion (default: ward)",
    )
    parser.add_argument(
        "--affinity",
        type=str,
        choices=["euclidean", "l1", "l2", "manhattan", "cosine"],
        default="euclidean",
        help="Distance metric (default: euclidean). Note: ward linkage requires euclidean",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (not used in HAC but kept for consistency)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSON file path for cluster assignments (default: hac_clusters.json in same directory)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of clusters (requires matplotlib)",
    )
    parser.add_argument(
        "--viz-output",
        type=str,
        help="Path to save visualization (default: hac_clusters.png in same directory)",
    )
    parser.add_argument(
        "--viz-method",
        type=str,
        choices=["pca", "umap"],
        default="pca",
        help="Dimensionality reduction method for visualization (default: pca)",
    )
    parser.add_argument(
        "--dendrogram",
        action="store_true",
        help="Create dendrogram visualization (requires scipy)",
    )
    parser.add_argument(
        "--dendrogram-output",
        type=str,
        help="Path to save dendrogram (default: dendrogram.png in same directory)",
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
    
    if args.n_clusters is None and args.distance_threshold is None:
        parser.error("Either --n-clusters or --distance-threshold must be specified")
    
    if args.linkage == "ward" and args.affinity != "euclidean":
        parser.error("Ward linkage can only be used with euclidean affinity")
    
    embedding_path = Path(args.embedding_file)
    
    print(f"Loading embeddings from: {embedding_path}")
    embedding_data, embeddings = load_embeddings(embedding_path)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    print(f"\nPerforming HAC clustering...")
    print(f"  Linkage: {args.linkage}")
    print(f"  Affinity: {args.affinity}")
    if args.n_clusters:
        print(f"  Number of clusters: {args.n_clusters}")
    else:
        print(f"  Distance threshold: {args.distance_threshold}")
    
    clustering, labels = perform_hac(
        embeddings,
        n_clusters=args.n_clusters,
        linkage=args.linkage,
        distance_threshold=args.distance_threshold,
        affinity=args.affinity,
    )
    
    n_clusters_found = len(np.unique(labels))
    print(f"\nFound {n_clusters_found} clusters")
    
    # Calculate silhouette score
    if n_clusters_found > 1:
        silhouette_avg = silhouette_score(embeddings, labels)
        print(f"Silhouette score: {silhouette_avg:.4f}")
    else:
        silhouette_avg = None
        print("Silhouette score: N/A (only 1 cluster)")
    
    # Print cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} thoughts ({count/len(labels)*100:.1f}%)")
    
    # Save cluster assignments
    output_path = Path(args.output) if args.output else embedding_path.parent / "hac_clusters.json"
    
    # Calculate distances to cluster centers (for each cluster, use mean embedding)
    cluster_centers = {}
    for cluster_id in unique:
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_centers[cluster_id] = cluster_embeddings.mean(axis=0)
    
    cluster_data = [
        {
            "id": item["id"],
            "cluster": int(label),
            "distance_to_center": float(np.linalg.norm(item["embedding"] - cluster_centers[label])),
        }
        for item, label in zip(embedding_data, labels)
    ]
    
    output_json = {
        "n_clusters": n_clusters_found,
        "n_samples": len(embeddings),
        "linkage": args.linkage,
        "affinity": args.affinity,
        "n_clusters_specified": args.n_clusters,
        "distance_threshold": args.distance_threshold,
    }
    
    if silhouette_avg is not None:
        output_json["silhouette_score"] = float(silhouette_avg)
    
    output_json["cluster_distribution"] = {int(k): int(v) for k, v in zip(unique, counts)}
    output_json["assignments"] = cluster_data
    
    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\nCluster assignments saved to: {output_path}")
    
    # Create dendrogram if requested
    if args.dendrogram:
        if not SCIPY_AVAILABLE:
            print("Warning: SciPy not available. Install with: pip install scipy")
        else:
            dendrogram_path = Path(args.dendrogram_output) if args.dendrogram_output else embedding_path.parent / "dendrogram.png"
            max_d = args.distance_threshold if args.distance_threshold else None
            plot_dendrogram(embeddings, args.linkage, args.affinity, dendrogram_path, max_d)
    
    # Visualize if requested
    if args.visualize:
        viz_output = Path(args.viz_output) if args.viz_output else embedding_path.parent / "hac_clusters.png"
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

