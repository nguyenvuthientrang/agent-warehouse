#!/usr/bin/env python3
"""Visualize causal cosine similarity matrix for thought embeddings."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_embeddings(embedding_path: Path) -> tuple[list[dict], np.ndarray, list[int]]:
    """Load embeddings from JSON file and sort by ID.
    
    Args:
        embedding_path: Path to thought_embedding.json file
        
    Returns:
        Tuple of (embedding_data sorted by ID, embeddings_array sorted by ID, sorted_ids)
    """
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    with open(embedding_path, 'r') as f:
        embedding_data = json.load(f)
    
    if not embedding_data:
        raise ValueError("No embeddings found in file")
    
    # Sort by ID to get causal/temporal order
    embedding_data_sorted = sorted(embedding_data, key=lambda x: x["id"])
    ids = [item["id"] for item in embedding_data_sorted]
    
    # Extract embeddings as numpy array in sorted order
    embeddings = np.array([item["embedding"] for item in embedding_data_sorted])
    
    return embedding_data_sorted, embeddings, ids


def calculate_causal_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Calculate causal cosine similarity matrix (only lower triangular - inverted causal direction).
    
    Args:
        embeddings: Embedding vectors sorted by ID, shape (n_samples, n_features)
        
    Returns:
        Causal similarity matrix where similarity[i,j] = similarity if i >= j, else 0
        This represents: thought j (earlier) can influence thought i (later) if j <= i
    """
    # Calculate full similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Make it causal (inverted): only lower triangular (thought j can influence thought i if j <= i)
    causal_matrix = np.tril(similarity_matrix, k=0)
    
    return causal_matrix


def visualize_causal_similarity(
    similarity_matrix: np.ndarray,
    ids: list[int],
    output_path: Path | None = None,
    title: str = "Causal Cosine Similarity Matrix",
    cmap: str = "viridis",
    figsize: tuple[int, int] = (12, 10),
):
    """Visualize causal cosine similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: Causal similarity matrix (upper triangular)
        ids: List of thought IDs in order
        output_path: Optional path to save the visualization
        title: Title for the plot
        cmap: Colormap to use
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    n = len(ids)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ids, rotation=45, ha='right')
    ax.set_yticklabels(ids)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    # Add labels
    ax.set_xlabel('Thought ID (earlier in conversation)', fontsize=12)
    ax.set_ylabel('Thought ID (later in conversation)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Add text annotations for high similarity values (optional, can be slow for large matrices)
    if n <= 50:  # Only annotate for smaller matrices
        for i in range(n):
            for j in range(i + 1):  # Only lower triangular
                if similarity_matrix[i, j] > 0.7:  # Only show high similarities
                    text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=6)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to visualize causal cosine similarity matrix."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize causal cosine similarity matrix from thought embeddings")
    parser.add_argument(
        "embedding_file",
        type=str,
        help="Path to thought_embedding.json file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output image file path (default: causal_similarity_matrix.png in same directory)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap to use (default: viridis). Options: viridis, plasma, coolwarm, RdYlBu, etc.",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,10",
        help="Figure size as 'width,height' (default: 12,10)",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Custom title for the plot",
    )
    parser.add_argument(
        "--save-matrix",
        action="store_true",
        help="Also save the similarity matrix as JSON",
    )
    
    args = parser.parse_args()
    
    embedding_path = Path(args.embedding_file)
    
    print(f"Loading embeddings from: {embedding_path}")
    embedding_data, embeddings, ids = load_embeddings(embedding_path)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    print(f"Thought IDs (in order): {ids[:10]}{'...' if len(ids) > 10 else ''}")
    
    print("\nCalculating causal cosine similarity matrix...")
    similarity_matrix = calculate_causal_similarity_matrix(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    print(f"Mean similarity (causal): {similarity_matrix[similarity_matrix > 0].mean():.4f}")
    
    # Parse figure size
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
    except ValueError:
        print(f"Warning: Invalid figsize '{args.figsize}', using default (12, 10)")
        figsize = (12, 10)
    
    # Generate output path
    output_path = Path(args.output) if args.output else embedding_path.parent / "causal_similarity_matrix.png"
    
    title = args.title if args.title else f"Causal Cosine Similarity Matrix ({len(ids)} thoughts)"
    
    print("\nCreating visualization...")
    visualize_causal_similarity(
        similarity_matrix,
        ids,
        output_path,
        title=title,
        cmap=args.cmap,
        figsize=figsize,
    )
    
    # Save matrix if requested
    if args.save_matrix:
        matrix_path = embedding_path.parent / "causal_similarity_matrix.json"
        with open(matrix_path, 'w') as f:
            json.dump({
                "ids": ids,
                "similarity_matrix": similarity_matrix.tolist(),
                "statistics": {
                    "min": float(similarity_matrix.min()),
                    "max": float(similarity_matrix.max()),
                    "mean": float(similarity_matrix[similarity_matrix > 0].mean()),
                    "std": float(similarity_matrix[similarity_matrix > 0].std()),
                },
            }, f, indent=2)
        print(f"Similarity matrix saved to: {matrix_path}")
    
    # Print some statistics
    print("\nCausal similarity statistics:")
    print(f"  Self-similarity (diagonal): mean = {np.diag(similarity_matrix).mean():.4f}")
    
    # Calculate similarity between consecutive thoughts
    consecutive_similarities = []
    for i in range(len(ids) - 1):
        # For lower triangular: similarity[i+1, i] shows how thought i influences thought i+1
        consecutive_similarities.append(similarity_matrix[i+1, i])
    if consecutive_similarities:
        print(f"  Consecutive thoughts: mean = {np.mean(consecutive_similarities):.4f}, "
              f"min = {np.min(consecutive_similarities):.4f}, "
              f"max = {np.max(consecutive_similarities):.4f}")
    
    # Find most similar non-consecutive pairs
    masked_matrix = similarity_matrix.copy()
    # Mask diagonal and consecutive pairs
    np.fill_diagonal(masked_matrix, -np.inf)
    for i in range(len(ids) - 1):
        masked_matrix[i+1, i] = -np.inf  # Mask consecutive pairs
    
    max_idx = np.unravel_index(np.argmax(masked_matrix), masked_matrix.shape)
    max_sim = masked_matrix[max_idx]
    if max_sim > -np.inf:
        print(f"\nMost similar non-consecutive pair:")
        print(f"  Thought {ids[max_idx[1]]} -> Thought {ids[max_idx[0]]}: similarity = {max_sim:.4f}")


if __name__ == "__main__":
    main()

