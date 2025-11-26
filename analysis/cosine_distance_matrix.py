#!/usr/bin/env python3
"""Calculate cosine distance matrix from thought embeddings."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


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


def calculate_cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Calculate pairwise cosine distance matrix.
    
    Args:
        embeddings: Embedding vectors, shape (n_samples, n_features)
        
    Returns:
        Cosine distance matrix, shape (n_samples, n_samples)
        Distance is in range [0, 2] where 0 = identical, 2 = opposite
    """
    # Cosine distance = 1 - cosine similarity
    distance_matrix = cosine_distances(embeddings)
    return distance_matrix


def calculate_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Calculate pairwise cosine similarity matrix.
    
    Args:
        embeddings: Embedding vectors, shape (n_samples, n_features)
        
    Returns:
        Cosine similarity matrix, shape (n_samples, n_samples)
        Similarity is in range [-1, 1] where 1 = identical, -1 = opposite
    """
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def main():
    """Main function to calculate cosine distance matrix."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate cosine distance matrix from thought embeddings")
    parser.add_argument(
        "embedding_file",
        type=str,
        help="Path to thought_embedding.json file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSON file path (default: cosine_distance_matrix.json in same directory)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "npy"],
        default="json",
        help="Output format: json (human-readable) or npy (numpy binary) (default: json)",
    )
    parser.add_argument(
        "--include-similarity",
        action="store_true",
        help="Also calculate and save cosine similarity matrix",
    )
    parser.add_argument(
        "--include-ids",
        action="store_true",
        help="Include thought IDs in the output JSON",
    )
    
    args = parser.parse_args()
    
    embedding_path = Path(args.embedding_file)
    
    print(f"Loading embeddings from: {embedding_path}")
    embedding_data, embeddings = load_embeddings(embedding_path)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    print("\nCalculating cosine distance matrix...")
    distance_matrix = calculate_cosine_distance_matrix(embeddings)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")
    print(f"Mean distance: {distance_matrix.mean():.4f}")
    print(f"Median distance: {np.median(distance_matrix):.4f}")
    
    # Save distance matrix
    if args.output_format == "json":
        output_path = Path(args.output) if args.output else embedding_path.parent / "cosine_distance_matrix.json"
        
        output_data = {
            "n_samples": len(embeddings),
            "distance_matrix": distance_matrix.tolist(),
            "statistics": {
                "min": float(distance_matrix.min()),
                "max": float(distance_matrix.max()),
                "mean": float(distance_matrix.mean()),
                "median": float(np.median(distance_matrix)),
                "std": float(distance_matrix.std()),
            },
        }
        
        if args.include_ids:
            output_data["ids"] = [item["id"] for item in embedding_data]
        
        if args.include_similarity:
            print("\nCalculating cosine similarity matrix...")
            similarity_matrix = calculate_cosine_similarity_matrix(embeddings)
            output_data["similarity_matrix"] = similarity_matrix.tolist()
            output_data["similarity_statistics"] = {
                "min": float(similarity_matrix.min()),
                "max": float(similarity_matrix.max()),
                "mean": float(similarity_matrix.mean()),
                "median": float(np.median(similarity_matrix)),
                "std": float(similarity_matrix.std()),
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDistance matrix saved to: {output_path}")
        
    else:  # npy format
        output_path = Path(args.output) if args.output else embedding_path.parent / "cosine_distance_matrix.npy"
        np.save(output_path, distance_matrix)
        print(f"\nDistance matrix saved to: {output_path}")
        
        if args.include_similarity:
            similarity_matrix = calculate_cosine_similarity_matrix(embeddings)
            similarity_path = embedding_path.parent / "cosine_similarity_matrix.npy"
            np.save(similarity_path, similarity_matrix)
            print(f"Similarity matrix saved to: {similarity_path}")
    
    # Print some statistics about closest and farthest pairs
    print("\nClosest pairs (excluding diagonal):")
    # Mask diagonal (self-distances)
    masked_matrix = distance_matrix.copy()
    np.fill_diagonal(masked_matrix, np.inf)
    min_indices = np.unravel_index(np.argmin(masked_matrix), masked_matrix.shape)
    min_distance = masked_matrix[min_indices]
    if args.include_ids:
        id1 = embedding_data[min_indices[0]]["id"]
        id2 = embedding_data[min_indices[1]]["id"]
        print(f"  IDs {id1} and {id2}: distance = {min_distance:.4f}")
    else:
        print(f"  Indices {min_indices[0]} and {min_indices[1]}: distance = {min_distance:.4f}")
    
    print("\nFarthest pairs:")
    # For farthest, mask diagonal with -inf instead
    masked_matrix_max = distance_matrix.copy()
    np.fill_diagonal(masked_matrix_max, -np.inf)
    max_indices = np.unravel_index(np.argmax(masked_matrix_max), masked_matrix_max.shape)
    max_distance = masked_matrix_max[max_indices]
    if args.include_ids:
        id1 = embedding_data[max_indices[0]]["id"]
        id2 = embedding_data[max_indices[1]]["id"]
        print(f"  IDs {id1} and {id2}: distance = {max_distance:.4f}")
    else:
        print(f"  Indices {max_indices[0]} and {max_indices[1]}: distance = {max_distance:.4f}")


if __name__ == "__main__":
    main()

