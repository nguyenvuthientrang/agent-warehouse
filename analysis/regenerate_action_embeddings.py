#!/usr/bin/env python3
"""Regenerate action embeddings from existing action.json files."""

import json
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Error: sentence-transformers is not available. Please install it first.")
    exit(1)


def regenerate_action_embeddings(
    action_json_path: Path,
    embedding_model_name: str = "nomic-ai/nomic-embed-code",
) -> None:
    """Regenerate action embeddings from action.json file.
    
    Args:
        action_json_path: Path to action.json file
        embedding_model_name: Name of the embedding model to use
    """
    if not action_json_path.exists():
        raise FileNotFoundError(f"Action file not found: {action_json_path}")
    
    # Load action messages
    with open(action_json_path, 'r') as f:
        action_messages = json.load(f)
    
    if not action_messages:
        print(f"No action messages found in {action_json_path}")
        return
    
    print(f"Loaded {len(action_messages)} action messages from {action_json_path}")
    
    # Generate embeddings
    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    
    texts = [msg["raw_text"] for msg in action_messages]
    print(f"Generating embeddings for {len(texts)} actions...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Create embedding data
    action_embeddings = [
        {"id": msg["id"], "embedding": embedding.tolist()}
        for msg, embedding in zip(action_messages, embeddings)
    ]
    
    # Save to action_embedding.json in the same directory
    output_path = action_json_path.parent / "action_embedding.json"
    with open(output_path, 'w') as f:
        json.dump(action_embeddings, f, indent=2)
    
    print(f"Saved {len(action_embeddings)} action embeddings to {output_path}")
    print(f"Embedding dimension: {len(action_embeddings[0]['embedding'])}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate action embeddings from action.json")
    parser.add_argument(
        "path",
        type=str,
        help="Path to directory containing action.json or path to action.json file itself",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nomic-ai/nomic-embed-code",
        help="Embedding model to use (default: nomic-ai/nomic-embed-code)",
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    # If it's a directory, look for action.json inside
    if path.is_dir():
        action_json_path = path / "action.json"
    elif path.name == "action.json":
        action_json_path = path
    else:
        raise ValueError(f"Path must be a directory or action.json file: {path}")
    
    regenerate_action_embeddings(action_json_path, args.model)


if __name__ == "__main__":
    main()

