import dataclasses
import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from minisweagent import Agent, __version__

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


def _get_class_name_with_module(obj: Any) -> str:
    """Get the full class name with module path."""
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def _asdict(obj: Any) -> dict:
    """Convert config objects to dicts."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)  # type: ignore[arg-type]
    return obj  # let's try our luck


def save_traj(
    agent: Agent | None,
    path: Path | None,
    *,
    print_path: bool = True,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
    print_fct: Callable = print,
    embedding_model: str | None = None,
    **kwargs,
):
    """Save the trajectory of the agent to a file.

    Args:
        agent: The agent to save the trajectory of.
        path: The path to save the trajectory to.
        print_path: Whether to print confirmation of path to the terminal.
        exit_status: The exit status of the agent.
        result: The result/submission of the agent.
        extra_info: Extra information to save (will be merged into the info dict).
        embedding_model: Name of the embedding model to use for thought embeddings (default: BAAI/bge-large-en-v1.5)
        **kwargs: Additional information to save (will be merged into top level)

    """
    if path is None:
        return
    data = {
        "info": {
            "exit_status": exit_status,
            "submission": result,
            "model_stats": {
                "instance_cost": 0.0,
                "api_calls": 0,
            },
            "mini_version": __version__,
        },
        "messages": [],
        "trajectory_format": "mini-swe-agent-1",
    } | kwargs
    if agent is not None:
        data["info"]["model_stats"]["instance_cost"] = agent.model.cost
        data["info"]["model_stats"]["api_calls"] = agent.model.n_calls
        data["messages"] = agent.messages
        data["info"]["config"] = {
            "agent": _asdict(agent.config),
            "model": _asdict(agent.model.config),
            "environment": _asdict(agent.env.config),
            "agent_type": _get_class_name_with_module(agent),
            "model_type": _get_class_name_with_module(agent.model),
            "environment_type": _get_class_name_with_module(agent.env),
        }
    if extra_info:
        data["info"].update(extra_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    
    # Extract and save assistant/user messages to separate files
    if agent is not None and agent.messages:
        save_extracted_messages(agent.messages, path, embedding_model=embedding_model)
    
    if print_path:
        print_fct(f"Saved trajectory to '{path}'")


def get_log_path(
    run_type: str,
    model_name: str,
    instance_id: str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Generate a structured log path.
    
    Args:
        run_type: Either "swebench" or "mini"
        model_name: Name of the model being used
        instance_id: Instance ID (for swebench) or None (for mini)
        base_dir: Base directory for logs (defaults to current directory)
    
    Returns:
        Path to the log file in format: logs/{run_type}/{model}_{time}/{instance_id}/{instance_id}.traj.json
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    # Sanitize model name for filesystem
    safe_model_name = re.sub(r'[^\w\-_.]', '_', model_name.replace('/', '_'))
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build path
    if instance_id:
        safe_instance = re.sub(r'[^\w\-_.]', '_', instance_id)
        log_path = base_dir / "logs" / run_type / f"{safe_model_name}_{timestamp}" / safe_instance / f"{safe_instance}.traj.json"
    else:
        log_path = base_dir / "logs" / run_type / f"{safe_model_name}_{timestamp}" / "run.traj.json"
    
    return log_path


def _extract_thought_and_action(content: str) -> tuple[str, str]:
    """Extract thought and action sections from assistant message content.
    
    Args:
        content: Full message content from assistant
        
    Returns:
        Tuple of (thought_text, action_text)
    """
    thought_text = ""
    action_text = ""
    
    # Pattern to match code blocks: ```language\ncontent\n```
    code_block_pattern = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
    
    # Extract THOUGHT section - handle both formats:
    # 1. THOUGHT: prefix format
    # 2. <thought>...</thought> tag format
    thought_parts = []
    
    # Try THOUGHT: prefix format
    thought_match = re.search(r'THOUGHT:\s*(.*?)(?=```|</response>|$)', content, re.DOTALL)
    if thought_match:
        thought_parts.append(thought_match.group(1).strip())
    
    # Try <thought> tag format (can have multiple tags)
    thought_tags = re.findall(r'<thought>\s*(.*?)\s*</thought>', content, re.DOTALL)
    if thought_tags:
        thought_parts.extend([tag.strip() for tag in thought_tags])
    
    # Combine all thought parts
    if thought_parts:
        thought_text = "\n\n".join(thought_parts)
    
    # Extract all code blocks and actions
    code_blocks = code_block_pattern.findall(content)
    if code_blocks:
        action_parts = []
        for lang, code in code_blocks:
            lang_part = f"```{lang}\n" if lang else "```\n"
            action_parts.append(f"{lang_part}{code.strip()}\n```")
        action_text = "\n\n".join(action_parts)
    
    # Also capture any other executable content patterns that might not be in code blocks
    if not action_text:
        command_patterns = [
            r'sed\s+-i',
            r'cat\s+<<',
            r'python\s+-c',
            r'python\s+-m',
            r'grep\s+-',
            r'find\s+\.',
        ]
        for pattern in command_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                lines = content.split('\n')
                command_lines = [line.strip() for line in lines if re.search(pattern, line, re.IGNORECASE)]
                if command_lines:
                    action_text = '\n'.join(command_lines)
                break
    
    return thought_text, action_text


def _generate_embeddings(
    thought_messages: list[dict], embedding_model_name: str | None = None
) -> list[dict] | None:
    """Generate embeddings for thought messages.
    
    Args:
        thought_messages: List of thought message dicts with 'id' and 'raw_text'
        embedding_model_name: Name of the embedding model to use (default: BAAI/bge-large-en-v1.5)
        
    Returns:
        List of dicts with 'id' and 'embedding', or None if sentence-transformers is not available
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    if not thought_messages:
        return []
    
    if embedding_model_name is None:
        embedding_model_name = "BAAI/bge-large-en-v1.5"
    
    try:
        model = SentenceTransformer(embedding_model_name)
        texts = [msg["raw_text"] for msg in thought_messages]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        return [
            {"id": msg["id"], "embedding": embedding.tolist()}
            for msg, embedding in zip(thought_messages, embeddings)
        ]
    except Exception as e:
        # If embedding generation fails, return None (don't break the save process)
        import warnings
        warnings.warn(f"Failed to generate embeddings: {e}", UserWarning)
        return None


def save_extracted_messages(
    messages: list[dict], traj_path: Path, embedding_model: str | None = None
) -> None:
    """Extract and save assistant/user messages to separate JSON files.
    
    Args:
        messages: List of message dicts from the agent
        traj_path: Path to the trajectory file (used to determine output directory)
        embedding_model: Name of the embedding model to use for thought embeddings (default: BAAI/bge-large-en-v1.5)
    """
    if not messages or traj_path is None:
        return
    
    # Extract assistant and user messages
    assistant_messages = []
    user_messages = []
    thought_messages = []
    action_messages = []
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "assistant":
            assistant_messages.append({"id": i, "raw_text": content})
            # Extract thought and action sections
            thought_text, action_text = _extract_thought_and_action(content)
            if thought_text:
                thought_messages.append({"id": i, "raw_text": thought_text})
            if action_text:
                action_messages.append({"id": i, "raw_text": action_text})
        elif role == "user":
            user_messages.append({"id": i, "raw_text": content})
    
    # Save to the same directory as the trajectory file
    output_dir = traj_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save assistant messages
    assistant_path = output_dir / "assistant.json"
    assistant_path.write_text(json.dumps(assistant_messages, indent=2))
    
    # Save user messages
    user_path = output_dir / "user.json"
    user_path.write_text(json.dumps(user_messages, indent=2))
    
    # Save thought messages
    thought_path = output_dir / "thought.json"
    thought_path.write_text(json.dumps(thought_messages, indent=2))
    
    # Save action messages
    action_path = output_dir / "action.json"
    action_path.write_text(json.dumps(action_messages, indent=2))
    
    # Generate and save thought embeddings
    thought_embeddings = _generate_embeddings(thought_messages, embedding_model)
    if thought_embeddings is not None:
        embedding_path = output_dir / "thought_embedding.json"
        embedding_path.write_text(json.dumps(thought_embeddings, indent=2))
