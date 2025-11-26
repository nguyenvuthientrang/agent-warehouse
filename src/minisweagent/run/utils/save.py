import dataclasses
import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from minisweagent import Agent, __version__


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
        Path to the log file in format: logs/{run_type}/{model}_{time}/{instance}.traj.json
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
        log_path = base_dir / "logs" / run_type / f"{safe_model_name}_{timestamp}" / f"{safe_instance}.traj.json"
    else:
        log_path = base_dir / "logs" / run_type / f"{safe_model_name}_{timestamp}" / "run.traj.json"
    
    return log_path
