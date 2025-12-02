#!/usr/bin/env python3
"""Analyze agent trajectories using LM-as-a-judge to evaluate causal relationships."""

import json
import re
from pathlib import Path
from typing import Any

from minisweagent.models import get_model


def load_json_file(file_path: Path) -> list[dict[str, Any]]:
    """Load a JSON file and return its contents."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return json.loads(file_path.read_text())


def format_elements_for_prompt(elements: list[dict[str, Any]], element_type: str) -> str:
    """Format elements as a JSON string for the prompt."""
    formatted = json.dumps(elements, indent=2)
    return f"## {element_type}.json\n```json\n{formatted}\n```\n"


def build_prompt(thoughts: list[dict], actions: list[dict], observations: list[dict]) -> str:
    """Build the prompt for the LM judge."""
    prompt_template = Path(__file__).parent / "prompt.md"
    prompt_base = prompt_template.read_text()
    
    thoughts_str = format_elements_for_prompt(thoughts, "thought")
    actions_str = format_elements_for_prompt(actions, "action")
    observations_str = format_elements_for_prompt(observations, "observation")
    
    return f"""{prompt_base}

----------------------------------------------------------------------
INPUT DATA
----------------------------------------------------------------------

{thoughts_str}

{actions_str}

{observations_str}

----------------------------------------------------------------------
YOUR ANALYSIS
----------------------------------------------------------------------

Output your six JSON objects now:
"""


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract JSON objects from LM response text."""
    json_objects = []
    seen_objects = set()  # Track seen objects to avoid duplicates
    
    # Strategy 1: Look for JSON objects in code blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        # Try to extract multiple JSON objects from the code block
        objects_in_block = _extract_json_from_text(code_content)
        for obj in objects_in_block:
            obj_str = json.dumps(obj, sort_keys=True)
            if obj_str not in seen_objects:
                seen_objects.add(obj_str)
                json_objects.append(obj)
    
    # Strategy 2: Find standalone JSON objects in the text
    objects_in_text = _extract_json_from_text(text)
    for obj in objects_in_text:
        obj_str = json.dumps(obj, sort_keys=True)
        if obj_str not in seen_objects:
            seen_objects.add(obj_str)
            json_objects.append(obj)
    
    return json_objects


def _extract_json_from_text(text: str) -> list[dict[str, Any]]:
    """Extract all JSON objects from a text string."""
    objects = []
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    json_str = text[start_idx:i+1]
                    obj = json.loads(json_str)
                    if _is_valid_relationship(obj):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    return objects


def _is_valid_relationship(obj: dict[str, Any]) -> bool:
    """Check if an object is a valid relationship JSON."""
    required_fields = [
        "preceding_element_type", "preceding_element_id",
        "following_element_type", "following_element_id",
        "edit_category", "next_motivated_by_before", "important_score"
    ]
    return all(field in obj for field in required_fields)


def get_relationship_filename(preceding_type: str, following_type: str) -> str:
    """Get the filename for a relationship type."""
    return f"{preceding_type}_to_{following_type}.json"


def analyze_instance(instance_dir: Path, model) -> None:
    """Analyze a single instance directory."""
    print(f"Analyzing instance: {instance_dir.name}")
    
    # Load input files
    thought_file = instance_dir / "thought.json"
    action_file = instance_dir / "action.json"
    user_file = instance_dir / "user.json"
    
    thoughts = load_json_file(thought_file)
    actions = load_json_file(action_file)
    observations = load_json_file(user_file)
    
    # Build prompt
    prompt = build_prompt(thoughts, actions, observations)
    
    # Query LM
    messages = [{"role": "user", "content": prompt}]
    response = model.query(messages)
    lm_output = response["content"]
    
    # Extract JSON objects
    json_objects = extract_json_objects(lm_output)
    
    if len(json_objects) == 0:
        print(f"Error: No valid JSON objects found in LM output")
        print(f"LM output:\n{lm_output}")
        return
    
    print(f"  Extracted {len(json_objects)} relationship(s)")
    
    # Create output directory
    output_dir = instance_dir / "reverse-lm"
    output_dir.mkdir(exist_ok=True)
    
    # Group by relationship type and save
    relationship_types = {
        ("thought", "thought"): [],
        ("action", "thought"): [],
        ("observation", "thought"): [],
        ("thought", "action"): [],
        ("action", "action"): [],
        ("observation", "action"): [],
    }
    
    for obj in json_objects:
        preceding = obj["preceding_element_type"]
        following = obj["following_element_type"]
        key = (preceding, following)
        if key in relationship_types:
            relationship_types[key].append(obj)
        else:
            print(f"  Warning: Unknown relationship type: {preceding} -> {following}")
    
    # Save each relationship type to its own file
    for (preceding, following), objects in relationship_types.items():
        filename = get_relationship_filename(preceding, following)
        output_file = output_dir / filename
        output_file.write_text(json.dumps(objects, indent=2))
        if objects:
            print(f"  Saved {len(objects)} relationship(s) to {filename}")
    
    print(f"Completed analysis for {instance_dir.name}\n")


def find_instance_directories(logs_dir: Path) -> list[Path]:
    """Find all instance directories in the logs structure."""
    instances = []
    
    # Structure: logs/swebench/{run_dir}/{instance}/
    swebench_dir = logs_dir / "swebench"
    if not swebench_dir.exists():
        return instances
    
    for run_dir in swebench_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for instance_dir in run_dir.iterdir():
            if not instance_dir.is_dir():
                continue
            # Check if it has the required files
            if all((instance_dir / f).exists() for f in ["thought.json", "action.json", "user.json"]):
                instances.append(instance_dir)
    
    return instances


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze agent trajectories using LM-as-a-judge")
    parser.add_argument(
        "logs_dir",
        type=Path,
        help="Path to logs directory (e.g., logs/)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        help="Process only a specific instance (e.g., django__django-7530)",
    )
    args = parser.parse_args()
    
    # Initialize model
    model = get_model(
        input_model_name="claude-sonnet-4-5-20250929",
        config={
            "model_kwargs": {
                "api_base": "http://localhost:8080",
                "api_key": "dummy",
                "temperature": 0.0,
            },
            "cost_tracking": "ignore_errors",
        }
    )
    
    # Find instances
    if args.instance:
        # Find the instance directory
        swebench_dir = args.logs_dir / "swebench"
        instances = []
        for run_dir in swebench_dir.iterdir():
            if not run_dir.is_dir():
                continue
            instance_dir = run_dir / args.instance
            if instance_dir.exists() and instance_dir.is_dir():
                if all((instance_dir / f).exists() for f in ["thought.json", "action.json", "user.json"]):
                    instances.append(instance_dir)
                    break
        if not instances:
            print(f"Instance '{args.instance}' not found or missing required files")
            return
    else:
        instances = find_instance_directories(args.logs_dir)
    
    print(f"Found {len(instances)} instance(s) to analyze\n")
    
    # Process each instance
    for instance_dir in instances:
        try:
            analyze_instance(instance_dir, model)
        except Exception as e:
            print(f"Error analyzing {instance_dir.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

