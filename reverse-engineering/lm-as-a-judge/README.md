# LM-as-a-Judge Analysis

This directory contains code to analyze agent trajectories using LM-as-a-judge to evaluate causal relationships between log elements.

## Usage

Analyze all instances in a logs directory:

```bash
python reverse-engineering/lm-as-a-judge/analyze.py logs/
```

Analyze a specific instance:

```bash
python reverse-engineering/lm-as-a-judge/analyze.py logs/ --instance django__django-7530
```

## Output

For each instance, the script creates a `reverse-lm/` subdirectory containing 6 JSON files:

1. `thought_to_thought.json` - Relationships between thought elements
2. `action_to_thought.json` - Relationships from actions to thoughts
3. `observation_to_thought.json` - Relationships from observations to thoughts
4. `thought_to_action.json` - Relationships from thoughts to actions
5. `action_to_action.json` - Relationships between action elements
6. `observation_to_action.json` - Relationships from observations to actions

Each file contains a list of relationship objects with the following schema:

```json
{
  "preceding_element_type": "thought | action | observation",
  "preceding_element_id": <int>,
  "following_element_type": "thought | action | observation",
  "following_element_id": <int>,
  "edit_category": "none | increment | divergent | backtrack",
  "next_motivated_by_before": true | false,
  "important_score": 0 | 1 | 2 | 3
}
```

## Configuration

The script uses the local model configured in `src/minisweagent/config/mini_local.yaml`:
- Model: `claude-sonnet-4-5-20250929`
- API Base: `http://localhost:8080`
- API Key: `dummy`

