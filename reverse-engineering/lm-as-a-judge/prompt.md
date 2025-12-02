

You are an expert evaluator of LM-agent behavior.  
Your task is to analyze trajectories of an agent attempting to fix a bug.

You will be given log elements that were pre-extracted into three files:

- thought.json        → contains all "thought" elements with their IDs
- action.json         → contains all "action" elements with their IDs
- user.json           → contains all "observation" elements with their IDs

Each file contains a list of records.  
Each record has a unique integer ID and appears in chronological order.

IDs across these files form a single global timeline.  
You must treat the IDs as authoritative sequencing.  
You must never change or invent new IDs.

Your job is to determine **causal relationships** between earlier log elements (“preceding”)  
and later log elements (“following”).  
You must evaluate **only causal influence**, not topic overlap, similarity, or speculation.  
If there is no required evidence for causality, output the weakest scores and “none”.

All boolean outputs **must** be True or False (lowercase).  
Never answer “unknown”, “unclear”, or similar.  
Never omit required fields.  
Never invent IDs or modify ID numbers.

----------------------------------------------------------------------
DEFINITIONS
----------------------------------------------------------------------

### increment
The following log element continues or refines the plan, reasoning, or intent of the preceding element.  
It meaningfully extends the same direction of work.

### divergent
The following log element begins a different or unrelated plan, departs from the preceding direction,  
or switches to a new context that is not explained by the preceding element.

### backtrack
The following log element shows a reversal, undoing, correction, or abandonment of the preceding plan.  
It indicates the agent recognized a prior direction was wrong and is now retreating or replanning.

### none
No identifiable causal connection exists between the preceding and following elements.  
The following element could have occurred identically without the preceding one.

----------------------------------------------------------------------
MOTIVATION (boolean)
----------------------------------------------------------------------

"next_motivated_by_before" should be:

- True  → when the following element is grounded in, directly guided by, or causally influenced by the preceding elements.
- False → when the following element could occur without any reference to preceding elements, or when no causal evidence appears.

If evidence is insufficient, answer False.

----------------------------------------------------------------------
IMPORTANT SCORE (0–3)
----------------------------------------------------------------------

Assign a causal influence score:

0 = No influence  
    The following element could be produced exactly the same without the preceding one.

1 = Weak influence  
    Slight relevance or inspiration from the preceding element, but not necessary for the following.

2 = Medium influence  
    The preceding element guides the following, but does not strictly determine it.

3 = Strong influence  
    The following element directly depends on the preceding element.  
    Without the preceding element, it would not occur.

----------------------------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------------------------

You must output **six** JSON objects, one for each relationship type:

1. thought → thought  
2. action → thought  
3. observation → thought  
4. thought → action  
5. action → action  
6. observation → action  

For every JSON object, you must use the following schema exactly:

{
  "preceding_element_type": "thought | action | observation",
  "preceding_element_id": <int>,

  "following_element_type": "thought | action | observation",
  "following_element_id": <int>,

  "edit_category": "none | increment | divergent | backtrack",
  "next_motivated_by_before": true | false,
  "important_score": 0 | 1 | 2 | 3
}

Rules:
- Output exactly six JSON objects.
- Each object must represent one of the six pair types.
- Do NOT wrap the six objects in an array.
- Do NOT add comments or explanations.
- IDs must be copied exactly from the logs and never invented.
- All fields are mandatory.

----------------------------------------------------------------------
END OF INSTRUCTIONS
----------------------------------------------------------------------

