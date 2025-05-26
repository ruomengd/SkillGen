skillgen_demoac_skill = '''
{instructions}

## Golden Segment (What to Imitate)
Here is a related action sequence, which may not be fully accurate, but help identify promising directions:
{trajectories}

## Step-wise Reusable Skills (Context-Aware Guidance)
These skills are relevant to the current context based on your most recent action.
They suggest promising steps to explore the environment:
{skills}

## Instruction
- You should use the following commands for help when your action cannot be understood: check valid actions.
- You should use the following commands for help when your action cannot be understood: inventory.
- Generate the **next best action** to reach the goal.  

Goal: {goal}
{history}
Action:
'''

