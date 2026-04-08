"""
Baseline Inference Script for Warehouse Environment.

Runs an LLM agent against all 3 warehouse tasks and reports scores.
Uses the OpenAI API client with environment variables for configuration.

MANDATORY REQUIREMENTS:
- API_BASE_URL   — The API endpoint for the LLM
- MODEL_NAME     — The model identifier to use for inference
- HF_TOKEN       — Your Hugging Face / API key

STDOUT FORMAT (REQUIRED FOR EVALUATION):
- [START] task=<task_name> env=warehouse_env model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your-api-key"
    python inference.py
"""

import json
import os
import sys
import time
import requests

from openai import OpenAI

# ─── Configuration ──────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
BENCHMARK = "warehouse_env"

TASKS = ["easy", "medium", "hard"]
MAX_AGENT_STEPS = 50  # max LLM calls per task


def format_action_str(tool_name: str, tool_args: dict) -> str:
    """Format action as a human-readable string."""
    if tool_name == "move":
        return f"move('{tool_args.get('direction', 'up')}')"
    elif tool_name == "pick_item":
        return f"pick_item('{tool_args.get('item_id', '')}')"
    elif tool_name == "deliver_order":
        return f"deliver_order('{tool_args.get('order_id', '')}')"
    elif tool_name == "view_warehouse":
        return "view_warehouse()"
    elif tool_name == "list_tasks":
        return "list_tasks()"
    elif tool_name == "get_score":
        return "get_score()"
    else:
        return f"{tool_name}({json.dumps(tool_args)})"


# ─── OpenAI Client Setup ───────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─── Environment Interaction Helpers ────────────────────

def env_reset(task_id: str) -> dict:
    """Reset the environment for a given task via HTTP."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_call_tool(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool on the environment via HTTP step endpoint."""
    payload = {
        "tool_name": tool_name,
        "arguments": arguments,
    }
    resp = requests.post(
        f"{ENV_URL}/step",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    """Get current environment state."""
    resp = requests.get(f"{ENV_URL}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─── LLM Agent ──────────────────────────────────────────

SYSTEM_PROMPT = """You are a warehouse fulfillment agent. You operate in a grid-based warehouse
where you must navigate aisles, pick items from shelves, and deliver them to a packing station
to fulfill customer orders.

You have the following tools:
1. view_warehouse() - See the warehouse layout, item locations, orders, and your position
2. move(direction) - Move: "up", "down", "left", "right" (cannot walk on shelves)
3. pick_item(item_id) - Pick an item when adjacent to its shelf
4. deliver_order(order_id) - Deliver an order at the packing station
5. get_score() - Check your current score

Strategy tips:
- First call view_warehouse() to understand the layout
- Plan efficient routes to collect items
- You must be adjacent to a shelf to pick items
- Return to the packing station to deliver orders
- Prioritize high-priority orders

Respond with EXACTLY ONE tool call in JSON format:
{"tool": "<tool_name>", "args": {<arguments>}}

Examples:
{"tool": "view_warehouse", "args": {}}
{"tool": "move", "args": {"direction": "up"}}
{"tool": "pick_item", "args": {"item_id": "item_1"}}
{"tool": "deliver_order", "args": {"order_id": "order_1"}}
"""


def parse_llm_action(response_text: str) -> tuple[str, dict]:
    """Parse LLM response to extract tool call."""
    text = response_text.strip()

    # Try to find JSON in the response
    # Look for { ... } pattern
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        json_str = text[start:end]
        try:
            data = json.loads(json_str)
            tool = data.get("tool", "view_warehouse")
            args = data.get("args", {})
            return tool, args
        except json.JSONDecodeError:
            pass

    # Fallback: view warehouse
    return "view_warehouse", {}


def run_task(task_id: str) -> dict:
    """
    Run the agent on a single task.
    Returns: dict with score, steps, rewards list, and success status.
    """
    # [START] — Task begins
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    # Reset environment
    try:
        reset_result = env_reset(task_id)
        metadata = reset_result.get("observation", {}).get("metadata", {})
        task_desc = metadata.get("task_description", task_id)
    except Exception as e:
        print(f"[END] success=false steps=0 rewards= error={json.dumps(str(e))}")
        return {"score": 0.0, "steps": 0, "rewards": [], "success": False, "error": str(e)}

    # Initialize conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task: {task_desc}\n\nStart by viewing the warehouse layout.",
        },
    ]

    done = False
    step_num = 0
    rewards = []
    last_error = None

    while not done and step_num < MAX_AGENT_STEPS:
        step_num += 1

        # Get LLM response
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            llm_text = response.choices[0].message.content or ""
        except Exception as e:
            last_error = str(e)
            error_msg = f"LLM call failed: {e}"
            action_str = "error"
            print(f"[STEP] step={step_num} action={action_str} reward=0.00 done=false error={json.dumps(error_msg)}")
            break

        # Parse action from LLM
        tool_name, tool_args = parse_llm_action(llm_text)
        action_str = format_action_str(tool_name, tool_args)

        # Execute action in environment
        try:
            result = env_call_tool(tool_name, tool_args)
            obs = result.get("observation", {})
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            rewards.append(reward)

            # Extract result text for the LLM
            obs_meta = obs.get("metadata", {})
            if "result" in obs_meta:
                result_text = str(obs_meta["result"])
            else:
                result_text = json.dumps(obs_meta, indent=2)

            # [STEP] — Each step of the episode
            done_str = "true" if done else "false"
            error_str = "null"
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")

            # Add to conversation
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{result_text}\n\nReward: {reward}, Done: {done}\n\nWhat's your next action?",
            })

        except Exception as e:
            last_error = str(e)
            error_msg = str(e)
            done_str = "false"
            print(f"[STEP] step={step_num} action={action_str} reward=0.00 done={done_str} error={json.dumps(error_msg)}")
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({
                "role": "user",
                "content": f"Error: {e}\n\nTry a different action.",
            })

    # Get final score
    final_score = 0.0
    try:
        score_result = env_call_tool("get_score", {})
        score_meta = score_result.get("observation", {}).get("metadata", {})
        if "result" in score_meta:
            score_data = json.loads(score_meta["result"])
            final_score = score_data.get("score", 0.0)
        else:
            final_score = 0.0
    except Exception as e:
        final_score = 0.0

    # [END] — Episode complete
    success = final_score > 0.0 and not last_error
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else ""
    
    if last_error:
        error_json = json.dumps(last_error)
        print(f"[END] success={success_str} steps={step_num} rewards={rewards_str} error={error_json}")
    else:
        print(f"[END] success={success_str} steps={step_num} rewards={rewards_str}")

    return {
        "score": final_score,
        "steps": step_num,
        "rewards": rewards,
        "success": success,
        "error": last_error,
    }


# ─── Run all tasks ──────────────────────────────────────

def main():
    """Run inference on all tasks and validate configuration."""
    # Validate required environment variables
    if not API_BASE_URL:
        print("ERROR: API_BASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable not set", file=sys.stderr)
        sys.exit(1)
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN or OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Run tasks
    results = {}
    for task_id in TASKS:
        try:
            result = run_task(task_id)
            results[task_id] = result
        except Exception as e:
            print(f"[END] success=false steps=0 rewards= error={json.dumps(str(e))}")
            results[task_id] = {
                "score": 0.0,
                "steps": 0,
                "rewards": [],
                "success": False,
                "error": str(e),
            }


if __name__ == "__main__":
    main()
