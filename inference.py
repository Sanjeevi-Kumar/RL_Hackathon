"""
Baseline Inference Script for Warehouse Environment.

Runs an LLM agent against all 3 warehouse tasks and reports scores.
Uses the OpenAI API client with environment variables for configuration.

Required environment variables:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier to use
    HF_TOKEN      — Your HuggingFace / API key

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

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

TASKS = ["easy", "medium", "hard"]
MAX_AGENT_STEPS = 50  # max LLM calls per task

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


def run_task(task_id: str) -> float:
    """Run the agent on a single task. Returns the score."""
    print(f"\n{'='*60}")
    print(f"[START] task={task_id}")
    print(f"{'='*60}")

    # Reset environment
    reset_result = env_reset(task_id)
    metadata = reset_result.get("observation", {}).get("metadata", {})
    task_desc = metadata.get("task_description", task_id)
    print(f"  Task: {task_desc}")

    # Get initial warehouse view
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task: {task_desc}\n\nStart by viewing the warehouse layout.",
        },
    ]

    done = False
    step_num = 0
    total_reward = 0.0

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
            print(f"  [STEP] step={step_num} error=\"LLM call failed: {e}\"")
            break

        # Parse action from LLM
        tool_name, tool_args = parse_llm_action(llm_text)

        # Execute action in environment
        try:
            result = env_call_tool(tool_name, tool_args)
            obs = result.get("observation", {})
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            total_reward += reward

            # Extract result text for the LLM
            obs_meta = obs.get("metadata", {})
            if "result" in obs_meta:
                result_text = str(obs_meta["result"])
            else:
                result_text = json.dumps(obs_meta, indent=2)

            print(
                f"  [STEP] step={step_num} "
                f"tool={tool_name} "
                f"args={json.dumps(tool_args)} "
                f"reward={reward} "
                f"done={done}"
            )

            # Add to conversation
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{result_text}\n\nReward: {reward}, Done: {done}\n\nWhat's your next action?",
            })

        except Exception as e:
            print(f"  [STEP] step={step_num} error=\"Env call failed: {e}\"")
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({
                "role": "user",
                "content": f"Error: {e}\n\nTry a different action.",
            })

    # Get final score
    try:
        score_result = env_call_tool("get_score", {})
        score_meta = score_result.get("observation", {}).get("metadata", {})
        if "result" in score_meta:
            score_data = json.loads(score_meta["result"])
            score = score_data.get("score", 0.0)
        else:
            score = 0.0
    except Exception:
        score = 0.0

    print(
        f"[END] task={task_id} "
        f"score={score} "
        f"steps={step_num} "
        f"total_reward={round(total_reward, 4)}"
    )

    return score


# ─── Main ───────────────────────────────────────────────

def main():
    """Run inference on all tasks and report results."""
    print("=" * 60)
    print("  Warehouse Environment — Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API: {API_BASE_URL}")
    print(f"  Environment: {ENV_URL}")
    print("=" * 60)

    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    results = {}
    for task_id in TASKS:
        try:
            score = run_task(task_id)
            results[task_id] = score
        except Exception as e:
            print(f"[END] task={task_id} score=0.0 error=\"{e}\"")
            results[task_id] = 0.0

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for task_id, score in results.items():
        print(f"  {task_id:10s}: {score:.4f}")
    avg = sum(results.values()) / len(results)
    print(f"  {'average':10s}: {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
