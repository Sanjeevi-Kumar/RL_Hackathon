---
title: Warehouse RL Environment
emoji: 🏭
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# 🏭 Warehouse RL Environment — Meta OpenEnv Hackathon

> **OpenEnv-compliant, stateful, multi-objective Reinforcement Learning environment
> for multi-product warehouse navigation with LLM-powered agent inference.**

---

## 🎯 Overview

This environment challenges an RL agent to operate an autonomous robot inside a
**12×12 warehouse grid** filled with shelf obstacles, forklift collision zones,
and 15 products spread across three priority tiers. The agent must:

1. **Navigate** around shelves and collision zones
2. **Collect** products in priority order (HIGH → MEDIUM → LOW)
3. **Manage battery** by visiting recharge stations proactively
4. **Deposit** collected items at the loading dock
5. **Maximise score** within 200 steps

The multi-phase nature forces the LLM/agent to **reason across many turns** — exactly
what the OpenEnv hackathon judges are looking for.

---

## 🗂️ Project Structure

```
warehouse_env/
├── src/
│   └── envs/
│       └── warehouse_env/
│           ├── __init__.py
│           ├── models.py          ← Action / Observation / State (Pydantic)
│           ├── client.py          ← HTTP client (type-safe)
│           └── server/
│               ├── __init__.py
│               ├── environment.py ← Core RL logic, step(), reset()
│               └── app.py         ← FastAPI server + built-in /web UI
├── inference.py                   ← LLM agent loop (HF Router API)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the environment server

```bash
uvicorn src.envs.warehouse_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the built-in Web UI

Navigate to **http://localhost:8000/web** — you'll see a live interactive warehouse
grid where you can manually control the agent and inspect every product.

### 4. Run the LLM inference loop

```bash
export HF_TOKEN=hf_your_token_here
python inference.py --model meta-llama/Llama-3.1-70B-Instruct --max_steps 80 --verbose
```

### 5. Multi-episode evaluation

```bash
python inference.py --episodes 5 --eval --max_steps 80
```

---

## 🐳 Docker

```bash
# Build
docker build -t warehouse-rl-env .

# Run
docker run -p 8000:8000 warehouse-rl-env

# With HF token for inference (run inference.py separately)
docker run -p 8000:8000 -e HF_TOKEN=$HF_TOKEN warehouse-rl-env
```

---

## 📐 Environment Design

### Grid (12×12)

| Symbol | Meaning                        |
| ------ | ------------------------------ |
| `R`  | Robot agent                    |
| `★` | Loading dock (goal)            |
| `⚡` | Recharge station               |
| `■` | Shelf obstacle (wall)          |
| `⬛` | Forklift collision zone (wall) |
| `🔴` | High-priority product          |
| `🟠` | Medium-priority product        |
| `🟢` | Low-priority product           |

### Reward Function

| Event                             | Reward                              |
| --------------------------------- | ----------------------------------- |
| Deposit high-priority product     | `+value × 2.0`                   |
| Deposit medium-priority product   | `+value × 1.3`                   |
| Deposit low-priority product      | `+value × 1.0`                   |
| Per movement step                 | `-0.05`                           |
| Carrying weight > 0               | `-0.03 × n_items` extra per step |
| Invalid action (wall/obstacle)    | `-1.00`                           |
| Repeated invalid actions          | `-0.30 × consecutive_count`      |
| Battery < 20%                     | `-0.30` per step                  |
| Battery exhausted                 | `-5.00` + episode ends            |
| All high-priority deposited first | `+5.00` bonus                     |
| All products deposited            | `+10.00` completion bonus         |

### Why It Requires Multi-Turn Reasoning

- The agent must **plan routes** across a large grid with obstacles
- **Battery management** requires anticipating future energy needs
- **Inventory management** (max 3 items) requires batching decisions
- **Priority ordering** requires global awareness of the product manifest
- **Deposit/recharge tradeoffs** depend on inventory state + distance to dock

---

## 🤖 LLM Agent Design

The LLM receives a rich, structured prompt including:

- Agent position, battery level, inventory
- Full product manifest with distances and priority
- 5×5 visible cell grid around the agent
- Recent action history (last 6 steps)
- Environment-generated reasoning hint
- Last action result

The LLM outputs structured JSON:

```json
{
  "reasoning": "Battery at 18% — must recharge before collecting more items.",
  "movement": "north",
  "interact": "recharge",
  "target_sku": null
}
```

---

## 🔌 OpenEnv API

All endpoints follow the OpenEnv standard:

```
POST /reset    → Observation   Start new episode
POST /step     → Observation   Execute one action
GET  /state    → State         Full internal state
GET  /web      → HTML          Browser UI
GET  /health   → dict          Liveness check
```

### Python Client

```python
from src.envs.warehouse_env.client import WarehouseEnvClient
from src.envs.warehouse_env.models import Action

env = WarehouseEnvClient(base_url="http://localhost:8000")
obs = env.reset()

# Manual action
obs = env.step(Action(movement="north", interact="pickup", target_sku="SKU-H1"))

# Shorthand helpers
obs = env.move("east")
obs = env.pickup("SKU-H2")
obs = env.deposit()
obs = env.recharge()

state = env.state()   # full internal state
```

---

## 🏆 Hackathon Judging Criteria

| Criterion                    | How We Satisfy It                                                       |
| ---------------------------- | ----------------------------------------------------------------------- |
| **Complexity**         | 15 products × 3 priorities, battery management, obstacle navigation    |
| **Statefulness**       | Full `State` persists between steps; multi-phase episode structure    |
| **Reasoning required** | Route planning, inventory batching, priority ordering, energy tradeoffs |
| **OpenEnv compliance** | `step()`, `reset()`, `state()` via FastAPI; Pydantic schemas      |
| **Built-in UI**        | `/web` endpoint with live canvas rendering of the full grid           |
| **LLM integration**    | HF Router API + chain-of-thought prompting + JSON action parsing        |
| **Dockerised**         | Single `Dockerfile` for reproducible deployment                       |
