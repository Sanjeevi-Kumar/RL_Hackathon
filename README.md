---
title: Warehouse Order Fulfillment
emoji: 🏭
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# 🏭 Warehouse Order Fulfillment — OpenEnv Environment

A real-world **warehouse operations** RL environment built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv). An AI agent navigates a grid-based warehouse, picks items from shelves, and delivers customer orders — simulating the core challenge faced by Amazon, Ocado, and warehouse robotics companies worldwide.

## 🎯 Why Warehouse Operations?

Warehouse order picking is a **multi-billion dollar logistics problem**. Companies actively train RL agents for exactly this task: deciding which items to pick, which routes to take, and how to prioritize orders under time pressure. This environment models that decision-making process.

---

## 🔧 Environment Overview

### Architecture
```
Agent ←→ MCP Tools ←→ Warehouse Simulation
              │
    ┌─────────┼──────────┐
    │         │          │
 move()  pick_item()  deliver_order()
```

### Action Space (MCP Tools)

| Tool | Args | Description |
|------|------|-------------|
| `view_warehouse()` | — | Get full warehouse state (grid, items, orders, agent position) |
| `move(direction)` | `"up"` `"down"` `"left"` `"right"` | Move agent through warehouse aisles |
| `pick_item(item_id)` | `"item_1"`, etc. | Pick item from adjacent shelf |
| `deliver_order(order_id)` | `"order_1"`, etc. | Deliver order at packing station |
| `list_tasks()` | — | See available task difficulties |
| `get_score()` | — | Get graded score (0.0–1.0) |

### Observation Space

Each tool call returns a JSON result containing:
- **Agent position** `[row, col]` on the grid
- **Inventory**: items currently held
- **Grid layout**: shelf positions and aisles
- **Items on shelves**: available items with positions
- **Orders**: status, required items, priority, progress
- **Episode info**: steps remaining, cumulative reward, done flag

### Reward Function

| Event | Reward |
|-------|--------|
| Correct item picked | `+0.1` |
| Order fulfilled | `+0.3` |
| High-priority order early | `+0.2` bonus |
| Each step (time cost) | `-0.01` |
| Invalid action | `-0.05` |

Final score normalized to **[0.0, 1.0]** based on: orders fulfilled (60%), items progress (20%), efficiency (10%), valid actions (10%).

---

## 📋 Tasks

| Task | Difficulty | Grid | Orders | Items | Max Steps |
|------|-----------|------|--------|-------|-----------|
| `easy` | ⭐ | 6×6 | 1 | 2 | 30 |
| `medium` | ⭐⭐ | 8×8 | 3 | 6 | 60 |
| `hard` | ⭐⭐⭐ | 10×10 | 5 (with priorities) | 10 | 80 |

- **Easy**: Pick 2 items, deliver 1 order. Teaches basic navigation and mechanics.
- **Medium**: 3 orders with 6 items spread across a larger grid. Requires route planning.
- **Hard**: 5 orders with urgency priorities on a 10×10 grid. Demands optimal routing and order prioritization under time pressure.

---

## 🚀 Setup & Usage

### Prerequisites
- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install
```bash
git clone <your-repo-url>
cd warehouse_env

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Run Server Locally
```bash
# With uv
uv run server

# Or with uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Test the API
```bash
# Health check
curl http://localhost:8000/health
# → {"status": "healthy"}
```

### Use the Client
```python
from warehouse_env import WarehouseEnv

with WarehouseEnv(base_url="http://localhost:8000") as env:
    env.reset(task="easy")
    
    # See the warehouse
    warehouse = env.call_tool("view_warehouse")
    print(warehouse)
    
    # Navigate and pick
    env.call_tool("move", direction="up")
    env.call_tool("pick_item", item_id="item_1")
    
    # Deliver
    env.call_tool("deliver_order", order_id="order_1")
    
    # Check score
    score = env.call_tool("get_score")
    print(score)
```

---

## 🐳 Docker

```bash
# Build
docker build -t warehouse-env -f server/Dockerfile .

# Run
docker run -d -p 8000:8000 warehouse-env

# Connect
curl http://localhost:8000/health
```

---

## 🤖 Baseline Inference

The inference script uses an LLM to play through all 3 tasks:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:8000"

python inference.py
```

### Expected Baseline Scores

| Task | Expected Score |
|------|---------------|
| easy | ~0.7–0.9 |
| medium | ~0.4–0.6 |
| hard | ~0.2–0.4 |

Scores depend on the model used. GPT-4o-mini should pass easy, partially solve medium, and struggle with hard.

---

## 📁 Project Structure

```
warehouse_env/
├── __init__.py              # Package exports
├── client.py                # MCPToolClient subclass
├── models.py                # Pydantic type documentation
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── inference.py             # Baseline LLM inference
├── README.md                # This file
├── .dockerignore
└── server/
    ├── __init__.py
    ├── warehouse_environment.py  # MCPEnvironment implementation
    ├── warehouse_logic.py        # Core simulation logic
    ├── tasks.py                  # Task definitions & graders
    ├── app.py                    # FastAPI server
    ├── Dockerfile
    └── requirements.txt
```

---

## 📄 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | LLM API endpoint | For inference |
| `MODEL_NAME` | Model identifier | For inference |
| `HF_TOKEN` | HuggingFace / API key | For inference |
| `ENV_URL` | Environment server URL | For inference (default: localhost:8000) |

---

## License

MIT
