"""
Task definitions and graders for the warehouse environment.

Defines 3 tasks with increasing difficulty:
- easy:   Single order, small grid, 2 items
- medium: Multiple orders, medium grid, 6 items
- hard:   Rush hour with priority orders, large grid, 10 items
"""

from .warehouse_logic import WarehouseConfig

# ─────────────────────────────────────────────────────────
# Task: EASY — Single Order
# ─────────────────────────────────────────────────────────
TASK_EASY = WarehouseConfig(
    grid_rows=6,
    grid_cols=6,
    max_steps=30,
    seed=42,
    items=[
        {"item_id": "item_1", "name": "Widget A", "position": [1, 2]},
        {"item_id": "item_2", "name": "Gadget B", "position": [1, 4]},
    ],
    orders=[
        {
            "order_id": "order_1",
            "required_items": ["item_1", "item_2"],
            "priority": 1,
        },
    ],
)

# ─────────────────────────────────────────────────────────
# Task: MEDIUM — Multi-Order
# ─────────────────────────────────────────────────────────
TASK_MEDIUM = WarehouseConfig(
    grid_rows=8,
    grid_cols=8,
    max_steps=60,
    seed=123,
    items=[
        {"item_id": "item_1", "name": "Widget A", "position": [1, 1]},
        {"item_id": "item_2", "name": "Gadget B", "position": [1, 3]},
        {"item_id": "item_3", "name": "Part C", "position": [1, 5]},
        {"item_id": "item_4", "name": "Sensor D", "position": [3, 2]},
        {"item_id": "item_5", "name": "Cable E", "position": [3, 4]},
        {"item_id": "item_6", "name": "Board F", "position": [3, 6]},
    ],
    orders=[
        {
            "order_id": "order_1",
            "required_items": ["item_1", "item_2"],
            "priority": 1,
        },
        {
            "order_id": "order_2",
            "required_items": ["item_3", "item_4"],
            "priority": 1,
        },
        {
            "order_id": "order_3",
            "required_items": ["item_5", "item_6"],
            "priority": 2,
        },
    ],
)

# ─────────────────────────────────────────────────────────
# Task: HARD — Rush Hour
# ─────────────────────────────────────────────────────────
TASK_HARD = WarehouseConfig(
    grid_rows=10,
    grid_cols=10,
    max_steps=80,
    seed=999,
    items=[
        {"item_id": "item_1", "name": "Widget A", "position": [1, 1]},
        {"item_id": "item_2", "name": "Gadget B", "position": [1, 3]},
        {"item_id": "item_3", "name": "Part C", "position": [1, 5]},
        {"item_id": "item_4", "name": "Sensor D", "position": [1, 7]},
        {"item_id": "item_5", "name": "Cable E", "position": [3, 2]},
        {"item_id": "item_6", "name": "Board F", "position": [3, 4]},
        {"item_id": "item_7", "name": "Module G", "position": [3, 6]},
        {"item_id": "item_8", "name": "Battery H", "position": [5, 1]},
        {"item_id": "item_9", "name": "Chip I", "position": [5, 3]},
        {"item_id": "item_10", "name": "Motor J", "position": [5, 5]},
    ],
    orders=[
        {
            "order_id": "order_1",
            "required_items": ["item_1", "item_2"],
            "priority": 3,  # URGENT
        },
        {
            "order_id": "order_2",
            "required_items": ["item_3", "item_4"],
            "priority": 2,
        },
        {
            "order_id": "order_3",
            "required_items": ["item_5", "item_6"],
            "priority": 1,
        },
        {
            "order_id": "order_4",
            "required_items": ["item_7", "item_8"],
            "priority": 2,
        },
        {
            "order_id": "order_5",
            "required_items": ["item_9", "item_10"],
            "priority": 1,
        },
    ],
)

# Registry of all tasks
TASKS = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
}

TASK_DESCRIPTIONS = {
    "easy": "Single Order — Pick 2 items and deliver 1 order on a 6×6 grid (30 steps max).",
    "medium": "Multi-Order — Pick 6 items and deliver 3 orders on an 8×8 grid (60 steps max).",
    "hard": "Rush Hour — Pick 10 items and deliver 5 priority orders on a 10×10 grid (80 steps max).",
}


def get_task_config(task_id: str) -> WarehouseConfig:
    """Get configuration for a task by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]
