"""
Warehouse Environment Implementation.

An MCP environment that simulates warehouse order fulfillment operations.
The agent navigates a grid warehouse, picks items from shelves, and
delivers them to fulfill customer orders.

Tools:
- `view_warehouse()`: Get the current warehouse state
- `move(direction)`: Move agent up/down/left/right
- `pick_item(item_id)`: Pick an item from an adjacent shelf
- `deliver_order(order_id)`: Deliver collected items for an order
"""

import json
from typing import Any, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

from .tasks import TASKS, TASK_DESCRIPTIONS, get_task_config
from .warehouse_logic import WarehouseConfig, WarehouseSimulation


class WarehouseEnvironment(MCPEnvironment):
    """
    Warehouse order fulfillment environment.

    Simulates a grid-based warehouse where an agent must:
    1. Navigate the warehouse aisles
    2. Pick items from shelves
    3. Deliver items to the packing station to fulfill orders

    Interactions happen through MCP tools:
    - `view_warehouse()`: See the full warehouse state
    - `move(direction)`: Navigate the warehouse
    - `pick_item(item_id)`: Pick items from adjacent shelves
    - `deliver_order(order_id)`: Deliver orders at the packing station
    """

    def __init__(self):
        """Initialize the warehouse environment with MCP tools."""
        mcp = FastMCP("warehouse_env")
        self._sim: Optional[WarehouseSimulation] = None
        self._current_task: str = "easy"

        # We need to capture self for closures
        env_ref = self

        @mcp.tool
        def view_warehouse() -> str:
            """
            View the current warehouse state.

            Returns a full observation including:
            - Agent position and grid layout
            - Items on shelves and in inventory
            - Order status and progress
            - Steps remaining and total reward

            Call this first after reset to understand the warehouse layout.
            """
            if env_ref._sim is None:
                return json.dumps({"error": "Environment not initialized. Call reset first."})
            obs = env_ref._sim.get_observation()
            return json.dumps(obs, indent=2)

        @mcp.tool
        def move(direction: str) -> str:
            """
            Move the agent in the warehouse.

            Args:
                direction: Direction to move — one of: "up", "down", "left", "right"

            Returns:
                Result with success status, message, reward, and new position.
                Moving into walls or shelves is invalid (-0.05 penalty).
                Each valid move costs -0.01 (time pressure).
            """
            if env_ref._sim is None:
                return json.dumps({"error": "Environment not initialized. Call reset first."})
            result = env_ref._sim.move(direction)
            return json.dumps(result)

        @mcp.tool
        def pick_item(item_id: str) -> str:
            """
            Pick an item from an adjacent shelf.

            The agent must be adjacent (Manhattan distance 1) to the shelf
            containing the item. Items are added to the agent's inventory.

            Args:
                item_id: The ID of the item to pick (e.g., "item_1")

            Returns:
                Result with success status, reward (+0.1 on success), and updated inventory.
            """
            if env_ref._sim is None:
                return json.dumps({"error": "Environment not initialized. Call reset first."})
            result = env_ref._sim.pick_item(item_id)
            return json.dumps(result)

        @mcp.tool
        def deliver_order(order_id: str) -> str:
            """
            Deliver items for an order at the packing station.

            The agent must be at the packing station. Matching items from
            the inventory are transferred to the order. If all required items
            are delivered, the order is marked as fulfilled (+0.3 reward).
            High-priority orders delivered early get bonus rewards.

            Args:
                order_id: The ID of the order to deliver (e.g., "order_1")

            Returns:
                Result with success status, reward, and fulfillment status.
            """
            if env_ref._sim is None:
                return json.dumps({"error": "Environment not initialized. Call reset first."})
            result = env_ref._sim.deliver_order(order_id)
            return json.dumps(result)

        @mcp.tool
        def list_tasks() -> str:
            """
            List all available tasks and their descriptions.

            Returns:
                JSON list of available task IDs with descriptions and difficulty.
            """
            tasks_info = []
            for task_id, desc in TASK_DESCRIPTIONS.items():
                config = TASKS[task_id]
                tasks_info.append({
                    "task_id": task_id,
                    "description": desc,
                    "grid_size": f"{config.grid_rows}x{config.grid_cols}",
                    "max_steps": config.max_steps,
                    "num_orders": len(config.orders),
                    "num_items": len(config.items),
                })
            return json.dumps(tasks_info, indent=2)

        @mcp.tool
        def get_score() -> str:
            """
            Get the current graded score for this episode.

            Returns a score between 0.0 and 1.0 based on:
            - Orders fulfilled (60%)
            - Items collected progress (20%)
            - Step efficiency (10%)
            - Invalid action penalty (10%)
            """
            if env_ref._sim is None:
                return json.dumps({"error": "Environment not initialized."})
            score = env_ref._sim.compute_score()
            return json.dumps({
                "score": score,
                "orders_delivered": env_ref._sim.orders_delivered,
                "total_orders": len(env_ref._sim.orders),
                "items_picked": env_ref._sim.items_picked,
                "steps_used": env_ref._sim.step_count,
                "max_steps": env_ref._sim.config.max_steps,
                "invalid_actions": env_ref._sim.invalid_actions,
                "done": env_ref._sim.done,
            })

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the warehouse environment.

        Args:
            seed: Optional random seed
            episode_id: Optional episode ID
            **kwargs: May include 'task' to select task (easy/medium/hard)

        Returns:
            Observation indicating the environment is ready
        """
        # Determine which task to use
        task_id = kwargs.get("task", "easy")
        if task_id not in TASKS:
            task_id = "easy"
        self._current_task = task_id

        # Get task config and create simulation
        config = get_task_config(task_id)
        if seed is not None:
            config.seed = seed
        self._sim = WarehouseSimulation(config)

        # Update state tracking
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        # Build initial observation info
        obs_data = self._sim.get_observation()

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task": task_id,
                "task_description": TASK_DESCRIPTIONS.get(task_id, ""),
                "grid_size": [config.grid_rows, config.grid_cols],
                "num_orders": len(config.orders),
                "num_items": len(config.items),
                "max_steps": config.max_steps,
                "warehouse": obs_data,
                "instructions": (
                    "You are a warehouse agent. Use the tools to navigate the warehouse, "
                    "pick items from shelves, and deliver them to fulfill orders. "
                    "Call view_warehouse() to see the layout. "
                    "Use move(direction) to navigate. "
                    "Use pick_item(item_id) when adjacent to a shelf with the item. "
                    "Use deliver_order(order_id) at the packing station."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (returns error)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use MCP tools: view_warehouse, move, pick_item, deliver_order."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step in the environment."""
        self._state.step_count += 1

        # Let base class handle MCP actions
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Update done status from simulation
        if self._sim and self._sim.done:
            obs.done = True
            obs.reward = self._sim.total_reward
            obs.metadata = obs.metadata or {}
            obs.metadata["final_score"] = self._sim.compute_score()

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step for WebSocket handler."""
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        if self._sim and self._sim.done:
            obs.done = True
            obs.reward = self._sim.total_reward
            obs.metadata = obs.metadata or {}
            obs.metadata["final_score"] = self._sim.compute_score()

        return obs

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
