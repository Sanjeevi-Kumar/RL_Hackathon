"""
Warehouse Environment — An OpenEnv environment for warehouse order fulfillment.

This environment simulates warehouse operations where an AI agent must navigate
a grid-based warehouse, pick items from shelves, and deliver them to a packing
station to fulfill customer orders.

Available tools:
- view_warehouse(): See the full warehouse state
- move(direction): Navigate up/down/left/right
- pick_item(item_id): Pick items from adjacent shelves
- deliver_order(order_id): Deliver orders at packing station
- list_tasks(): See available task difficulties
- get_score(): Get graded score for current episode

Example:
    >>> from warehouse_env import WarehouseEnv
    >>>
    >>> with WarehouseEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(task="easy")
    ...     tools = env.list_tools()
    ...     result = env.call_tool("view_warehouse")
    ...     result = env.call_tool("move", direction="up")
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import WarehouseEnv

__all__ = ["WarehouseEnv", "CallToolAction", "ListToolsAction"]
