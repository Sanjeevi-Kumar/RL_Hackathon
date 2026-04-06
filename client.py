"""
Warehouse Environment Client.

This module provides the client for connecting to a Warehouse Environment server.
WarehouseEnv extends MCPToolClient for tool-calling style interactions.

Example:
    >>> with WarehouseEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(task="easy")
    ...     tools = env.list_tools()
    ...     result = env.call_tool("view_warehouse")
    ...     result = env.call_tool("move", direction="up")
    ...     result = env.call_tool("pick_item", item_id="item_1")
    ...     result = env.call_tool("deliver_order", order_id="order_1")
"""

from openenv.core.mcp_client import MCPToolClient


class WarehouseEnv(MCPToolClient):
    """
    Client for the Warehouse Environment.

    Provides a simple interface for interacting with the warehouse
    environment via MCP tools. Inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action

    Available tools:
    - `view_warehouse()`: See warehouse grid, items, orders, agent position
    - `move(direction)`: Move agent (up/down/left/right)
    - `pick_item(item_id)`: Pick an item from adjacent shelf
    - `deliver_order(order_id)`: Deliver order at packing station
    - `list_tasks()`: See available tasks
    - `get_score()`: Get current graded score

    Example:
        >>> with WarehouseEnv(base_url="http://localhost:8000") as env:
        ...     env.reset(task="medium")
        ...     warehouse = env.call_tool("view_warehouse")
        ...     env.call_tool("move", direction="up")
        ...     env.call_tool("pick_item", item_id="item_1")
        ...     env.call_tool("deliver_order", order_id="order_1")
        ...     score = env.call_tool("get_score")
    """

    pass  # MCPToolClient provides all needed functionality
