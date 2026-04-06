"""
Pydantic models for documentation and type reference.

In the MCP-based architecture, the actual action/observation types used
are CallToolAction and CallToolObservation from openenv.core. These models
are provided for documentation and potential custom client implementations.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─── Action Types (for documentation) ───────────────────

class MoveAction(BaseModel):
    """Move the agent in the warehouse."""
    direction: str = Field(
        ...,
        description="Direction to move: 'up', 'down', 'left', or 'right'",
    )


class PickItemAction(BaseModel):
    """Pick an item from an adjacent shelf."""
    item_id: str = Field(
        ...,
        description="ID of the item to pick (e.g., 'item_1')",
    )


class DeliverOrderAction(BaseModel):
    """Deliver collected items for an order at the packing station."""
    order_id: str = Field(
        ...,
        description="ID of the order to deliver (e.g., 'order_1')",
    )


# ─── Observation Types (for documentation) ──────────────

class WarehouseObservation(BaseModel):
    """Full warehouse state observation."""
    agent_position: list[int] = Field(description="Agent position [row, col]")
    packing_station: list[int] = Field(description="Packing station position")
    inventory: list[str] = Field(description="Items currently held by agent")
    grid: list[list[str]] = Field(description="2D grid layout")
    grid_size: list[int] = Field(description="Grid dimensions [rows, cols]")
    items_on_shelves: list[dict] = Field(description="Available items on shelves")
    orders: list[dict] = Field(description="Order status and progress")
    step_count: int = Field(description="Current step number")
    max_steps: int = Field(description="Maximum allowed steps")
    steps_remaining: int = Field(description="Steps left in episode")
    total_reward: float = Field(description="Cumulative reward")
    done: bool = Field(description="Whether episode is finished")


class ActionResult(BaseModel):
    """Result from any action."""
    success: bool = Field(description="Whether the action succeeded")
    message: str = Field(description="Description of what happened")
    reward: float = Field(description="Reward for this action")
    new_position: Optional[list[int]] = Field(None, description="Agent position after move")
    inventory: Optional[list[str]] = Field(None, description="Updated inventory after pick")
    order_fulfilled: Optional[bool] = Field(None, description="Whether order was completed")


class ScoreResult(BaseModel):
    """Graded score for the episode."""
    score: float = Field(description="Score between 0.0 and 1.0")
    orders_delivered: int = Field(description="Number of orders fulfilled")
    total_orders: int = Field(description="Total number of orders")
    items_picked: int = Field(description="Number of items picked")
    steps_used: int = Field(description="Steps consumed")
    max_steps: int = Field(description="Maximum allowed steps")
    invalid_actions: int = Field(description="Number of invalid actions taken")
    done: bool = Field(description="Whether the episode ended")
