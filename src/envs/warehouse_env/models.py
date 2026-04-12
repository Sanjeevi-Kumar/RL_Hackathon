"""
Warehouse RL Environment - Type-Safe Models
============================================
OpenEnv-compliant schemas using Pydantic for the
Multi-Product Warehouse Navigation Environment.

The agent must:
  1. Navigate a 12x12 grid warehouse
  2. Collect prioritised products (high → medium → low priority)
  3. Avoid shelf obstacles and forklift collision zones
  4. Deposit collected items at the loading dock
  5. Manage battery / energy (depletes per step, recharge stations available)

This multi-objective, multi-phase structure forces the LLM agent to reason
across many turns — exactly what OpenEnv judges want to see.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# ACTION
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    What the agent sends to the environment each turn.

    movement   : Cardinal movement OR special interaction.
    interact   : Whether to PICK UP a product or DEPOSIT at dock.
    target_sku : If picking up, which product SKU to grab (must be adjacent).
    metadata   : Free dict for debugging / logging.
    """

    movement: Literal["north", "south", "east", "west", "stay"] = Field(
        ...,
        description="Direction to move. 'stay' keeps position (costs 0.05 energy).",
    )
    interact: Literal["pickup", "deposit", "recharge", "none"] = Field(
        default="none",
        description=(
            "'pickup'  – grab an adjacent product (must specify target_sku).\n"
            "'deposit' – deposit held products at the loading dock (must be on dock).\n"
            "'recharge'– recharge battery (must be on a recharge station).\n"
            "'none'    – no interaction this turn."
        ),
    )
    target_sku: Optional[str] = Field(
        default=None,
        description="SKU identifier of the product to pick up. Required when interact='pickup'.",
    )
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

class ProductInfo(BaseModel):
    """Visible information about a single product in the warehouse."""

    sku: str
    priority: Literal["high", "medium", "low"]
    position: Tuple[int, int]          # (row, col) in grid
    weight: float                      # affects energy cost when carried
    value: float                       # reward granted on deposit
    collected: bool = False


class CellType(BaseModel):
    """Compact description of a single visible grid cell."""

    position: Tuple[int, int]
    kind: Literal[
        "empty", "shelf_obstacle", "collision_zone",
        "product", "loading_dock", "recharge_station", "agent"
    ]
    sku: Optional[str] = None          # populated when kind == "product"


class Observation(BaseModel):
    """
    What the agent observes after each step.

    Includes:
    • Immediate neighbourhood (5×5 patch centred on agent)
    • Full product manifest (location + status of every item)
    • Agent telemetry (position, battery, inventory)
    • Task progress counters
    • Textual hint for LLM reasoning
    """

    # --- done / reward (mirrors OpenEnv StepResult) ---
    done: bool = False
    reward: float = 0.0
    is_success: bool = False
    failure_reason: Optional[str] = None

    # --- agent telemetry ---
    agent_position: Tuple[int, int]
    battery_level: float = Field(..., ge=0.0, le=1.0, description="0=dead, 1=full")
    inventory: List[str] = Field(
        default_factory=list,
        description="SKUs currently held by the agent (max capacity = 3).",
    )
    inventory_weight: float = 0.0
    carrying_capacity: int = 3

    # --- environment layout (partial observation) ---
    visible_cells: List[CellType] = Field(
        description="All cells in the 5×5 view centred on the agent."
    )
    loading_dock_position: Tuple[int, int]
    recharge_stations: List[Tuple[int, int]]

    # --- product manifest (global, always visible) ---
    products: List[ProductInfo] = Field(
        description="Full list of all products and their current status."
    )

    # --- task progress ---
    total_products: int
    products_deposited: int
    products_remaining: int
    score: float = 0.0
    steps_taken: int = 0
    steps_remaining: int

    # --- LLM reasoning hint ---
    hint: str = Field(
        description=(
            "Natural-language summary of the current situation to aid LLM reasoning. "
            "E.g., nearest high-priority product, battery warning, etc."
        )
    )

    # --- last action outcome ---
    last_action_valid: bool = True
    last_action_message: str = ""

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# STATE  (internal tracking — richer than Observation)
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    position: Tuple[int, int]
    battery: float
    inventory: List[str]
    inventory_weight: float
    total_reward: float = 0.0
    consecutive_invalid: int = 0      # penalty escalation counter
    failed_priority_bonus: bool = False # set to true if deposited lower priority before higher


class ProductState(BaseModel):
    sku: str
    priority: Literal["high", "medium", "low"]
    position: Tuple[int, int]
    weight: float
    value: float
    collected: bool = False
    deposited: bool = False


class State(BaseModel):
    """
    Full internal state. Never sent to the agent; used server-side only.
    Exposed via GET /state for debugging and RL framework inspection.
    """

    episode_id: str
    step_count: int = 0
    max_steps: int = 200

    # grid dimensions
    grid_rows: int = 12
    grid_cols: int = 12

    # static layout sets (stored as list-of-tuples for JSON serializability)
    shelf_obstacles: List[Tuple[int, int]]
    collision_zones: List[Tuple[int, int]]
    loading_dock: Tuple[int, int]
    recharge_stations: List[Tuple[int, int]]

    # dynamic entities
    agent: AgentState
    products: List[ProductState]

    # episode outcome
    done: bool = False
    success: bool = False
    failure_reason: Optional[str] = None
    final_score: float = 0.0

    model_config = {"extra": "ignore"}
