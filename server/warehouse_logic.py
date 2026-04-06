"""
Core warehouse simulation logic.

This module contains the pure simulation logic for the warehouse environment,
with no OpenEnv dependencies. It handles:
- Grid-based warehouse layout with shelves and items
- Agent movement and item picking
- Order fulfillment tracking
- Reward calculation with partial progress signals
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CellType(str, Enum):
    EMPTY = "empty"
    SHELF = "shelf"
    PACKING_STATION = "packing"
    AGENT = "agent"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Item:
    item_id: str
    name: str
    shelf_position: tuple[int, int]  # (row, col) of the shelf holding this item


@dataclass
class Order:
    order_id: str
    required_items: list[str]  # list of item_ids
    priority: int = 1  # 1=normal, 2=high, 3=urgent
    fulfilled: bool = False
    items_collected: list[str] = field(default_factory=list)

    @property
    def progress(self) -> float:
        if not self.required_items:
            return 1.0
        return len(self.items_collected) / len(self.required_items)

    @property
    def is_ready(self) -> bool:
        """All items collected, ready for delivery."""
        return set(self.items_collected) >= set(self.required_items)


@dataclass
class WarehouseConfig:
    grid_rows: int = 8
    grid_cols: int = 8
    max_steps: int = 60
    orders: list[dict] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    seed: int = 42


class WarehouseSimulation:
    """Core warehouse simulation with grid-based movement and item management."""

    def __init__(self, config: WarehouseConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.grid: list[list[CellType]] = []
        self.agent_pos: tuple[int, int] = (0, 0)
        self.packing_pos: tuple[int, int] = (0, 0)
        self.items: dict[str, Item] = {}
        self.orders: dict[str, Order] = {}
        self.inventory: list[str] = []  # items the agent is carrying
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.done: bool = False
        self.invalid_actions: int = 0
        self.items_picked: int = 0
        self.orders_delivered: int = 0
        self.action_log: list[dict] = []

        self._initialize()

    def _initialize(self):
        """Set up the warehouse grid, shelves, items, and orders."""
        rows, cols = self.config.grid_rows, self.config.grid_cols

        # Create empty grid
        self.grid = [[CellType.EMPTY for _ in range(cols)] for _ in range(rows)]

        # Place packing station at bottom-left
        self.packing_pos = (rows - 1, 0)
        self.grid[self.packing_pos[0]][self.packing_pos[1]] = CellType.PACKING_STATION

        # Agent starts at packing station
        self.agent_pos = self.packing_pos

        # Place shelves in a grid pattern (every other row, with aisles)
        shelf_positions = []
        for r in range(1, rows - 1, 2):
            for c in range(1, cols - 1):
                if not (r == self.packing_pos[0] and c == self.packing_pos[1]):
                    self.grid[r][c] = CellType.SHELF
                    shelf_positions.append((r, c))

        # Place items on shelves
        if self.config.items:
            for item_def in self.config.items:
                pos = tuple(item_def["position"])
                item = Item(
                    item_id=item_def["item_id"],
                    name=item_def["name"],
                    shelf_position=pos,
                )
                self.items[item.item_id] = item
        else:
            # Auto-generate items on random shelves
            item_names = [
                "Widget A", "Widget B", "Gadget C", "Part D", "Component E",
                "Module F", "Sensor G", "Battery H", "Cable I", "Board J",
            ]
            available_shelves = list(shelf_positions)
            self.rng.shuffle(available_shelves)
            for i, name in enumerate(item_names):
                if i >= len(available_shelves):
                    break
                pos = available_shelves[i]
                item = Item(
                    item_id=f"item_{i+1}",
                    name=name,
                    shelf_position=pos,
                )
                self.items[item.item_id] = item

        # Create orders
        if self.config.orders:
            for order_def in self.config.orders:
                order = Order(
                    order_id=order_def["order_id"],
                    required_items=order_def["required_items"],
                    priority=order_def.get("priority", 1),
                )
                self.orders[order.order_id] = order

    def get_observation(self) -> dict:
        """Return the current observation of the warehouse state."""
        # Build grid representation
        grid_repr = []
        for r in range(self.config.grid_rows):
            row = []
            for c in range(self.config.grid_cols):
                cell = self.grid[r][c].value
                # Annotate agent position
                if (r, c) == self.agent_pos:
                    cell = "agent"
                # Annotate items on shelves
                row.append(cell)
            grid_repr.append(row)

        # Items on shelves (not yet picked)
        items_on_shelves = []
        for item_id, item in self.items.items():
            if item_id not in self.inventory and not any(
                item_id in o.items_collected for o in self.orders.values()
            ):
                items_on_shelves.append({
                    "item_id": item.item_id,
                    "name": item.name,
                    "shelf_position": list(item.shelf_position),
                })

        # Order status
        order_status = []
        for order in self.orders.values():
            order_status.append({
                "order_id": order.order_id,
                "required_items": order.required_items,
                "items_collected": order.items_collected,
                "priority": order.priority,
                "fulfilled": order.fulfilled,
                "progress": round(order.progress, 2),
                "ready_to_deliver": order.is_ready,
            })

        return {
            "agent_position": list(self.agent_pos),
            "packing_station": list(self.packing_pos),
            "inventory": list(self.inventory),
            "grid": grid_repr,
            "grid_size": [self.config.grid_rows, self.config.grid_cols],
            "items_on_shelves": items_on_shelves,
            "orders": order_status,
            "step_count": self.step_count,
            "max_steps": self.config.max_steps,
            "steps_remaining": self.config.max_steps - self.step_count,
            "total_reward": round(self.total_reward, 4),
            "done": self.done,
        }

    def move(self, direction: str) -> dict:
        """Move the agent in the given direction. Returns action result."""
        if self.done:
            return {"success": False, "message": "Episode is done.", "reward": 0.0}

        self.step_count += 1
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}.get(
            direction.lower(), (0, 0)
        )

        if dr == 0 and dc == 0:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": f"Invalid direction: {direction}. Use up/down/left/right.",
                "reward": reward,
                "new_position": list(self.agent_pos),
            }

        new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # Check bounds
        if not (0 <= new_r < self.config.grid_rows and 0 <= new_c < self.config.grid_cols):
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": "Cannot move there — wall!",
                "reward": reward,
                "new_position": list(self.agent_pos),
            }

        # Check if shelf is blocking (agent can't walk onto shelves)
        if self.grid[new_r][new_c] == CellType.SHELF:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": "Cannot walk onto a shelf! Move around it.",
                "reward": reward,
                "new_position": list(self.agent_pos),
            }

        # Move agent
        self.agent_pos = (new_r, new_c)
        reward = -0.01  # small time cost
        self.total_reward += reward
        self._check_done()

        return {
            "success": True,
            "message": f"Moved {direction} to ({new_r}, {new_c}).",
            "reward": reward,
            "new_position": list(self.agent_pos),
        }

    def pick_item(self, item_id: str) -> dict:
        """Pick an item from an adjacent shelf."""
        if self.done:
            return {"success": False, "message": "Episode is done.", "reward": 0.0}

        self.step_count += 1

        # Check if item exists
        if item_id not in self.items:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": f"Item '{item_id}' does not exist.",
                "reward": reward,
            }

        # Check if already picked
        if item_id in self.inventory:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": f"Item '{item_id}' is already in your inventory.",
                "reward": reward,
            }

        item = self.items[item_id]

        # Check adjacency: agent must be next to the shelf
        ar, ac = self.agent_pos
        sr, sc = item.shelf_position
        if abs(ar - sr) + abs(ac - sc) > 1:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": f"You must be adjacent to the shelf at ({sr},{sc}) to pick '{item_id}'. "
                           f"Your position: ({ar},{ac}).",
                "reward": reward,
            }

        # Pick the item
        self.inventory.append(item_id)
        self.items_picked += 1
        reward = 0.1
        self.total_reward += reward
        self._check_done()

        return {
            "success": True,
            "message": f"Picked up '{item.name}' ({item_id}).",
            "reward": reward,
            "inventory": list(self.inventory),
        }

    def deliver_order(self, order_id: str) -> dict:
        """Deliver an order at the packing station."""
        if self.done:
            return {"success": False, "message": "Episode is done.", "reward": 0.0}

        self.step_count += 1

        # Must be at packing station
        if self.agent_pos != self.packing_pos:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": f"You must be at the packing station {list(self.packing_pos)} to deliver. "
                           f"Your position: {list(self.agent_pos)}.",
                "reward": reward,
            }

        # Check order exists
        if order_id not in self.orders:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": f"Order '{order_id}' does not exist.",
                "reward": reward,
            }

        order = self.orders[order_id]

        if order.fulfilled:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            return {
                "success": False,
                "message": f"Order '{order_id}' is already fulfilled.",
                "reward": reward,
            }

        # Transfer matching items from inventory to order
        items_transferred = []
        for req_item in order.required_items:
            if req_item in self.inventory and req_item not in order.items_collected:
                self.inventory.remove(req_item)
                order.items_collected.append(req_item)
                items_transferred.append(req_item)

        if not items_transferred:
            self.invalid_actions += 1
            reward = -0.05
            self.total_reward += reward
            self._check_done()
            missing = [i for i in order.required_items if i not in order.items_collected]
            return {
                "success": False,
                "message": f"No matching items in inventory for order '{order_id}'. "
                           f"Still need: {missing}.",
                "reward": reward,
            }

        # Check if order is now complete
        reward = 0.0
        if order.is_ready:
            order.fulfilled = True
            self.orders_delivered += 1
            reward = 0.3  # base delivery reward

            # Priority bonus: deliver high-priority orders early
            steps_fraction = self.step_count / self.config.max_steps
            if order.priority >= 2 and steps_fraction < 0.5:
                reward += 0.2  # early high-priority bonus
            elif order.priority >= 3 and steps_fraction < 0.3:
                reward += 0.1  # extra urgent bonus

            self.total_reward += reward
            self._check_done()
            return {
                "success": True,
                "message": f"Order '{order_id}' fulfilled! Items delivered: {items_transferred}.",
                "reward": reward,
                "order_fulfilled": True,
            }
        else:
            # Partial delivery
            reward = 0.05 * len(items_transferred)
            self.total_reward += reward
            self._check_done()
            remaining = [i for i in order.required_items if i not in order.items_collected]
            return {
                "success": True,
                "message": f"Partial delivery for order '{order_id}'. "
                           f"Delivered: {items_transferred}. Still need: {remaining}.",
                "reward": reward,
                "order_fulfilled": False,
            }

    def _check_done(self):
        """Check if episode should end."""
        # All orders fulfilled
        if all(o.fulfilled for o in self.orders.values()):
            self.done = True
            return

        # Max steps reached
        if self.step_count >= self.config.max_steps:
            self.done = True
            return

    def compute_score(self) -> float:
        """
        Compute final normalized score in [0.0, 1.0].

        Scoring components:
        - 60% Orders fulfilled ratio
        - 20% Items picked ratio (for unfulfilled orders, partial credit)
        - 10% Efficiency (fewer steps = better)
        - 10% Penalty for invalid actions
        """
        total_orders = len(self.orders)
        if total_orders == 0:
            return 1.0

        # Orders fulfilled ratio (60%)
        fulfilled_ratio = self.orders_delivered / total_orders
        order_score = fulfilled_ratio * 0.6

        # Items progress ratio (20%) - credit for partially fulfilled orders
        total_items_needed = sum(len(o.required_items) for o in self.orders.values())
        total_items_collected = sum(len(o.items_collected) for o in self.orders.values())
        items_ratio = total_items_collected / max(total_items_needed, 1)
        items_score = items_ratio * 0.2

        # Efficiency (10%) - bonus for using fewer steps
        steps_ratio = 1.0 - (self.step_count / self.config.max_steps)
        efficiency_score = max(0.0, steps_ratio) * 0.1

        # Invalid action penalty (10%)
        max_invalid = 10  # after 10 invalid actions, this component is 0
        invalid_ratio = min(self.invalid_actions / max_invalid, 1.0)
        valid_score = (1.0 - invalid_ratio) * 0.1

        score = order_score + items_score + efficiency_score + valid_score
        return round(min(max(score, 0.0), 1.0), 4)
