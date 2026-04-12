"""
Warehouse RL Environment — Core Logic
=======================================
Implements the OpenEnv-compatible Environment with:

  • A rich 12×12 warehouse grid
  • 15 products across 3 priority tiers
  • Shelf obstacles + forklift collision zones
  • Battery management + recharge stations
  • Multi-phase reward shaping

Reward Function
---------------
  +value * priority_multiplier   on successful deposit
  -0.10                          per movement step  (energy drain)
  -0.05                          per 'stay' action
  -0.20                          if carrying weight > 1.5  (extra drain)
  -0.30                          battery warning (< 20 %)
  -1.00                          invalid action (wall / obstacle hit)
  -2.00                          repeated invalid actions (×consecutive)
  +5.00                          bonus for clearing ALL high-priority first
  +10.00                         episode completion bonus (all deposited)
  -5.00                          battery dead → episode failure
"""

from __future__ import annotations

import random
import uuid
from typing import List, Set, Tuple

from src.envs.warehouse_env.models import (
    Action,
    AgentState,
    CellType,
    Observation,
    ProductInfo,
    ProductState,
    State,
)

# Priority multipliers for reward shaping
PRIORITY_MULT = {"high": 2.0, "medium": 1.3, "low": 1.0}
PRIORITY_LEVEL = {"high": 3, "medium": 2, "low": 1}

# Energy costs per action
ENERGY_MOVE   = 0.05
ENERGY_STAY   = 0.02
ENERGY_CARRY  = 0.03   # extra per step when inventory non-empty
ENERGY_RECHARGE = 0.40  # energy restored per recharge action

MAX_STEPS = 200
GRID_ROWS = 12
GRID_COLS = 12
BATTERY_WARN = 0.20
INVENTORY_CAP = 3


class WarehouseEnvironment:
    """
    Core warehouse simulation.  Called by the FastAPI server.

    Methods
    -------
    reset()  → Observation
    step(action: Action) → Observation
    state    → State  (property)
    """

    def __init__(self):
        self._state: State = self._build_initial_state()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a fresh episode."""
        self._state = self._build_initial_state()
        return self._make_observation(reward=0.0, last_valid=True, msg="Episode started. Good luck!")

    def step(self, action: Action) -> Observation:
        """Execute one agent action and return the resulting observation."""
        s = self._state

        if s.done:
            return self._make_observation(
                reward=0.0, last_valid=False, msg="Episode already done. Call reset()."
            )

        s.step_count += 1
        reward = 0.0
        valid = True
        msg = ""

        # ---------------------------------------------------------------
        # 1. MOVEMENT
        # ---------------------------------------------------------------
        new_pos = self._apply_movement(s.agent.position, action.movement)

        if new_pos is None:
            # Stayed in place due to invalid direction (no movement cost)
            reward -= 1.0
            s.agent.consecutive_invalid += 1
            reward -= 0.30 * (s.agent.consecutive_invalid - 1)   # escalating
            valid = False
            msg = f"Invalid movement '{action.movement}': out of bounds or blocked."
        else:
            s.agent.consecutive_invalid = 0
            cost = ENERGY_MOVE if action.movement != "stay" else ENERGY_STAY
            cost += ENERGY_CARRY * len(s.agent.inventory)
            s.agent.battery = max(0.0, s.agent.battery - cost)
            s.agent.position = new_pos
            reward -= cost
            msg = f"Moved {action.movement} to {new_pos}."

        # ---------------------------------------------------------------
        # 2. BATTERY DEATH CHECK
        # ---------------------------------------------------------------
        if s.agent.battery <= 0.0:
            s.done = True
            s.success = False
            s.failure_reason = "Battery exhausted."
            reward -= 5.0
            return self._make_observation(reward=reward, last_valid=False, msg=s.failure_reason)

        # Battery warning penalty
        if s.agent.battery < BATTERY_WARN:
            reward -= 0.30

        # ---------------------------------------------------------------
        # 3. INTERACTION
        # ---------------------------------------------------------------
        if action.interact == "pickup":
            r, msg_i = self._handle_pickup(action)
            reward += r
            msg += " " + msg_i
            valid = valid and (r >= 0)

        elif action.interact == "deposit":
            r, msg_i = self._handle_deposit()
            reward += r
            msg += " " + msg_i
            valid = valid and (r >= 0)

        elif action.interact == "recharge":
            r, msg_i = self._handle_recharge()
            reward += r
            msg += " " + msg_i
            valid = valid and (r >= 0)

        # ---------------------------------------------------------------
        # 4. EPISODE TERMINATION
        # ---------------------------------------------------------------
        all_deposited = all(p.deposited for p in s.products)
        if all_deposited:
            reward += 10.0
            s.agent.total_reward += reward
            # Bonus if high-priority products were ALL cleared first
            high_order_bonus = self._check_priority_order_bonus()
            reward += high_order_bonus
            s.done = True
            s.success = True
            s.final_score = s.agent.total_reward
            msg += " 🎉 All products deposited! Episode complete!"

        elif s.step_count >= MAX_STEPS:
            s.done = True
            s.success = False
            s.failure_reason = "Maximum steps exceeded."
            msg += " ⏱️ Out of time."

        s.agent.total_reward += reward

        return self._make_observation(reward=round(reward, 4), last_valid=valid, msg=msg.strip())

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # INTERACTION HANDLERS
    # ------------------------------------------------------------------

    def _handle_pickup(self, action: Action) -> Tuple[float, str]:
        s = self._state
        pos = s.agent.position

        if len(s.agent.inventory) >= INVENTORY_CAP:
            s.agent.consecutive_invalid += 1
            return -1.0, "Inventory full (max 3 items). Deposit first."

        if not action.target_sku:
            return -0.5, "pickup action requires a target_sku."

        product = next((p for p in s.products if p.sku == action.target_sku), None)
        if product is None:
            return -0.5, f"Unknown SKU '{action.target_sku}'."
        if product.collected or product.deposited:
            return -0.5, f"SKU '{action.target_sku}' already collected/deposited."
        if not self._is_adjacent_or_same(pos, product.position):
            return -1.0, f"SKU '{action.target_sku}' is not adjacent (at {product.position})."

        product.collected = True
        s.agent.inventory.append(action.target_sku)
        s.agent.inventory_weight += product.weight
        return 0.10, f"Picked up {action.target_sku} ({product.priority} priority)."

    def _handle_deposit(self) -> Tuple[float, str]:
        s = self._state
        if s.agent.position != s.loading_dock:
            return -1.0, f"Must be at loading dock {s.loading_dock} to deposit (currently at {s.agent.position})."
        if not s.agent.inventory:
            return -0.5, "Nothing in inventory to deposit."

        total_reward = 0.0
        deposited_skus = []
        
        # Batch check for priority bonus violation
        if not s.agent.failed_priority_bonus:
            for sku in list(s.agent.inventory):
                prod_being_deposited = next(p for p in s.products if p.sku == sku)
                current_level = PRIORITY_LEVEL[prod_being_deposited.priority]
                
                # If there are ANY higher priority products that haven't been deposited yet
                # (and aren't being deposited in this exact batch), we broke the order.
                for other in s.products:
                    if (not other.deposited 
                        and other.sku not in s.agent.inventory
                        and PRIORITY_LEVEL[other.priority] > current_level):
                        s.agent.failed_priority_bonus = True
                        break

        for sku in list(s.agent.inventory):
            product = next(p for p in s.products if p.sku == sku)
            product.deposited = True
            total_reward += product.value * PRIORITY_MULT[product.priority]
            deposited_skus.append(sku)

        s.agent.inventory.clear()
        s.agent.inventory_weight = 0.0
        return total_reward, f"Deposited {deposited_skus}. Reward: +{total_reward:.2f}"

    def _handle_recharge(self) -> Tuple[float, str]:
        s = self._state
        if s.agent.position not in s.recharge_stations:
            return -1.0, f"Not on a recharge station. Stations at {s.recharge_stations}."
        before = s.agent.battery
        s.agent.battery = min(1.0, s.agent.battery + ENERGY_RECHARGE)
        gain = s.agent.battery - before
        return 0.0, f"Recharged +{gain:.0%}. Battery now {s.agent.battery:.0%}."

    # ------------------------------------------------------------------
    # MOVEMENT
    # ------------------------------------------------------------------

    DELTAS = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1), "stay": (0, 0)}

    def _apply_movement(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int] | None:
        dr, dc = self.DELTAS.get(direction, (0, 0))
        nr, nc = pos[0] + dr, pos[1] + dc

        if direction == "stay":
            return pos

        s = self._state
        if not (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS):
            return None
        if (nr, nc) in s.shelf_obstacles:
            return None
        if (nr, nc) in s.collision_zones:
            return None
        return (nr, nc)

    # ------------------------------------------------------------------
    # OBSERVATION BUILDER
    # ------------------------------------------------------------------

    def _make_observation(self, reward: float, last_valid: bool, msg: str) -> Observation:
        s = self._state
        ag = s.agent

        # 5×5 view centred on agent
        visible = self._compute_visible_cells(ag.position)

        products_deposited = sum(1 for p in s.products if p.deposited)
        products_remaining = sum(1 for p in s.products if not p.deposited)

        hint = self._generate_hint()

        return Observation(
            done=s.done,
            reward=reward,
            is_success=s.success,
            failure_reason=s.failure_reason,
            agent_position=ag.position,
            battery_level=round(ag.battery, 3),
            inventory=list(ag.inventory),
            inventory_weight=round(ag.inventory_weight, 2),
            carrying_capacity=INVENTORY_CAP,
            visible_cells=visible,
            loading_dock_position=s.loading_dock,
            recharge_stations=list(s.recharge_stations),
            products=[
                ProductInfo(
                    sku=p.sku,
                    priority=p.priority,
                    position=p.position,
                    weight=p.weight,
                    value=p.value,
                    collected=p.collected,
                )
                for p in s.products
            ],
            total_products=len(s.products),
            products_deposited=products_deposited,
            products_remaining=products_remaining,
            score=round(ag.total_reward, 3),
            steps_taken=s.step_count,
            steps_remaining=MAX_STEPS - s.step_count,
            hint=hint,
            last_action_valid=last_valid,
            last_action_message=msg,
        )

    def _compute_visible_cells(self, pos: Tuple[int, int]) -> List[CellType]:
        s = self._state
        cells = []
        r0, c0 = pos
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
                    continue
                p = (r, c)
                if p == pos:
                    kind = "agent"
                    sku = None
                elif p in s.shelf_obstacles:
                    kind = "shelf_obstacle"
                    sku = None
                elif p in s.collision_zones:
                    kind = "collision_zone"
                    sku = None
                elif p == s.loading_dock:
                    kind = "loading_dock"
                    sku = None
                elif p in s.recharge_stations:
                    kind = "recharge_station"
                    sku = None
                else:
                    product = next((prod for prod in s.products
                                    if prod.position == p and not prod.collected and not prod.deposited), None)
                    if product:
                        kind = "product"
                        sku = product.sku
                    else:
                        kind = "empty"
                        sku = None
                cells.append(CellType(position=p, kind=kind, sku=sku))
        return cells

    def _generate_hint(self) -> str:
        s = self._state
        ag = s.agent
        hints = []

        if ag.battery < 0.15:
            stations = s.recharge_stations
            hints.append(f"⚠️ CRITICAL battery ({ag.battery:.0%})! Nearest recharge: {stations}.")
        elif ag.battery < BATTERY_WARN:
            hints.append(f"🔋 Low battery ({ag.battery:.0%}). Consider recharging.")

        uncollected = [p for p in s.products if not p.collected and not p.deposited]
        if uncollected:
            high = [p for p in uncollected if p.priority == "high"]
            target = high[0] if high else uncollected[0]
            dist = abs(ag.position[0] - target.position[0]) + abs(ag.position[1] - target.position[1])
            hints.append(
                f"Nearest target: {target.sku} ({target.priority} priority) at {target.position}, "
                f"Manhattan dist={dist}."
            )

        if ag.inventory:
            hints.append(f"Carrying {len(ag.inventory)}/{INVENTORY_CAP} items: {ag.inventory}. "
                         f"Dock at {s.loading_dock}.")

        if not uncollected and ag.inventory:
            hints.append("All products collected! Head to the loading dock to deposit.")

        return " | ".join(hints) if hints else "All systems nominal."

    # ------------------------------------------------------------------
    # PRIORITY ORDER BONUS
    # ------------------------------------------------------------------

    def _check_priority_order_bonus(self) -> float:
        """Grant +5 if all items were deposited in strict priority order (high→med→low)."""
        if not self._state.agent.failed_priority_bonus:
            return 5.0
        return 0.0

    # ------------------------------------------------------------------
    # WORLD BUILDER
    # ------------------------------------------------------------------

    def _build_initial_state(self) -> State:
        random.seed(42)   # deterministic layout for reproducibility

        shelf_obstacles: Set[Tuple[int, int]] = {
            # Vertical shelf rows
            (1, 2), (2, 2), (3, 2),
            (1, 4), (2, 4), (3, 4),
            (1, 6), (2, 6), (3, 6),
            (1, 8), (2, 8), (3, 8),
            (5, 1), (6, 1), (7, 1),
            (5, 3), (6, 3), (7, 3),
            (5, 5), (6, 5), (7, 5),
            (5, 7), (6, 7), (7, 7),
            (5, 9), (6, 9), (7, 9),
            (9, 2), (10, 2),
            (9, 5), (10, 5),
            (9, 8), (10, 8),
        }

        collision_zones: Set[Tuple[int, int]] = {
            (4, 0), (4, 11), (8, 0), (8, 11),
            (0, 5), (11, 5),
        }

        loading_dock: Tuple[int, int] = (11, 11)
        recharge_stations: List[Tuple[int, int]] = [(0, 0), (0, 11), (11, 0)]

        # Ensure special cells aren't in obstacle sets
        for special in [loading_dock, *recharge_stations]:
            shelf_obstacles.discard(special)
            collision_zones.discard(special)

        # 15 products across 3 priority tiers
        product_defs = [
            # HIGH priority (5 products)
            ("SKU-H1", "high", (1, 1),  0.8, 4.0),
            ("SKU-H2", "high", (3, 3),  1.0, 4.5),
            ("SKU-H3", "high", (1, 7),  0.6, 3.8),
            ("SKU-H4", "high", (6, 2),  1.2, 4.2),
            ("SKU-H5", "high", (9, 1),  0.9, 4.0),
            # MEDIUM priority (5 products)
            ("SKU-M1", "medium", (2, 9),  0.5, 2.5),
            ("SKU-M2", "medium", (5, 6),  0.7, 2.8),
            ("SKU-M3", "medium", (7, 8),  0.4, 2.2),
            ("SKU-M4", "medium", (9, 4),  0.6, 2.6),
            ("SKU-M5", "medium", (3, 10), 0.8, 2.9),
            # LOW priority (5 products)
            ("SKU-L1", "low", (0, 3),  0.3, 1.0),
            ("SKU-L2", "low", (4, 6),  0.4, 1.2),
            ("SKU-L3", "low", (8, 3),  0.5, 1.1),
            ("SKU-L4", "low", (10, 7), 0.3, 1.0),
            ("SKU-L5", "low", (6, 10), 0.4, 1.3),
        ]

        # Validate product positions don't conflict with obstacles
        valid_products = []
        for sku, pri, pos, w, v in product_defs:
            if pos not in shelf_obstacles and pos not in collision_zones:
                valid_products.append(ProductState(sku=sku, priority=pri, position=pos, weight=w, value=v))

        agent = AgentState(
            position=(11, 0),          # starts bottom-left
            battery=1.0,
            inventory=[],
            inventory_weight=0.0,
        )

        return State(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            max_steps=MAX_STEPS,
            grid_rows=GRID_ROWS,
            grid_cols=GRID_COLS,
            shelf_obstacles=sorted(shelf_obstacles),
            collision_zones=sorted(collision_zones),
            loading_dock=loading_dock,
            recharge_stations=recharge_stations,
            agent=agent,
            products=valid_products,
        )

    def _is_adjacent_or_same(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1
