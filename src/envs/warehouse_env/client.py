"""
Warehouse RL Environment — HTTP Client
========================================
Provides a clean Python interface to the running FastAPI server.
Usage mirrors the OpenEnv pattern (step / reset / state).

Example
-------
    from src.envs.warehouse_env.client import WarehouseEnvClient
    from src.envs.warehouse_env.models import Action

    env = WarehouseEnvClient(base_url="http://localhost:8000")
    obs = env.reset()
    obs = env.step(Action(movement="north", interact="none"))
    state = env.state()
"""

from __future__ import annotations

from typing import Optional
import requests

from src.envs.warehouse_env.models import Action, Observation, State


class WarehouseEnvClient:
    """
    Type-safe HTTP client for the Warehouse RL environment server.

    Parameters
    ----------
    base_url : str
        URL where the FastAPI server is running, e.g. "http://localhost:8000".
    timeout : int
        Request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._last_obs: Optional[Observation] = None

    # ------------------------------------------------------------------
    # OPENENV STANDARD API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a new episode. Returns the initial observation."""
        resp = requests.post(f"{self.base_url}/reset", timeout=self.timeout)
        resp.raise_for_status()
        self._last_obs = Observation(**resp.json())
        return self._last_obs

    def step(self, action: Action) -> Observation:
        """
        Execute a single action and return the resulting observation.

        Parameters
        ----------
        action : Action
            Type-safe action object (movement + interact + optional SKU).
        """
        resp = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        self._last_obs = Observation(**resp.json())
        return self._last_obs

    def state(self) -> State:
        """Return the full internal state (for RL framework / debugging)."""
        resp = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return State(**resp.json())

    def health(self) -> dict:
        """Liveness check."""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # CONVENIENCE HELPERS
    # ------------------------------------------------------------------

    @property
    def last_observation(self) -> Optional[Observation]:
        """The most recent observation returned by reset() or step()."""
        return self._last_obs

    def move(self, direction: str) -> Observation:
        """Shorthand: move in a direction with no interaction."""
        return self.step(Action(movement=direction, interact="none"))  # type: ignore[arg-type]

    def pickup(self, sku: str, movement: str = "stay") -> Observation:
        """Shorthand: pick up a product by SKU (optionally move first)."""
        return self.step(Action(movement=movement, interact="pickup", target_sku=sku))  # type: ignore[arg-type]

    def deposit(self, movement: str = "stay") -> Observation:
        """Shorthand: deposit all held products."""
        return self.step(Action(movement=movement, interact="deposit"))  # type: ignore[arg-type]

    def recharge(self, movement: str = "stay") -> Observation:
        """Shorthand: recharge battery."""
        return self.step(Action(movement=movement, interact="recharge"))  # type: ignore[arg-type]
