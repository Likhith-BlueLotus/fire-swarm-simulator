"""
Shared pytest fixtures for the FireSwarm test suite.

All fixtures build against a deterministic seed=42 so tests are reproducible
across machines and CI runs.
"""

import sys
import os

# Ensure the project root is on sys.path so all modules resolve without
# installing the package (works both locally and inside Docker CI).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from models import DroneNodeAction, QoSProfile, SwarmAction
from server.environment import (
    ACTIVE_FIRE,
    EXTINGUISHED,
    HEALTHY_FUEL,
    TASK_CONFIG,
    FireSwarmEnvironment,
)
from server.app import app


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_easy():
    """Fresh FireSwarmEnvironment reset to task=easy, seed=42."""
    env = FireSwarmEnvironment()
    env.reset(task="easy", seed=42)
    return env


@pytest.fixture
def env_medium():
    """Fresh FireSwarmEnvironment reset to task=medium, seed=42."""
    env = FireSwarmEnvironment()
    env.reset(task="medium", seed=42)
    return env


@pytest.fixture
def env_hard():
    """Fresh FireSwarmEnvironment reset to task=hard, seed=42."""
    env = FireSwarmEnvironment()
    env.reset(task="hard", seed=42)
    return env


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def make_nop_action(drone_ids, pos_map=None) -> SwarmAction:
    """Build a do-nothing SwarmAction: drones hover at current position, no pump."""
    pos_map = pos_map or {}
    node_actions = [
        DroneNodeAction(
            agent_id=d_id,
            target_waypoint=(
                float(pos_map.get(d_id, (0, 0))[0]),
                float(pos_map.get(d_id, (0, 0))[1]),
                10.0,
            ),
            pump_activation=0.0,
            broadcast_message=None,
            qos_profile=QoSProfile.BEST_EFFORT,
        )
        for d_id in drone_ids
    ]
    return SwarmAction(node_actions=node_actions)


def make_pump_action(drone_id, tx, ty, throttle=1.0, qos=QoSProfile.BEST_EFFORT) -> SwarmAction:
    """Build a single-drone pump SwarmAction."""
    return SwarmAction(node_actions=[
        DroneNodeAction(
            agent_id=drone_id,
            target_waypoint=(float(tx), float(ty), 10.0),
            pump_activation=throttle,
            broadcast_message=None,
            qos_profile=qos,
        )
    ])


# ---------------------------------------------------------------------------
# FastAPI test client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient — shares the same process as the tests (no network)."""
    with TestClient(app) as client:
        yield client
