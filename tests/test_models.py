"""
Unit tests for models.py — Pydantic schema validation.

Covers:
  - SwarmAction field names, types, and constraints
  - DroneNodeAction validation (waypoint length, pump range, QoS enum)
  - SwarmObservation construction and reward bounds
  - SwarmState UUID validator
  - QoSProfile enum values
  - Extra-field rejection (model is strict about unknown keys)
"""

import pytest
from pydantic import ValidationError

from models import (
    DDSDataSpace,
    DroneNodeAction,
    QoSProfile,
    SwarmAction,
    SwarmObservation,
    SwarmState,
)


# ---------------------------------------------------------------------------
# QoSProfile
# ---------------------------------------------------------------------------

class TestQoSProfile:
    def test_valid_best_effort(self):
        assert QoSProfile("BEST_EFFORT") == QoSProfile.BEST_EFFORT

    def test_valid_reliable(self):
        assert QoSProfile("RELIABLE") == QoSProfile.RELIABLE

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            QoSProfile("UNKNOWN")


# ---------------------------------------------------------------------------
# DroneNodeAction
# ---------------------------------------------------------------------------

class TestDroneNodeAction:
    def _valid(self, **overrides):
        defaults = dict(
            agent_id="D0",
            target_waypoint=(5.0, 7.0, 10.0),
            pump_activation=0.0,
            broadcast_message=None,
            qos_profile=QoSProfile.BEST_EFFORT,
        )
        defaults.update(overrides)
        return DroneNodeAction(**defaults)

    def test_valid_construction(self):
        na = self._valid()
        assert na.agent_id == "D0"
        assert na.target_waypoint == (5.0, 7.0, 10.0)
        assert na.pump_activation == 0.0
        assert na.qos_profile == QoSProfile.BEST_EFFORT

    def test_pump_activation_upper_bound(self):
        na = self._valid(pump_activation=1.0)
        assert na.pump_activation == 1.0

    def test_pump_activation_below_zero_raises(self):
        with pytest.raises(ValidationError):
            self._valid(pump_activation=-0.1)

    def test_pump_activation_above_one_raises(self):
        with pytest.raises(ValidationError):
            self._valid(pump_activation=1.01)

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValidationError):
            self._valid(agent_id="")

    def test_broadcast_message_max_length(self):
        long_msg = "x" * 256
        na = self._valid(broadcast_message=long_msg)
        assert len(na.broadcast_message) == 256

    def test_broadcast_message_too_long_raises(self):
        with pytest.raises(ValidationError):
            self._valid(broadcast_message="x" * 257)

    def test_reliable_qos(self):
        na = self._valid(qos_profile=QoSProfile.RELIABLE)
        assert na.qos_profile == QoSProfile.RELIABLE


# ---------------------------------------------------------------------------
# SwarmAction
# ---------------------------------------------------------------------------

class TestSwarmAction:
    def _make_na(self, agent_id="D0"):
        return DroneNodeAction(
            agent_id=agent_id,
            target_waypoint=(0.0, 0.0, 10.0),
            pump_activation=0.0,
            broadcast_message=None,
            qos_profile=QoSProfile.BEST_EFFORT,
        )

    def test_single_drone(self):
        action = SwarmAction(node_actions=[self._make_na("D0")])
        assert len(action.node_actions) == 1
        assert action.node_actions[0].agent_id == "D0"

    def test_multi_drone(self):
        action = SwarmAction(node_actions=[self._make_na(f"D{i}") for i in range(5)])
        assert len(action.node_actions) == 5

    def test_empty_node_actions_allowed(self):
        # Empty list is structurally valid (server handles gracefully)
        action = SwarmAction(node_actions=[])
        assert action.node_actions == []

    def test_missing_node_actions_raises(self):
        with pytest.raises(ValidationError):
            SwarmAction()

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            SwarmAction(node_actions=[], unknown_field=True)


# ---------------------------------------------------------------------------
# SwarmObservation
# ---------------------------------------------------------------------------

class TestSwarmObservation:
    def test_default_construction(self):
        obs = SwarmObservation()
        assert obs.reward == 0.0
        assert obs.done is False
        assert obs.local_grid_thermal == []
        assert obs.neighbor_states == []

    def test_reward_lower_bound(self):
        with pytest.raises(ValidationError):
            SwarmObservation(reward=-0.01)

    def test_reward_upper_bound(self):
        with pytest.raises(ValidationError):
            SwarmObservation(reward=1.01)

    def test_reward_at_bounds(self):
        obs_low  = SwarmObservation(reward=0.0)
        obs_high = SwarmObservation(reward=1.0)
        assert obs_low.reward == 0.0
        assert obs_high.reward == 1.0

    def test_wind_vector_default(self):
        obs = SwarmObservation()
        assert obs.wind_vector == (0.0, 0.0)

    def test_done_flag(self):
        obs = SwarmObservation(done=True)
        assert obs.done is True


# ---------------------------------------------------------------------------
# DDSDataSpace
# ---------------------------------------------------------------------------

class TestDDSDataSpace:
    def test_default_construction(self):
        dds = DDSDataSpace()
        assert dds.active_peers == []
        assert dds.peer_telemetry == {}
        assert dds.gossip_messages == {}

    def test_populated(self):
        dds = DDSDataSpace(
            active_peers=["D0", "D1"],
            peer_telemetry={"D0": {"battery": 0.9}},
            gossip_messages={"D0": "hello"},
        )
        assert "D0" in dds.active_peers
        assert dds.peer_telemetry["D0"]["battery"] == 0.9


# ---------------------------------------------------------------------------
# SwarmState
# ---------------------------------------------------------------------------

class TestSwarmState:
    _VALID_UUID = "123e4567-e89b-4d3a-a456-426614174000"

    def test_valid_uuid4(self):
        state = SwarmState(episode_id=self._VALID_UUID, step_count=0)
        assert state.episode_id == self._VALID_UUID

    def test_invalid_uuid_raises(self):
        with pytest.raises(ValidationError):
            SwarmState(episode_id="not-a-uuid", step_count=0)

    def test_step_count_non_negative(self):
        with pytest.raises(ValidationError):
            SwarmState(episode_id=self._VALID_UUID, step_count=-1)

    def test_active_fires_non_negative(self):
        with pytest.raises(ValidationError):
            SwarmState(episode_id=self._VALID_UUID, step_count=0, active_fires=-1)

    def test_payload_levels_and_positions(self):
        state = SwarmState(
            episode_id=self._VALID_UUID,
            step_count=5,
            payload_levels={"D0": 8.5},
            drone_positions={"D0": [3.0, 4.0, 10.0]},
        )
        assert state.payload_levels["D0"] == 8.5
        assert state.drone_positions["D0"] == [3.0, 4.0, 10.0]
