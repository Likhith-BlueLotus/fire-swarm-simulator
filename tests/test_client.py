"""
Unit tests for client.py — FireSwarmEnv wire-format helpers.

Tests the three private methods that adapt between the OpenEnv wire protocol
(raw JSON dicts) and typed Pydantic models, without requiring a live server.
"""

import pytest

from client import FireSwarmEnv
from models import (
    DDSDataSpace,
    DroneNodeAction,
    QoSProfile,
    SwarmAction,
    SwarmObservation,
    SwarmState,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid payloads mirroring the server's actual responses
# ---------------------------------------------------------------------------

def _minimal_step_response(reward=0.1, done=False):
    """Mimic a /step JSON response envelope from the server."""
    return {
        "reward": reward,
        "done": done,
        "observation": {
            "local_grid_thermal": [[0.0] * 15 for _ in range(15)],
            "per_drone_grids": {"D0": [[0.0] * 15 for _ in range(15)]},
            "drone_telemetry": {
                "avg_battery": 0.95,
                "min_battery": 0.95,
                "avg_payload": 10.0,
                "min_payload": 10.0,
                "active_fires": 3.0,
                "burned_area": 0.0,
                "alive_drones": 1.0,
                "retx_count": 0.0,
            },
            "neighbor_states": [
                {"id": "D0", "pos": [3, 4], "alt": 10.0,
                 "battery": 0.95, "payload": 10.0, "channel": "Good"}
            ],
            "dds_global_space": {
                "active_peers": ["D0"],
                "peer_telemetry": {"D0": {"battery": 0.95, "payload": 10.0,
                                          "pos": [3, 4], "alt": 10.0, "loiter": 0}},
                "gossip_messages": {},
            },
            "wind_vector": [1.0, 45.0],
        },
    }


def _minimal_state_response():
    """Mimic a /state JSON response from the server."""
    return {
        "episode_id": "123e4567-e89b-4d3a-a456-426614174000",
        "step_count": 3,
        "total_burned_area": 0,
        "active_fires": 3,
        "global_wind_vector": [1.0, 45.0],
        "payload_levels": {"D0": 9.5},
        "drone_positions": {"D0": [3.0, 4.0, 10.0]},
    }


# ---------------------------------------------------------------------------
# _step_payload — SwarmAction → wire dict
# ---------------------------------------------------------------------------

class TestStepPayload:
    def _make_env(self):
        # FireSwarmEnv requires a base_url; we never connect — just test serialisation
        return FireSwarmEnv.__new__(FireSwarmEnv)

    def _action(self, agent_id="D0", tx=5, ty=7, pump=0.0, qos=QoSProfile.BEST_EFFORT):
        return SwarmAction(node_actions=[
            DroneNodeAction(
                agent_id=agent_id,
                target_waypoint=(float(tx), float(ty), 10.0),
                pump_activation=pump,
                broadcast_message=None,
                qos_profile=qos,
            )
        ])

    def test_output_has_node_actions_key(self):
        env = self._make_env()
        payload = env._step_payload(self._action())
        assert "node_actions" in payload

    def test_single_drone_serialised(self):
        env = self._make_env()
        payload = env._step_payload(self._action())
        assert len(payload["node_actions"]) == 1

    def test_agent_id_preserved(self):
        env = self._make_env()
        payload = env._step_payload(self._action(agent_id="D3"))
        assert payload["node_actions"][0]["agent_id"] == "D3"

    def test_target_waypoint_preserved(self):
        env = self._make_env()
        payload = env._step_payload(self._action(tx=8, ty=12))
        wp = payload["node_actions"][0]["target_waypoint"]
        assert wp[0] == pytest.approx(8.0)
        assert wp[1] == pytest.approx(12.0)

    def test_pump_activation_preserved(self):
        env = self._make_env()
        payload = env._step_payload(self._action(pump=1.0))
        assert payload["node_actions"][0]["pump_activation"] == pytest.approx(1.0)

    def test_qos_profile_preserved(self):
        env = self._make_env()
        payload = env._step_payload(self._action(qos=QoSProfile.RELIABLE))
        assert payload["node_actions"][0]["qos_profile"] == "RELIABLE"

    def test_multi_drone_action_serialised(self):
        env = self._make_env()
        action = SwarmAction(node_actions=[
            DroneNodeAction(
                agent_id=f"D{i}",
                target_waypoint=(float(i), float(i), 10.0),
                pump_activation=0.0,
                broadcast_message=None,
                qos_profile=QoSProfile.BEST_EFFORT,
            )
            for i in range(5)
        ])
        payload = env._step_payload(action)
        assert len(payload["node_actions"]) == 5
        ids = [na["agent_id"] for na in payload["node_actions"]]
        assert set(ids) == {"D0", "D1", "D2", "D3", "D4"}


# ---------------------------------------------------------------------------
# _parse_result — wire dict → StepResult[SwarmObservation]
# ---------------------------------------------------------------------------

class TestParseResult:
    def _make_env(self):
        return FireSwarmEnv.__new__(FireSwarmEnv)

    def test_reward_parsed(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response(reward=0.42))
        assert result.reward == pytest.approx(0.42)

    def test_done_false(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response(done=False))
        assert result.done is False

    def test_done_true(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response(done=True))
        assert result.done is True

    def test_observation_is_swarm_observation(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response())
        assert isinstance(result.observation, SwarmObservation)

    def test_local_grid_thermal_shape(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response())
        grid = result.observation.local_grid_thermal
        assert len(grid) == 15
        assert len(grid[0]) == 15

    def test_neighbor_states_parsed(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response())
        ns = result.observation.neighbor_states
        assert len(ns) == 1
        assert ns[0]["id"] == "D0"

    def test_wind_vector_parsed(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response())
        wv = result.observation.wind_vector
        assert len(wv) == 2
        assert wv[0] == pytest.approx(1.0)
        assert wv[1] == pytest.approx(45.0)

    def test_dds_space_parsed(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response())
        dds = result.observation.dds_global_space
        assert isinstance(dds, DDSDataSpace)
        assert "D0" in dds.active_peers

    def test_missing_observation_key_graceful(self):
        env = self._make_env()
        payload = {"reward": 0.0, "done": False}  # no "observation" key
        result = env._parse_result(payload)
        assert result.observation.local_grid_thermal == []

    def test_drone_telemetry_parsed(self):
        env = self._make_env()
        result = env._parse_result(_minimal_step_response())
        telem = result.observation.drone_telemetry
        assert "avg_battery" in telem
        assert "alive_drones" in telem


# ---------------------------------------------------------------------------
# _parse_state — wire dict → SwarmState
# ---------------------------------------------------------------------------

class TestParseState:
    def _make_env(self):
        return FireSwarmEnv.__new__(FireSwarmEnv)

    def test_returns_swarm_state(self):
        env = self._make_env()
        state = env._parse_state(_minimal_state_response())
        assert isinstance(state, SwarmState)

    def test_episode_id_parsed(self):
        env = self._make_env()
        state = env._parse_state(_minimal_state_response())
        assert state.episode_id == "123e4567-e89b-4d3a-a456-426614174000"

    def test_step_count_parsed(self):
        env = self._make_env()
        state = env._parse_state(_minimal_state_response())
        assert state.step_count == 3

    def test_active_fires_parsed(self):
        env = self._make_env()
        state = env._parse_state(_minimal_state_response())
        assert state.active_fires == 3

    def test_total_burned_area_parsed(self):
        env = self._make_env()
        state = env._parse_state(_minimal_state_response())
        assert state.total_burned_area == 0

    def test_global_wind_vector_parsed(self):
        env = self._make_env()
        state = env._parse_state(_minimal_state_response())
        wv = state.global_wind_vector
        assert wv[0] == pytest.approx(1.0)
        assert wv[1] == pytest.approx(45.0)

    def test_payload_levels_parsed(self):
        env = self._make_env()
        state = env._parse_state(_minimal_state_response())
        assert state.payload_levels["D0"] == pytest.approx(9.5)

    def test_drone_positions_parsed(self):
        env = self._make_env()
        state = env._parse_state(_minimal_state_response())
        pos = state.drone_positions["D0"]
        assert pos == [3.0, 4.0, 10.0]

    def test_missing_fields_use_defaults(self):
        env = self._make_env()
        minimal = {
            "episode_id": "123e4567-e89b-4d3a-a456-426614174000",
            "step_count": 0,
        }
        state = env._parse_state(minimal)
        assert state.active_fires == 0
        assert state.total_burned_area == 0
        assert state.payload_levels == {}
        assert state.drone_positions == {}
