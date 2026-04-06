"""
Integration tests for server/app.py — FastAPI HTTP endpoints.

Uses FastAPI's TestClient (no real network, no external process needed).
All state is per-request since the OpenEnv HTTP layer is stateless.

Coverage:
  GET  /health    — status, version, metadata fields
  GET  /tasks     — all three tasks, correct drone/seed counts
  POST /reset     — all tasks, observation structure, drone counts
  POST /step      — correct action schema, reward range, done flag
  GET  /state     — episode_id, step_count
  POST /grade     — score range [0,1] for all tasks
  GET  /docs      — Swagger UI loads
  GET  /openapi.json — title, version, required routes
  Error handling  — 422 on malformed action, 422 on missing action field
"""

import pytest
from fastapi.testclient import TestClient

from server.app import app

# Module-scoped client: server starts once, all tests share it.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_version_correct(self, client):
        resp = client.get("/health")
        assert resp.json()["environment"]["version"] == "0.1.0"

    def test_ge_model_present(self, client):
        resp = client.get("/health")
        assert "Gilbert-Elliott" in resp.json()["environment"]["dds_model"]

    def test_max_concurrent_sessions(self, client):
        resp = client.get("/health")
        assert resp.json()["environment"]["max_concurrent_sessions"] == 4

    def test_websocket_endpoint_listed(self, client):
        resp = client.get("/health")
        assert resp.json()["websocket_endpoint"] == "/ws"

    def test_uptime_seconds_non_negative(self, client):
        resp = client.get("/health")
        assert resp.json()["uptime_seconds"] >= 0.0

    def test_pid_is_integer(self, client):
        resp = client.get("/health")
        assert isinstance(resp.json()["pid"], int)


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------

class TestTasks:
    def test_returns_three_tasks(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        tasks = {t["id"]: t for t in resp.json()["tasks"]}
        assert set(tasks.keys()) == {"easy", "medium", "hard"}

    @pytest.mark.parametrize("task,drones,fires,steps,grid", [
        ("easy",   1, 3,  30, 15),
        ("medium", 3, 5,  50, 20),
        ("hard",   5, 8,  70, 25),
    ])
    def test_task_metadata(self, client, task, drones, fires, steps, grid):
        resp = client.get("/tasks")
        tasks = {t["id"]: t for t in resp.json()["tasks"]}
        t = tasks[task]
        assert t["num_drones"]  == drones
        assert t["fire_seeds"]  == fires
        assert t["max_steps"]   == steps
        assert t["grid_size"]   == grid
        assert t["score_range"] == [0.0, 1.0]
        assert t["grader"]      == "programmatic"


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

class TestReset:
    def _reset(self, client, task, seed=42):
        return client.post("/reset", json={"task": task, "seed": seed})

    def test_reset_easy_200(self, client):
        resp = self._reset(client, "easy")
        assert resp.status_code == 200

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_reset_all_tasks(self, client, task):
        resp = self._reset(client, task)
        assert resp.status_code == 200

    def test_reset_returns_observation_wrapper(self, client):
        resp = self._reset(client, "easy")
        assert "observation" in resp.json()

    def test_reset_observation_has_fire_grid(self, client):
        resp = self._reset(client, "easy")
        obs = resp.json()["observation"]
        assert "local_grid_thermal" in obs

    def test_reset_observation_has_drone_telemetry(self, client):
        resp = self._reset(client, "easy")
        obs = resp.json()["observation"]
        assert "drone_telemetry" in obs

    def test_reset_observation_has_neighbor_states(self, client):
        resp = self._reset(client, "easy")
        obs = resp.json()["observation"]
        assert "neighbor_states" in obs

    def test_reset_observation_has_wind_vector(self, client):
        resp = self._reset(client, "easy")
        obs = resp.json()["observation"]
        assert "wind_vector" in obs

    def test_reset_done_is_false(self, client):
        resp = self._reset(client, "easy")
        assert resp.json()["done"] is False

    def test_reset_reward_is_zero(self, client):
        resp = self._reset(client, "easy")
        assert resp.json()["reward"] == 0.0

    @pytest.mark.parametrize("task,expected_alive", [
        ("easy",   1),
        ("medium", 3),
        ("hard",   5),
    ])
    def test_drone_count_by_task(self, client, task, expected_alive):
        resp = self._reset(client, task)
        alive = resp.json()["observation"]["drone_telemetry"]["alive_drones"]
        assert alive == float(expected_alive)

    @pytest.mark.parametrize("task,expected_drones", [
        ("medium", ["D0", "D1", "D2"]),
        ("hard",   ["D0", "D1", "D2", "D3", "D4"]),
    ])
    def test_per_drone_grids_keyed_correctly(self, client, task, expected_drones):
        resp = self._reset(client, task)
        grids = resp.json()["observation"]["per_drone_grids"]
        for d_id in expected_drones:
            assert d_id in grids

    def test_neighbor_states_count_easy(self, client):
        resp = self._reset(client, "easy")
        ns = resp.json()["observation"]["neighbor_states"]
        assert len(ns) == 1

    def test_neighbor_states_count_medium(self, client):
        resp = self._reset(client, "medium")
        ns = resp.json()["observation"]["neighbor_states"]
        assert len(ns) == 3

    def test_neighbor_states_count_hard(self, client):
        resp = self._reset(client, "hard")
        ns = resp.json()["observation"]["neighbor_states"]
        assert len(ns) == 5

    def test_deterministic_with_same_seed(self, client):
        r1 = self._reset(client, "easy", seed=99)
        r2 = self._reset(client, "easy", seed=99)
        f1 = r1.json()["observation"]["local_grid_thermal"]
        f2 = r2.json()["observation"]["local_grid_thermal"]
        assert f1 == f2


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------

EASY_NOP = {
    "action": {
        "node_actions": [
            {
                "agent_id": "D0",
                "target_waypoint": [5, 5, 10],
                "pump_activation": 0.0,
                "qos_profile": "BEST_EFFORT",
                "broadcast_message": None,
            }
        ]
    }
}

MEDIUM_NOP = {
    "action": {
        "node_actions": [
            {"agent_id": f"D{i}", "target_waypoint": [5, 5, 10],
             "pump_activation": 0.0, "qos_profile": "BEST_EFFORT",
             "broadcast_message": None}
            for i in range(3)
        ]
    }
}

HARD_NOP = {
    "action": {
        "node_actions": [
            {"agent_id": f"D{i}", "target_waypoint": [5, 5, 10],
             "pump_activation": 0.0, "qos_profile": "BEST_EFFORT",
             "broadcast_message": None}
            for i in range(5)
        ]
    }
}

PUMP_RELIABLE = {
    "action": {
        "node_actions": [
            {
                "agent_id": "D0",
                "target_waypoint": [10, 7, 10],
                "pump_activation": 1.0,
                "qos_profile": "RELIABLE",
                "broadcast_message": "status_ok",
            }
        ]
    }
}


class TestStep:
    def test_step_easy_nop_200(self, client):
        resp = client.post("/step", json=EASY_NOP)
        assert resp.status_code == 200

    def test_step_returns_observation(self, client):
        resp = client.post("/step", json=EASY_NOP)
        assert "observation" in resp.json()

    def test_step_returns_reward(self, client):
        resp = client.post("/step", json=EASY_NOP)
        assert "reward" in resp.json()

    def test_step_returns_done(self, client):
        resp = client.post("/step", json=EASY_NOP)
        assert "done" in resp.json()

    def test_step_reward_in_unit_interval(self, client):
        for _ in range(5):
            resp = client.post("/step", json=EASY_NOP)
            r = resp.json()["reward"]
            assert 0.0 <= r <= 1.0, f"reward {r} out of [0,1]"

    def test_step_medium_3_drones(self, client):
        resp = client.post("/step", json=MEDIUM_NOP)
        assert resp.status_code == 200
        assert "reward" in resp.json()

    def test_step_hard_5_drones(self, client):
        resp = client.post("/step", json=HARD_NOP)
        assert resp.status_code == 200
        assert "reward" in resp.json()

    def test_step_pump_reliable(self, client):
        resp = client.post("/step", json=PUMP_RELIABLE)
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert "neighbor_states" in obs

    def test_step_fire_grid_present(self, client):
        resp = client.post("/step", json=EASY_NOP)
        obs = resp.json()["observation"]
        assert "local_grid_thermal" in obs

    def test_step_missing_action_field_422(self, client):
        resp = client.post("/step", json={})
        assert resp.status_code == 422

    def test_step_malformed_action_422(self, client):
        resp = client.post("/step", json={"action": {"totally_wrong": 999}})
        assert resp.status_code == 422

    def test_step_wrong_field_names_422(self, client):
        # Old field names (drones/drone_id/waypoint/pump/qos) must fail validation
        bad_action = {
            "action": {
                "drones": [
                    {"drone_id": "D0", "waypoint": [5, 5, 10],
                     "pump": False, "qos": "BEST_EFFORT"}
                ]
            }
        }
        resp = client.post("/step", json=bad_action)
        assert resp.status_code == 422

    def test_step_pump_activation_out_of_range_422(self, client):
        bad = {
            "action": {
                "node_actions": [
                    {"agent_id": "D0", "target_waypoint": [5, 5, 10],
                     "pump_activation": 5.0,  # > 1.0
                     "qos_profile": "BEST_EFFORT", "broadcast_message": None}
                ]
            }
        }
        resp = client.post("/step", json=bad)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------

class TestState:
    def test_state_200(self, client):
        resp = client.get("/state")
        assert resp.status_code == 200

    def test_state_has_episode_id(self, client):
        resp = client.get("/state")
        assert "episode_id" in resp.json()

    def test_state_has_step_count(self, client):
        resp = client.get("/state")
        assert "step_count" in resp.json()

    def test_step_count_non_negative(self, client):
        resp = client.get("/state")
        assert resp.json()["step_count"] >= 0


# ---------------------------------------------------------------------------
# POST /grade/{task}
# ---------------------------------------------------------------------------

class TestGrade:
    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_grade_returns_score(self, client, task):
        resp = client.post(f"/grade/{task}", json={"seed": 42})
        assert resp.status_code == 200
        assert "score" in resp.json()

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_grade_score_in_range(self, client, task):
        resp = client.post(f"/grade/{task}", json={"seed": 42})
        score = resp.json()["score"]
        assert 0.0 <= score <= 1.0, f"score {score} out of [0,1] for task={task}"

    def test_grade_deterministic_same_seed(self, client):
        r1 = client.post("/grade/easy", json={"seed": 7})
        r2 = client.post("/grade/easy", json={"seed": 7})
        assert r1.json()["score"] == pytest.approx(r2.json()["score"])

    def test_grade_nop_baseline_never_perfect(self, client):
        """A zero-effort NOP baseline should not score 1.0."""
        resp = client.post("/grade/easy", json={"seed": 42})
        assert resp.json()["score"] < 1.0

    def test_grade_unknown_task_handled(self, client):
        resp = client.post("/grade/impossible_task_xyz", json={"seed": 1})
        # Should either 404 or return a score — must not crash with 500
        assert resp.status_code in (200, 404, 422)


# ---------------------------------------------------------------------------
# GET /docs and GET /openapi.json
# ---------------------------------------------------------------------------

class TestDocs:
    def test_swagger_ui_loads(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower()

    def test_openapi_json_title(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "FireSwarm" in resp.json()["info"]["title"]

    def test_openapi_json_version(self, client):
        resp = client.get("/openapi.json")
        assert resp.json()["info"]["version"] == "0.1.0"

    @pytest.mark.parametrize("route", ["/reset", "/step", "/health", "/tasks"])
    def test_openapi_routes_present(self, client, route):
        resp = client.get("/openapi.json")
        paths = list(resp.json()["paths"].keys())
        assert route in paths, f"{route} not in OpenAPI paths: {paths}"

    def test_grade_route_in_openapi(self, client):
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/grade/{task}" in paths
