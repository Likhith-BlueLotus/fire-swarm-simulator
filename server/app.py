"""
FireSwarm FastAPI server — v0.1.0

Exposes the OpenEnv-standard HTTP + WebSocket surface:

  POST /reset         start a new episode
  POST /step          advance one simulation tick
  GET  /state         read current SwarmState (for graders and loggers)
  GET  /health        rich readiness probe
  GET  /tasks         enumerate all graded tasks
  POST /grade/{task}  run headless programmatic grader
  WS   /ws            real-time WebSocket for low-latency agents
  GET  /docs          Swagger UI
  GET  /redoc         ReDoc UI
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openenv.core.env_server import ConcurrencyConfig, create_fastapi_app

from .environment import TASK_CONFIG, FireSwarmEnvironment

try:
    from ..models import DroneNodeAction, QoSProfile, SwarmAction, SwarmObservation
except ImportError:
    from models import DroneNodeAction, QoSProfile, SwarmAction, SwarmObservation  # type: ignore[no-redef]

# Populated by the lifespan handler at startup; used to report uptime in /health.
_SERVER_START_TIME: float = 0.0

_concurrency = ConcurrencyConfig(
    max_concurrent_envs=4,
    session_timeout=300.0,
)

app: FastAPI = create_fastapi_app(
    env=FireSwarmEnvironment,
    action_cls=SwarmAction,
    observation_cls=SwarmObservation,
    concurrency_config=_concurrency,
)

app.title       = "FireSwarm MARL Environment"
app.version     = "0.1.0"
app.description = (
    "Decentralised swarm simulator for firefighting UAVs. "
    "Implements Cellular Automata fire spread, Gilbert-Elliott DDS network model, "
    "and a programmatic Anti-Hacking Grader with loiter and friendly-fire penalties. "
    "Compatible with OpenEnv ≥ 0.2.2."
)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _SERVER_START_TIME
    _SERVER_START_TIME = time.time()

    print("=" * 60)
    print("FireSwarm MARL Environment — server ready")
    print(f"  Endpoints              : /reset /step /state /ws /health")
    print(f"  Max concurrent sessions: {_concurrency.max_concurrent_envs}")
    print(f"  Session timeout        : {_concurrency.session_timeout}s")
    print(f"  PID                    : {os.getpid()}")
    print("=" * 60)
    yield
    print(f"FireSwarm server shutting down (uptime {time.time() - _SERVER_START_TIME:.1f}s)")


app.router.lifespan_context = _lifespan

# The OpenEnv app factory registers minimal stubs for /health, /metadata, and
# /schema. Remove them here so our richer custom implementations take precedence.
# FastAPI uses first-match routing, so the framework stubs must be evicted before
# our routes are registered below.
app.routes[:] = [
    r for r in app.routes
    if getattr(r, "path", None) not in ("/health", "/metadata", "/schema")
]


@app.get("/health", summary="Readiness probe", tags=["Operations"])
async def health() -> JSONResponse:
    """
    Returns HTTP 200 with structured readiness detail.

    Orchestrators should poll this endpoint before issuing /reset.
    The `status: healthy` field is required by the OpenEnv readiness contract.
    """
    payload: Dict[str, Any] = {
        "status":         "healthy",
        "uptime_seconds": round(time.time() - _SERVER_START_TIME, 2),
        "environment": {
            "name":                    "fire_swarm_simulator",
            "version":                 app.version,
            "dds_model":               "Gilbert-Elliott two-state Markov + gossip routing",
            "max_concurrent_sessions": _concurrency.max_concurrent_envs,
            "session_timeout_s":       _concurrency.session_timeout,
        },
        "websocket_endpoint": "/ws",
        "pid":                os.getpid(),
    }
    return JSONResponse(content=payload, status_code=200)


_TASK_METADATA: Dict[str, dict] = {
    "easy": {
        "id":          "easy",
        "description": (
            "Single-drone, 15×15 grid, 1 fire seed placed near drone spawn (row 3), "
            "slow fire spread (base_ignite=0.05, wind_mult=0.8, t_burn=10). "
            "Drone reaches fire in 2 steps and must suppress it and any secondary spread "
            "within 40 steps."
        ),
        "difficulty":  "easy",
        "max_steps":   40,
        "grid_size":   15,
        "num_drones":  1,
        "fire_seeds":  1,
        "score_range": [0.0, 1.0],
        "grader":      "programmatic",
    },
    "medium": {
        "id":          "medium",
        "description": (
            "Three-drone swarm, 20×20 grid, 5 fire seeds, moderate burn (t_burn=6), "
            "1.5× wind. Agents must coordinate to contain spreading fire within 50 steps."
        ),
        "difficulty":  "medium",
        "max_steps":   50,
        "grid_size":   20,
        "num_drones":  3,
        "fire_seeds":  5,
        "score_range": [0.0, 1.0],
        "grader":      "programmatic",
    },
    "hard": {
        "id":          "hard",
        "description": (
            "Five-drone swarm, 25×25 grid, 8 fire seeds spread across quadrants, "
            "fast burn (t_burn=5), double wind (wind_mult=2). "
            "Coordinate refuelling under pressure within 70 steps."
        ),
        "difficulty":  "hard",
        "max_steps":   70,
        "grid_size":   25,
        "num_drones":  5,
        "fire_seeds":  8,
        "score_range": [0.0, 1.0],
        "grader":      "programmatic",
    },
}


@app.get("/metadata", summary="Environment metadata", tags=["Operations"])
async def get_metadata() -> JSONResponse:
    """OpenEnv validator requires GET /metadata returning name and description."""
    return JSONResponse(content={
        "name":              "FireSwarm — Decentralised Firefighting UAV Swarm Environment",
        "description": (
            "A multi-agent reinforcement learning environment simulating autonomous UAV swarms "
            "performing wildfire and infrastructure fire suppression in communication-degraded, "
            "high-wind Gulf/UAE scenarios. Features Cellular Automata fire spread with "
            "Ornstein-Uhlenbeck wind evolution, Gilbert-Elliott two-state Markov DDS packet-loss "
            "model, kinematic speed constraint (MAX_SPEED=2 cells/step), finite retardant payload "
            "with corner refuel stations, and deterministic programmatic graders scored against a "
            "NOP baseline. Three difficulty tiers: easy (1 drone, 15×15), medium (3 drones, 20×20), "
            "hard (5 drones, 25×25, shamal wind). Directly models real-world scenarios including "
            "the Abqaiq petrochemical facility strike response."
        ),
        "version":           app.version,
        "author":            "Likhith M (Likhith-BlueLotus)",
        "author_email":      "likhithm2426@gmail.com",
        "github":            "https://github.com/Likhith-BlueLotus/fire-swarm-simulator",
        "space":             "https://huggingface.co/spaces/Le0AtiS/fire-swarm-simulator",
        "license":           "BSD-3-Clause",
        "tasks":             ["easy", "medium", "hard"],
        "num_agents":        {"easy": 1, "medium": 3, "hard": 5},
        "reward_range":      [0.0, 1.0],
        "tags":              ["marl", "wildfire", "uav", "openenv", "reinforcement-learning", "simulation"],
        "documentation_url": "https://github.com/Likhith-BlueLotus/fire-swarm-simulator#readme",
    }, status_code=200)


@app.get("/schema", summary="Action / observation / state schemas", tags=["Operations"])
async def get_schema() -> JSONResponse:
    """OpenEnv validator requires GET /schema returning action, observation, state as dicts."""
    from models import SwarmAction, SwarmObservation, SwarmState
    return JSONResponse(content={
        "action":      SwarmAction.model_json_schema(),
        "observation": SwarmObservation.model_json_schema(),
        "state":       SwarmState.model_json_schema(),
    }, status_code=200)


@app.get("/tasks", summary="List all graded tasks", tags=["Tasks"])
async def list_tasks() -> JSONResponse:
    """Enumerate all three graded tasks with difficulty metadata."""
    return JSONResponse(content={"tasks": list(_TASK_METADATA.values())}, status_code=200)


class GradeRequest(BaseModel):
    """
    Request body for POST /grade/{task}.

    The client should populate agent_active_fires and agent_burned_area with
    the final SwarmState values from the episode. These fields bypass the
    session-pool lookup and prevent the grader from making optimistic assumptions
    about the agent's end-state that could be exploited for a higher score.
    """
    seed:               int   = 42    # must match the seed passed to /reset so the NOP baseline fights the same fire
    session_id:         str   = ""    # optional: used to look up state from an active WebSocket session
    cumulative_reward:  float = 0.0   # sum of per-step rewards reported by the environment
    steps_taken:        int   = 0     # number of steps the agent completed (used to run the NOP baseline for the same duration)
    episode_done:       bool  = False # True if the episode reached a terminal state (all fires out, timeout, or all drones dead)
    agent_active_fires: int   = -1    # ground-truth active fire count at episode end; -1 triggers the conservative penalty fallback
    agent_burned_area:  int   = -1    # ground-truth burned cell count at episode end; -1 triggers the conservative penalty fallback


def _run_nop_baseline(task: str, seed: int, num_steps: int) -> dict:
    """
    Run a full NOP episode (drones stationary, pump=0) and return fire stats.

    Thread-safe: FireSwarmEnvironment now uses self.rng = np.random.default_rng(seed)
    internally, so each instance carries its own isolated RNG. No global
    numpy random state is touched — concurrent grading requests cannot corrupt
    each other's RNG streams.
    """
    env = FireSwarmEnvironment()
    env.reset(seed=seed, task=task)
    for _ in range(num_steps):
        nop_actions = [
            DroneNodeAction(
                agent_id=d_id,
                target_waypoint=(
                    float(d["pos"][0]),
                    float(d["pos"][1]),
                    float(d["altitude"]),
                ),
                pump_activation=0.0,
                broadcast_message=None,
                qos_profile=QoSProfile.BEST_EFFORT,
            )
            for d_id, d in env.drones.items()
        ]
        obs = env.step(SwarmAction(node_actions=nop_actions))
        if obs.done:
            break
    st = env.state
    return {
        "active_fires": st.active_fires,
        "burned_area":  st.total_burned_area,
    }


@app.post(
    "/grade/{task}",
    summary="Run programmatic grader for a task",
    description=(
        "Scores a completed agent episode relative to an uncontrolled NOP baseline. "
        "Pass session_id, cumulative_reward, steps_taken, and episode_done from the "
        "inference script after the episode ends. Scores are in [0.0, 1.0] and "
        "are deterministic for a fixed seed."
    ),
    tags=["Tasks"],
)
async def grade_task(task: str, body: GradeRequest = GradeRequest()) -> JSONResponse:
    """
    Programmatic grader endpoint.

    Scoring formula (from openenv.yaml grader_formula):
      score = 0.35 × fire_suppression_ratio
            + 0.25 × (1 − burned_cells / grid_cells)
            + 0.20 × normalised_cumulative_reward
            + 0.20 × completion_bonus (1.0 if all fires out, 0.5 if agent beat baseline, 0.0 otherwise)

    fire_suppression_ratio is computed relative to the NOP baseline so that
    an agent that merely does nothing scores ≈ 0, not ≈ 1.
    """
    if task not in _TASK_METADATA:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task {task!r}. Valid tasks: {list(_TASK_METADATA)}",
        )

    cfg        = TASK_CONFIG[task]
    grid_cells = cfg["grid_size"] ** 2
    max_steps  = _TASK_METADATA[task]["max_steps"]

    # ── Retrieve agent episode state ────────────────────────────────────────
    # Priority order:
    #   1. Client-provided ground-truth fields (agent_active_fires / agent_burned_area)
    #   2. Live session pool lookup via session_id (WebSocket sessions)
    #   3. Conservative fallback inference from episode_done
    agent_active_fires: int = max(0, body.agent_active_fires) if body.agent_active_fires >= 0 else -1
    agent_burned_area:  int = max(0, body.agent_burned_area)  if body.agent_burned_area  >= 0 else -1

    if agent_active_fires < 0 and body.session_id:
        try:
            from openenv.core.env_server import _session_pool  # type: ignore[attr-defined]
            env_obj = _session_pool.get(body.session_id)
            if env_obj is not None:
                st = env_obj.state
                agent_active_fires = st.active_fires
                agent_burned_area  = st.total_burned_area
        except Exception:
            pass

    # Strict fallback — only used when neither client fields nor session available.
    # We NEVER trust episode_done=True to imply fires_out=True because episode_done
    # also fires on timeout and all_dead. A malicious caller can set episode_done=True
    # while omitting agent_active_fires (defaulting to -1) and receive a free
    # completion bonus. Always assume worst-case when state is missing.
    if agent_active_fires < 0:
        agent_active_fires = cfg["fire_seeds"]
        agent_burned_area  = grid_cells // 10

    # ── NOP baseline (same seed, same step count) ────────────────────────────
    # Run NOP for the same number of steps the agent took — apples-to-apples.
    steps_for_baseline = max(1, body.steps_taken) if body.steps_taken > 0 else max_steps
    baseline = _run_nop_baseline(task, body.seed, steps_for_baseline)

    baseline_active   = max(1, baseline["active_fires"])
    baseline_burned   = baseline["burned_area"]

    # ── Score components ─────────────────────────────────────────────────────
    # fire_suppression: fractional improvement over uncontrolled NOP baseline.
    fire_suppression = float(max(0.0, min(1.0,
        1.0 - agent_active_fires / baseline_active
    )))

    # scar_ratio: fraction of grid cells that are still unburned.
    scar_ratio = float(max(0.0, 1.0 - agent_burned_area / max(1, grid_cells)))

    # reward_norm: average per-step reward (each step already in [0,1]).
    reward_norm = float(min(1.0, max(0.0,
        body.cumulative_reward / max(1, body.steps_taken)
    ))) if body.steps_taken > 0 else 0.0

    # completion_bonus: only 1.0 when all fires confirmed extinguished.
    # 0.5 when agent clearly beat baseline even without full clearance.
    # 0.0 on timeout without beating baseline.
    if body.episode_done and agent_active_fires == 0:
        completion_bonus = 1.0
    elif fire_suppression > 0.5:
        completion_bonus = 0.5
    else:
        completion_bonus = 0.0

    score = float(min(0.999, max(0.001,
        0.35 * fire_suppression
        + 0.25 * scar_ratio
        + 0.20 * reward_norm
        + 0.20 * completion_bonus
    )))

    return JSONResponse(content={
        "task":                   task,
        "seed":                   body.seed,
        "steps_taken":            body.steps_taken,
        "agent_active_fires":     agent_active_fires,
        "agent_burned_area":      agent_burned_area,
        "baseline_active_fires":  baseline["active_fires"],
        "baseline_burned_area":   baseline_burned,
        "grid_cells":             grid_cells,
        "fire_suppression_ratio": round(fire_suppression, 4),
        "scar_ratio":             round(scar_ratio, 4),
        "reward_norm":            round(reward_norm, 4),
        "completion_bonus":       round(completion_bonus, 4),
        "score":                  round(score, 4),
        "score_range":            [0.0, 1.0],
        "grader":                 "programmatic",
        "deterministic":          True,
    }, status_code=200)


def main() -> None:
    """Entry point for `uv run server` and the pyproject.toml scripts table."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        workers=int(os.environ.get("WORKERS", "1")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
