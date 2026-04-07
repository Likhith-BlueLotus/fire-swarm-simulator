#!/usr/bin/env python3
"""
Docker HEALTHCHECK probe for the FireSwarm MARL environment.

Performs two sequential checks against the running server:
  1. GET /health   — verifies the FastAPI process is alive and reports status=healthy.
  2. POST /reset   — verifies the environment initialises without error,
                     confirming the physics engine, Pydantic models, and
                     session pool are all ready to accept episode requests.

Exit codes follow the Docker HEALTHCHECK convention:
  0 — healthy (both checks passed)
  1 — unhealthy (at least one check failed; error written to stderr)

HEALTHCHECK parameters (set in Dockerfile):
  --interval=30s      probe runs every 30 seconds
  --timeout=10s       each urllib call has a 5-second hard timeout
  --start-period=20s  container gets 20 seconds to warm up before first probe
  --retries=3         three consecutive failures mark the container unhealthy
"""
import json
import sys
import urllib.request

BASE = "http://localhost:7860"

try:
    # Verify the server process is alive and the lifespan handler has completed.
    # The /health response must contain {"status": "healthy"} — any other value or
    # a connection error means the uvicorn process has not started cleanly.
    with urllib.request.urlopen(BASE + "/health", timeout=5) as resp:
        data = json.load(resp)
    assert data.get("status") == "healthy", f"/health returned unexpected payload: {data}"

    # Verify the environment can initialise a full episode.
    # Using task=easy (1 drone, 15×15 grid) to keep the probe lightweight;
    # it exercises the complete reset path: RNG seeding, CA grid setup,
    # drone placement, and observation serialisation.
    req = urllib.request.Request(
        BASE + "/reset",
        data=json.dumps({"task": "easy"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        obs = json.load(resp)

    # Accept both the flat observation dict and the {observation: {...}} envelope
    # that some OpenEnv server versions wrap the response in.
    assert (
        "local_grid_thermal" in obs or "observation" in obs
    ), f"/reset returned unexpected payload shape: {list(obs.keys())}"

    sys.exit(0)

except Exception as exc:
    print(f"HEALTHCHECK FAILED: {exc}", file=sys.stderr)
    sys.exit(1)
