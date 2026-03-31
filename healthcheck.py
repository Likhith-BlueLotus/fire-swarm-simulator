#!/usr/bin/env python3
"""
healthcheck.py — M12: Full /reset + /health readiness smoke test.
Used by the Dockerfile HEALTHCHECK CMD.
"""
import sys
import json
import urllib.request

BASE = "http://localhost:7860"

try:
    # Step 1: /health check
    with urllib.request.urlopen(BASE + "/health", timeout=5) as resp:
        data = json.load(resp)
    assert data.get("status") == "ok", f"/health returned: {data}"

    # Step 2: /reset smoke test — proves environment is fully initialised
    req = urllib.request.Request(
        BASE + "/reset",
        data=json.dumps({}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        obs = json.load(resp)
    # Accept both flat observation and nested {observation: ...} envelope
    assert (
        "local_grid_thermal" in obs or "observation" in obs
    ), f"/reset returned unexpected payload: {list(obs.keys())}"

    sys.exit(0)

except Exception as exc:
    print(f"HEALTHCHECK FAILED: {exc}", file=sys.stderr)
    sys.exit(1)
