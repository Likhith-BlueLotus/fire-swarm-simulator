"""
Inference Script — FireSwarm MARL Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

Runs one episode per task (easy / medium / hard) against a live FireSwarm
server and reports per-task scores in [0.0, 1.0].

Optional:
  OPENENV_ENDPOINT   FireSwarm server URL (default: http://localhost:7860)

Usage:
  python inference.py

Interaction model (per course modules 2–4):
  Uses the FireSwarmEnv WebSocket client (client.py) which maintains
  episode state across steps via a persistent WebSocket connection —
  the same pattern shown in all OpenEnv course notebooks.
"""

import json
import logging
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from client import FireSwarmEnv
from models import DroneNodeAction, QoSProfile, SwarmAction, SwarmObservation


# ---------------------------------------------------------------------------
# Credentials — exactly as required by the hackathon spec
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME")
OPENENV_URL  = os.getenv("OPENENV_ENDPOINT") or "http://localhost:7860"

# ---------------------------------------------------------------------------
# Episode limits
# ---------------------------------------------------------------------------
MAX_STEPS_PER_TASK = {"easy": 30, "medium": 50, "hard": 70}
WALL_CLOCK_BUDGET  = 18 * 60   # 18 min — 2-min margin under the 20-min cap
TEMPERATURE        = 0.2
MAX_TOKENS         = 1024

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("fire_swarm.inference")


# ---------------------------------------------------------------------------
# Health check (plain HTTP, no state needed)
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post(url: str, body: dict, timeout: int = 30) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


EPISODE_SEED = 42   # fixed seed so agent and NOP baseline fight the same fire


def grade_episode(
    task:              str,
    cumulative_reward: float,
    steps_taken:       int,
    episode_done:      bool,
    active_fires:      int,
    burned_area:       int,
    seed:              int = EPISODE_SEED,
) -> float:
    """
    POST /grade/{task} — scores the completed episode vs. an uncontrolled NOP baseline.

    Sends ground-truth active_fires and burned_area so the grader does not
    have to guess the agent's end-state from session_id alone.

    Returns a score in [0.0, 1.0]. Falls back to 0.0 if endpoint is unavailable.
    """
    try:
        resp = _post(
            f"{OPENENV_URL}/grade/{task}",
            {
                "seed":               seed,
                "session_id":         "",
                "cumulative_reward":  cumulative_reward,
                "steps_taken":        steps_taken,
                "episode_done":       episode_done,
                "agent_active_fires": active_fires,
                "agent_burned_area":  burned_area,
            },
        )
        return float(resp.get("score", 0.0))
    except Exception as exc:
        log.warning("Grade endpoint error (%s) — score=0.0", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_GRID_LEGEND = "0=EMPTY 1=FUEL 2=FIRE 3=EXTINGUISHED 4=SCAR"

_SYSTEM_PROMPT = f"""\
You are the central coordinator for a firefighting UAV swarm.

Grid cells: {_GRID_LEGEND}
Wind vector: (magnitude_m_s, bearing_degrees). Fire spreads downwind and upslope.
Payload: each drone carries ≤10 kg. Extinguishing one centre cell costs 1 kg.
Refill corners (altitude MUST be < 5 m to dock):
  easy  (15×15):  (0,0), (0,14), (14,0), (14,14)
  medium (20×20): (0,0), (0,19), (19,0), (19,19)
  hard  (25×25):  (0,0), (0,24), (24,0), (24,24)

MOVEMENT: each drone moves AT MOST 2 cells per axis per step (server-enforced clamp).
  A drone at (0,8) sent to (10,8) only reaches (2,8). Plan multi-step routes.

STRICT RULES:
1. ALWAYS use the SYSTEM-CALCULATED WAYPOINTS provided in the observation.
   Copy next_waypoint and pump values exactly — do NOT invent different coordinates.
2. When pump hint says "PUMP NOW": set pump_activation=1.0 and use the given waypoint.
   When hint says "TRANSIT": set pump_activation=0.0 and move toward the given waypoint.
   The Gaussian footprint covers a 3×3 area — pumping from 1 cell away still suppresses fire.
3. NEVER send two drones to the same (x,y,pump>0) — friendly fire zeroes the entire step reward.
4. When the SYSTEM-CALCULATED WAYPOINT shows "LOW PAYLOAD": obey it — fly to the
   listed dock corner at altitude=0.0 with pump=0.0. Do NOT override this with a fire waypoint.
   Other drones with sufficient payload continue suppressing independently.
5. Keep altitude=10.0 during transit; use altitude=0.0 ONLY at corners to refill.
6. Use RELIABLE QoS only when battery < 0.3.

Respond with ONLY valid JSON. No prose, no markdown fences."""

_ACTION_SCHEMA = {
    "type": "object",
    "required": ["node_actions"],
    "properties": {
        "node_actions": {
            "type":  "array",
            "items": {
                "type":     "object",
                "required": ["agent_id", "target_waypoint", "pump_activation", "qos_profile"],
                "properties": {
                    "agent_id": {"type": "string"},
                    "target_waypoint": {
                        "type":        "array",
                        "items":       {"type": "number"},
                        "minItems":    3,
                        "maxItems":    3,
                        "description": "[x, y, altitude_m]; x,y in [0, grid_size-1]",
                    },
                    "pump_activation":   {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "broadcast_message": {"type": ["string", "null"], "maxLength": 256},
                    "qos_profile":       {"type": "string", "enum": ["BEST_EFFORT", "RELIABLE"]},
                },
            },
        }
    },
}


def _format_obs(obs: SwarmObservation, task: str) -> str:
    """Render a SwarmObservation into a compact string for the LLM."""
    telem     = obs.drone_telemetry
    wind      = obs.wind_vector
    dds       = obs.dds_global_space
    grid_size = 15 if task == "easy" else (20 if task == "medium" else 25)
    R         = 7

    peer_telem  = dds.peer_telemetry
    active_ids  = dds.active_peers

    # Build GE-independent position map from neighbor_states (populated for ALL
    # live drones regardless of DDS channel state).  peer_telem only carries
    # drones whose GE channel is Good; absent entries would give pos=[0,0] and
    # corrupt every fire-coordinate and navigation-hint calculation.
    ns_pos: Dict[str, List[int]] = {
        ns["id"]: [int(ns["pos"][0]), int(ns["pos"][1])]
        for ns in obs.neighbor_states
    }

    drone_lines: List[str] = []
    for pid in active_ids:
        pt  = peer_telem.get(pid, {})
        # Use GE-independent ns_pos for position; peer_telem for battery/payload
        # (only available on Good channel — fall back to swarm average).
        pos = ns_pos.get(pid) or pt.get("pos", [0, 0])
        bat = pt.get("battery", telem.get("avg_battery", 1.0))
        pay = pt.get("payload",  telem.get("avg_payload", 10.0))
        ch  = "Good" if pid in peer_telem else "Bad(GE)"
        drone_lines.append(
            f"  {pid}: pos=({int(pos[0])},{int(pos[1])}) "
            f"battery={float(bat):.2f} payload={float(pay):.1f}kg channel={ch}"
        )
    drones_str = "\n".join(drone_lines) or "  (no telemetry)"

    fire_abs: Dict[str, List[List[int]]] = {}
    for d_id, grid in obs.per_drone_grids.items():
        # Use neighbor_states position (GE-independent) first; fall back to
        # peer_telem if somehow absent from neighbor_states.
        pos    = ns_pos.get(d_id) or peer_telem.get(d_id, {}).get("pos", [0, 0])
        cx, cy = int(pos[0]), int(pos[1])
        fires: List[List[int]] = []
        for row_idx, row in enumerate(grid):
            for col_idx, val in enumerate(row):
                if int(val) == 2:
                    ax = max(0, min(grid_size - 1, cx + (row_idx - R)))
                    ay = max(0, min(grid_size - 1, cy + (col_idx - R)))
                    fires.append([ax, ay])
        if fires:
            fire_abs[d_id] = fires[:4]

    fire_str = "; ".join(f"{d}→{coords}" for d, coords in fire_abs.items())
    if not fire_str:
        fire_str = "none visible in FOV — move toward grid centre"

    if obs.per_drone_grids:
        first_id, first_grid = next(iter(obs.per_drone_grids.items()))
        grids_str = f"{first_id} FOV:\n" + "\n".join(
            "".join(str(int(c)) for c in row) for row in first_grid
        )
    else:
        grids_str = "D0 FOV:\n" + "\n".join(
            "".join(str(int(c)) for c in row)
            for row in obs.local_grid_thermal
        )

    # Pre-compute optimal next waypoint per drone so the LLM never needs to
    # do clip/clamp arithmetic itself — it just uses the provided values.
    # Each hint gives the MAX_SPEED=2 step toward the nearest visible fire.
    # IMPORTANT: iterate active_ids (not peer_telem) — on a GE Bad channel
    # a drone is absent from peer_telem but still needs an action.
    #
    # Fire zone anchor: fires are placed at grid_size//3*2 (lower 2/3 of grid).
    # When no fires are in FOV, the patrol target must be this anchor row (not
    # grid_size//2 = centre), otherwise drones stop 3+ rows short and hover.
    fire_zone_row = grid_size // 3 * 2   # 10 for easy, 13 for medium, 16 for hard
    grid_col_mid  = grid_size // 2

    # Refill threshold: below this payload a drone cannot make a meaningful Gaussian
    # drop (centre costs 1 kg; cardinal 0.5 kg; diagonal 0.3 kg).
    # 2.0 kg = enough for exactly 2 centre-cell hits — once below this, refuel now.
    # Using ns_pos (GE-independent) throughout so Bad-channel drones get correct hints.
    REFILL_THRESHOLD    = 2.0
    FULL_TANK_THRESHOLD = 9.5   # MAX_PAYLOAD(10.0) − 0.5; must match server environment.py
    corners = [
        (0,             0),
        (0,             grid_size - 1),
        (grid_size - 1, 0),
        (grid_size - 1, grid_size - 1),
    ]
    corners_set = set(corners)  # O(1) membership test inside the per-drone loop

    navigation_hints: List[str] = []
    for d_id in active_ids:
        pt     = peer_telem.get(d_id, {})
        pos    = ns_pos.get(d_id) or pt.get("pos", [0, 0])
        cx, cy = int(pos[0]), int(pos[1])

        # Per-drone payload from peer_telem (Good channel) or fall back to avg.
        # This is the individual payload, not the swarm average.
        current_payload = float(pt.get("payload", telem.get("avg_payload", 10.0)))

        # ── PRIORITY 1: Refill if payload critically low OR already docking ────
        # "Sticky" refuelling: once a drone commits to a corner it must stay
        # until the tank is full (≥ 9.5 kg), not just above REFILL_THRESHOLD.
        #
        # Without this, the drone yo-yos:
        #   tick N  : payload 1.6 → routes to corner
        #   tick N+3: payload 4.6 (one REFILL_RATE=3.0 tick) → 4.6 > 2.0 → flies away
        #   tick N+7: payload ~0 → routes to corner again
        # That wastes ~14 steps per refuel cycle on pure transit.
        #
        # With stickiness (9.5 kg threshold):
        #   tick N  : arrives 1.6 → 4.6 kg
        #   tick N+1: stays   4.6 → 7.6 kg
        #   tick N+2: stays   7.6 → 10.0 kg  (capped at MAX_PAYLOAD)
        #   tick N+3: 10.0 ≥ 9.5 → deploy with full tank
        # Three docking ticks, then the drone fights fires for much longer
        # before needing another refuel trip.
        #
        # 9.5 kg threshold: one tick below MAX_PAYLOAD=10.0 so a drone that
        # arrives with exactly 10.0 kg does NOT get pinned at the corner.
        at_corner = (cx, cy) in corners_set

        if current_payload <= REFILL_THRESHOLD or (at_corner and current_payload < FULL_TANK_THRESHOLD):
            rx, ry = min(corners, key=lambda c: abs(cx - c[0]) + abs(cy - c[1]))
            nx = int(max(cx - 2, min(cx + 2, rx)))
            ny = int(max(cy - 2, min(cy + 2, ry)))
            status_msg = "LOW PAYLOAD" if current_payload <= REFILL_THRESHOLD else "REFILLING (stay docked)"
            navigation_hints.append(
                f"  {d_id}: {status_msg} ({current_payload:.1f}kg)"
                f" → dock at ({rx},{ry})"
                f" | next_waypoint=[{nx},{ny},0.0] pump=0.0"
            )
            continue  # skip fire-targeting until tank is full

        # ── PRIORITY 2: Target nearest fire ──────────────────────────────────
        # Prefer fires visible in this drone's own FOV; fall back to any fire
        # seen by any drone in the swarm so the hint is never empty.
        fires_for_drone = fire_abs.get(d_id, [])
        if not fires_for_drone:
            all_fires = [f for fl in fire_abs.values() for f in fl]
            fires_for_drone = all_fires

        if fires_for_drone:
            fires_for_drone.sort(key=lambda f: abs(f[0] - cx) + abs(f[1] - cy))
            fx, fy = fires_for_drone[0]
            nx = int(max(cx - 2, min(cx + 2, fx)))
            ny = int(max(cy - 2, min(cy + 2, fy)))

            # Pump when drone is ON the fire cell OR directly adjacent (Chebyshev ≤1).
            # The Gaussian footprint covers a 3×3 area so a drone 1 cell away still
            # suppresses the centre cell — holding back pump for exact overlap wastes steps.
            cheb_dist = max(abs(fx - cx), abs(fy - cy))
            if cheb_dist <= 1:
                pump_hint = "1.0 (ADJACENT TO FIRE — PUMP NOW)"
                nx, ny = cx, cy  # stay in place and pump
            else:
                pump_hint = "0.0 (TRANSIT)"

            navigation_hints.append(
                f"  {d_id}: pos=({cx},{cy}) → fire=({fx},{fy}) dist={cheb_dist}"
                f" | next_waypoint=[{nx},{ny},10.0] pump={pump_hint}"
            )
        else:
            # ── PRIORITY 3: No fires visible — advance to fire zone ──────────
            # Patrol to fire_zone_row (2/3 down the grid), not grid_centre (halfway).
            # Fires are placed at grid_size//3*2; grid_centre stops drones 3 rows short.
            nx = int(max(cx - 2, min(cx + 2, fire_zone_row)))
            ny = int(max(cy - 2, min(cy + 2, grid_col_mid)))
            navigation_hints.append(
                f"  {d_id}: no fire in FOV — advance to fire zone ({nx},{ny},10.0) pump=0.0"
            )

    hints_str = "\n".join(navigation_hints)

    return (
        f"=== OBSERVATION ===\n"
        f"grid_size={grid_size}  wind=({wind[0]:.1f}m/s, {wind[1]:.0f}°)\n"
        f"active_fires={int(telem.get('active_fires', 0))}  "
        f"avg_battery={float(telem.get('avg_battery', 1)):.2f}  "
        f"avg_payload={float(telem.get('avg_payload', 10)):.1f}kg\n"
        f"\nDRONE POSITIONS:\n{drones_str}\n"
        f"\nFIRE CELLS (absolute x,y per drone FOV):\n  {fire_str}\n"
        f"\nSYSTEM-CALCULATED WAYPOINTS (follow these exactly):\n{hints_str}\n"
        f"\nACTION REMINDERS:\n"
        f"  • Copy waypoints above exactly — do NOT invent different coordinates\n"
        f"  • pump=1.0 whenever hint says PUMP NOW (drone is on or adjacent to fire)\n"
        f"  • Assign DIFFERENT fire cells to each drone — no two drones pump same (x,y)\n"
        f"  • altitude=0.0 at corners (0,0),(0,{grid_size-1}),({grid_size-1},0),({grid_size-1},{grid_size-1}) to refill\n"
        f"\n{grids_str}"
    )


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

_VALID_QOS = {"BEST_EFFORT", "RELIABLE"}


def _nop_action(obs: SwarmObservation) -> SwarmAction:
    """Stay-in-place zero-pump fallback for LLM parse failures.

    Uses neighbor_states (GE-independent) for position so drones on a Bad
    channel don't get sent to (0,0) — which would look like a legitimate move
    command and could trigger a loiter penalty or collision on the next step.
    """
    ns_pos: Dict[str, List[int]] = {
        ns["id"]: [int(ns["pos"][0]), int(ns["pos"][1])]
        for ns in obs.neighbor_states
    }
    node_actions = []
    for pid in obs.dds_global_space.active_peers:
        pt  = obs.dds_global_space.peer_telemetry.get(pid, {})
        # Prefer GE-independent ns_pos; fall back to peer_telem only if absent
        # from neighbor_states (should never happen for a live drone).
        pos = ns_pos.get(pid) or pt.get("pos", [0, 0])
        node_actions.append(DroneNodeAction(
            agent_id          = pid,
            target_waypoint   = (float(pos[0]), float(pos[1]), 10.0),
            pump_activation   = 0.0,
            broadcast_message = None,
            qos_profile       = QoSProfile.BEST_EFFORT,
        ))
    if not node_actions:
        node_actions = [DroneNodeAction(
            agent_id="D0", target_waypoint=(0.0, 0.0, 10.0),
            pump_activation=0.0, broadcast_message=None,
            qos_profile=QoSProfile.BEST_EFFORT,
        )]
    return SwarmAction(node_actions=node_actions)


def _parse_and_clamp(raw: dict, obs: SwarmObservation, task: str) -> SwarmAction:
    """
    Parse an LLM JSON dict into a validated SwarmAction.

    Deduplicates pump targets to prevent friendly-fire zeroing the reward.
    """
    grid_size = 15 if task == "easy" else (20 if task == "medium" else 25)
    gmax      = float(grid_size - 1)
    safe_nas  = []

    pumping_targets: set = set()

    for na in raw.get("node_actions", []):
        wp = list(na.get("target_waypoint", [0, 0, 10]))
        while len(wp) < 3:
            wp.append(10.0)
        wp[0] = float(max(0.0, min(gmax,  wp[0])))
        wp[1] = float(max(0.0, min(gmax,  wp[1])))
        wp[2] = float(max(0.0, min(200.0, wp[2])))

        pump = float(max(0.0, min(1.0, na.get("pump_activation", 0.0))))

        # Friendly-fire deduplication
        key = (int(wp[0]), int(wp[1]))
        if pump > 0.5:
            if key in pumping_targets:
                pump = 0.0
            else:
                pumping_targets.add(key)

        qos_str = str(na.get("qos_profile", "BEST_EFFORT")).upper()
        qos     = QoSProfile.RELIABLE if qos_str == "RELIABLE" else QoSProfile.BEST_EFFORT

        msg = na.get("broadcast_message")
        if msg is not None:
            msg = str(msg)[:256]

        agent_id = str(na.get("agent_id", "D0")) or "D0"

        safe_nas.append(DroneNodeAction(
            agent_id          = agent_id,
            target_waypoint   = (wp[0], wp[1], wp[2]),
            pump_activation   = pump,
            broadcast_message = msg,
            qos_profile       = qos,
        ))

    return SwarmAction(node_actions=safe_nas) if safe_nas else _nop_action(obs)


def get_llm_action(
    client: OpenAI,
    obs:    SwarmObservation,
    task:   str,
    step:   int,
) -> SwarmAction:
    """
    Issue one LLM call and return a validated SwarmAction.

    Falls back to NOP on any parse or API error.
    """
    obs_text = _format_obs(obs, task)
    user_msg = (
        f"Task: {task} | Step: {step}\n\n"
        f"Observation:\n{obs_text}\n\n"
        f"Schema to conform to:\n{json.dumps(_ACTION_SCHEMA, indent=2)}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        raw    = resp.choices[0].message.content or "{}"
        action = json.loads(raw)
        return _parse_and_clamp(action, obs, task)
    except Exception as exc:
        log.warning("LLM error at step %d: %s — NOP fallback", step, exc)
        return _nop_action(obs)


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task: str, deadline: float) -> Dict[str, Any]:
    """
    Run one complete episode using the FireSwarmEnv WebSocket client.

    Uses `with FireSwarmEnv(base_url=...).sync() as env:` — the canonical
    OpenEnv pattern from modules 2–4 — which maintains episode state across
    steps via a persistent WebSocket connection.
    """
    max_steps = MAX_STEPS_PER_TASK[task]
    log.info("=" * 60)
    log.info("TASK: %s  (max_steps=%d)", task.upper(), max_steps)
    log.info("=" * 60)

    t0                = time.time()
    cumulative_reward = 0.0
    last_reward       = 0.0
    step              = 0
    done              = False

    with FireSwarmEnv(base_url=OPENENV_URL).sync() as env:
        # Lock seed so the NOP baseline in the grader fights the identical fire.
        result = env.reset(task=task, seed=EPISODE_SEED)
        obs    = result.observation

        log.info(
            "Reset OK — active_fires=%.0f  drones=%d",
            obs.drone_telemetry.get("active_fires", -1),
            len(obs.dds_global_space.active_peers),
        )

        while not done and step < max_steps:
            if time.time() > deadline:
                log.warning("Wall-clock budget exceeded — ending task %s early", task)
                break

            action         = get_llm_action(client, obs, task, step + 1)
            result         = env.step(action)
            obs            = result.observation
            reward         = float(result.reward or 0.0)
            done           = bool(result.done)

            cumulative_reward += reward
            last_reward        = reward
            step              += 1

            telem = obs.drone_telemetry
            log.info(
                "Step %3d | fires=%.0f | avg_battery=%.3f | avg_payload=%.1f"
                " | alive=%.0f | reward=%.4f | done=%s",
                step,
                telem.get("active_fires",  -1),
                telem.get("avg_battery",   -1),
                telem.get("avg_payload",   -1),
                telem.get("alive_drones",  -1),
                reward,
                done,
            )

    active_fires = int(obs.drone_telemetry.get("active_fires", 0))
    burned_area  = int(obs.drone_telemetry.get("burned_area",  0))

    # ── Programmatic grader ───────────────────────────────────────────────
    score = grade_episode(
        task              = task,
        cumulative_reward = cumulative_reward,
        steps_taken       = step,
        episode_done      = done,
        active_fires      = active_fires,
        burned_area       = burned_area,
    )

    elapsed = time.time() - t0
    log.info(
        "TASK %s COMPLETE — steps=%d  score=%.4f  "
        "(fires_left=%d  cum_reward=%.3f  done=%s)  elapsed=%.1fs",
        task.upper(), step, score, active_fires, cumulative_reward, done, elapsed,
    )

    return {
        "task":         task,
        "steps":        step,
        "total_reward": round(cumulative_reward, 4),
        "final_reward": round(last_reward, 4),
        "active_fires": active_fires,
        "done":         done,
        "elapsed_s":    round(elapsed, 1),
        "score":        score,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    missing = [
        name for name, value in [
            ("API_BASE_URL", API_BASE_URL),
            ("MODEL_NAME",   MODEL_NAME),
            ("HF_TOKEN",     API_KEY),
        ]
        if not value
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Set API_BASE_URL, MODEL_NAME, and HF_TOKEN before running."
        )

    try:
        health = _get(f"{OPENENV_URL}/health", timeout=10)
        log.info("Server health: %s", health.get("status", "unknown"))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach FireSwarm server at {OPENENV_URL}: {exc}\n"
            "Start the server: uvicorn server.app:app --port 7860"
        ) from exc

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    deadline   = time.time() + WALL_CLOCK_BUDGET

    log.info("Model : %s", MODEL_NAME)
    log.info("Server: %s", OPENENV_URL)
    log.info("Budget: %d min", WALL_CLOCK_BUDGET // 60)

    results = []
    for task in ("easy", "medium", "hard"):
        if time.time() > deadline:
            log.warning("Budget exhausted — skipping task %s", task)
            results.append({"task": task, "score": 0.0, "skipped": True})
            continue
        results.append(run_task(llm_client, task, deadline))

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for r in results:
        if r.get("skipped"):
            print(f"  {r['task']:8s}  SKIPPED (budget exceeded)   score=0.0000")
        else:
            print(
                f"  {r['task']:8s}  steps={r['steps']:3d}  "
                f"fires_left={r['active_fires']:3d}  "
                f"score={r['score']:.4f}"
            )

    overall = sum(r["score"] for r in results) / len(results)
    print(f"\n  OVERALL (mean): {overall:.4f}")
    print("=" * 60)
    print("\nJSON_SCORES:", json.dumps({r["task"]: r["score"] for r in results}))


if __name__ == "__main__":
    main()
