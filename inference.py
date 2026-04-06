"""
FireSwarm MARL Environment — Inference Script
=============================================

Runs one full episode per task (easy / medium / hard) against a live
FireSwarm server and reports per-task programmatic scores in [0.0, 1.0].

Required environment variables (hackathon spec):
  API_BASE_URL   OpenAI-compatible LLM endpoint (default: https://api.openai.com/v1)
  MODEL_NAME     Model identifier string        (default: gpt-4o-mini)
  HF_TOKEN       Bearer token / API key         (mandatory — no default)

Optional:
  OPENENV_ENDPOINT   FireSwarm server base URL (https://le0atis-fire-swarm-simulator.hf.space, default: http://localhost:7860)

Usage:
  python inference.py

Stdout format (emitted to stdout for the hackathon validator):
  [START] task=<name> env=fire-swarm-simulator model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Interaction model:
  Uses FireSwarmEnv (client.py), which wraps the OpenEnv WebSocket client.
  A single persistent WebSocket connection is held for the full episode so
  that server-side session state (drone positions, fire grid, battery levels)
  is maintained across every step.
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
# Credentials — spec requires defaults for API_BASE_URL and MODEL_NAME;
# HF_TOKEN is mandatory with no default (validated in main()).
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENENV_URL  = os.getenv("OPENENV_ENDPOINT", "http://localhost:7860")

# ---------------------------------------------------------------------------
# Episode limits
# ---------------------------------------------------------------------------
MAX_STEPS_PER_TASK = {"easy": 30, "medium": 50, "hard": 70}
WALL_CLOCK_BUDGET  = 18 * 60   # 18 min — 2-min margin under the 20-min cap
TEMPERATURE        = 0.2
MAX_TOKENS         = 1024

# ---------------------------------------------------------------------------
# Logging — writes to stderr so it does not pollute the [START]/[STEP]/[END]
# stdout lines that the hackathon validator parses.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=__import__("sys").stderr,
)
log = logging.getLogger("fire_swarm.inference")


# ---------------------------------------------------------------------------
# Lightweight HTTP helpers (stateless calls — health probe and grader only)
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


EPISODE_SEED = 42   # fixed seed so the agent and NOP baseline fight the same fire


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

    Sends ground-truth active_fires and burned_area so the grader does not have
    to guess the agent's end-state from session_id alone.

    Returns a score in [0.0, 1.0]. Falls back to 0.0 if the endpoint is unavailable.
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
4. When the SYSTEM-CALCULATED WAYPOINT shows "LOW PAYLOAD" or "REFILLING": obey it exactly.
   Copy the next_waypoint and altitude from the hint. Do NOT change altitude=10.0 to 0.0 mid-transit.
   altitude=0.0 is ONLY valid when the waypoint IS the dock corner itself (rx,ry) — never before.
5. Keep altitude=10.0 during all transit steps; altitude=0.0 ONLY when arriving at the dock corner.
6. Use BEST_EFFORT QoS unless battery < 0.3. NEVER use RELIABLE when battery ≥ 0.3 — it wastes
   energy on retransmission penalties even on a Good channel.
7. ALWAYS set broadcast_message to the compact status string provided in the
   SYSTEM-CALCULATED WAYPOINTS section (format: "id:x,y;bat:B;pay:P;tgt:tx,ty").
   This gossip keeps all drones informed of each other's position even when the
   DDS channel is in Bad state. NEVER leave broadcast_message null.

Respond with ONLY valid JSON. No prose, no markdown fences."""

_ACTION_SCHEMA = {
    "type": "object",
    "required": ["node_actions"],
    "properties": {
        "node_actions": {
            "type":  "array",
            "items": {
                "type":     "object",
                "required": ["agent_id", "target_waypoint", "pump_activation", "qos_profile", "broadcast_message"],
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

    peer_telem = dds.peer_telemetry
    # Sort for deterministic sector assignment in the PRIORITY 3 sweep.
    # dds.active_peers ordering is not guaranteed across steps — without sorting,
    # the drone_idx used to assign grid sectors would shuffle each tick, causing
    # drones to chase each other's previous sector target.
    active_ids = sorted(dds.active_peers)

    # Build a GE-independent position map from neighbor_states, which is populated
    # for ALL live drones regardless of their DDS channel state. peer_telemetry only
    # carries drones whose GE channel is currently Good — absent entries would yield
    # pos=[0,0] and corrupt every fire-coordinate and navigation-hint calculation.
    ns_pos: Dict[str, List[int]] = {
        ns["id"]: [int(ns["pos"][0]), int(ns["pos"][1])]
        for ns in obs.neighbor_states
    }

    drone_lines: List[str] = []
    for pid in active_ids:
        pt  = peer_telem.get(pid, {})
        # Use GE-independent ns_pos for position; use peer_telem for battery/payload
        # since those are only available on a Good channel — fall back to the swarm average.
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
        # peer_telem only if somehow absent from neighbor_states.
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

    # ── Gossip-seeded fire hints ────────────────────────────────────────────
    # When residual fires drop to 1-2 cells they often fall outside every drone's
    # FOV (radius=7), leaving fire_abs empty and triggering the PRIORITY 3 patrol.
    # Mine tgt: fields from delivered gossip messages — each drone's broadcast
    # contains its last known target which was a fire coordinate. These stale
    # coordinates give the swarm a last-resort search direction even when no
    # drone can currently see the remaining fire.
    gossip_fire_hints: List[List[int]] = []
    for gos_msg in dds.gossip_messages.values():
        try:
            # Format: "id:x,y;bat:B;pay:P;tgt:tx,ty"
            tgt_part = [p for p in gos_msg.split(";") if p.startswith("tgt:")]
            if tgt_part:
                tx_str, ty_str = tgt_part[0].replace("tgt:", "").split(",")
                gtx, gty = int(tx_str), int(ty_str)
                if 0 <= gtx < grid_size and 0 <= gty < grid_size:
                    # Only use if clearly a fire target (not a dock corner or patrol point).
                    corner_set = {(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)}
                    if (gtx, gty) not in corner_set:
                        gossip_fire_hints.append([gtx, gty])
        except Exception:
            pass

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

    # Pre-compute the optimal next waypoint per drone so the LLM never needs to
    # perform clip/clamp arithmetic — it just copies the provided values.
    # Each hint gives the MAX_SPEED=2 constrained step toward the nearest visible fire.
    # Iterating active_ids (not peer_telem) ensures Bad-channel drones still receive hints.
    #
    # Fire zone anchor: fires are placed at grid_size//3*2 (lower two-thirds of the grid).
    # When no fires are in FOV the patrol target must be this anchor row, not grid_size//2
    # (the centre), which would stop drones 3+ rows short of any fire.
    fire_zone_row = grid_size // 3 * 2   # 10 for easy, 13 for medium, 16 for hard
    grid_col_mid  = grid_size // 2

    # Refill threshold: below this payload a drone cannot complete a meaningful Gaussian drop.
    # For multi-drone tasks (medium/hard) 2.0 kg is fine — another drone covers while one refuels.
    # For easy (single drone) we keep suppressing until 1.6 kg (the absolute minimum for one
    # centre-cell drop) so the sole drone does not abandon active fires unnecessarily.
    n_drones = len(active_ids)
    REFILL_THRESHOLD    = 1.6 if n_drones == 1 else 2.0
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

        # Per-drone payload from peer_telem (Good channel) or fall back to the swarm average.
        current_payload = float(pt.get("payload", telem.get("avg_payload", 10.0)))

        # ── PRIORITY 1: Refill if payload is critically low OR drone is already docking ──
        # "Sticky" refuelling: once a drone commits to a corner it stays there until the
        # tank reaches FULL_TANK_THRESHOLD (9.5 kg), not just above REFILL_THRESHOLD.
        # Without stickiness the drone yo-yos — arrives at 1.6 kg, gets one REFILL_RATE=3.0
        # tick to 4.6 kg, then flies back to the fire with less than half a tank, depletes
        # in one pump, and makes the same trip again — wasting ~14 steps per cycle on transit.
        # With stickiness the drone docks for exactly 3 ticks (1.6→4.6→7.6→10.0 kg) and
        # then deploys with a full tank for extended suppression.
        at_corner = (cx, cy) in corners_set

        if current_payload <= REFILL_THRESHOLD or (at_corner and current_payload < FULL_TANK_THRESHOLD):
            rx, ry = min(corners, key=lambda c: abs(cx - c[0]) + abs(cy - c[1]))
            nx = int(max(cx - 2, min(cx + 2, rx)))
            ny = int(max(cy - 2, min(cy + 2, ry)))
            # Altitude rule: use 0.0 ONLY when the next waypoint IS the corner itself
            # (one step away or already there). During multi-step transit use 10.0 so the
            # server does not attempt a ground-level dock on a non-corner cell — docking
            # at altitude=0.0 outside a refill station wastes battery and blocks refill.
            arriving_at_corner = (nx == rx and ny == ry)
            dock_alt = 0.0 if arriving_at_corner else 10.0
            status_msg = "LOW PAYLOAD" if current_payload <= REFILL_THRESHOLD else "REFILLING (stay docked)"
            bat_val = float(peer_telem.get(d_id, {}).get("battery", telem.get("avg_battery", 1.0)))
            bcast = f"{d_id}:{cx},{cy};bat:{bat_val:.2f};pay:{current_payload:.1f};tgt:{rx},{ry}"
            navigation_hints.append(
                f"  {d_id}: {status_msg} ({current_payload:.1f}kg)"
                f" → dock at ({rx},{ry})"
                f" | next_waypoint=[{nx},{ny},{dock_alt}] pump=0.0"
                f" | broadcast_message=\"{bcast}\""
            )
            continue  # skip fire-targeting until the tank is full

        # ── PRIORITY 2: Target the densest nearby fire cluster ───────────────────────────
        # Prefer fires visible in this drone's own FOV; fall back to any fire seen by any
        # drone in the swarm; then fall back to gossip-seeded last-known fire targets.
        # This triple-fallback prevents the loiter trap when residual fires (1-2 cells)
        # fall outside every drone's FOV — without it all drones hit PRIORITY 3,
        # converge on the same fixed anchor, and hover for 20+ steps with reward=0.
        fires_for_drone = fire_abs.get(d_id, [])
        if not fires_for_drone:
            all_fires = [f for fl in fire_abs.values() for f in fl]
            fires_for_drone = all_fires
        if not fires_for_drone and gossip_fire_hints:
            fires_for_drone = gossip_fire_hints

        if fires_for_drone:
            # Score each candidate fire by a combined metric:
            #   cluster_density = number of other fire cells within Chebyshev-3 radius
            #   proximity_score = 1.0 / (1 + manhattan distance)
            # Weighted sum: 0.6 × density + 0.4 × proximity.
            # This steers a single drone toward the largest local cluster — each Gaussian
            # drop suppresses a 3×3 footprint, so arriving at the cluster centre
            # extinguishes far more cells per unit payload than hitting an isolated cell.
            # For multi-drone tasks the density bias also prevents all drones clustering
            # on the same isolated fire while a larger clump goes unchecked.
            # Copy to avoid mutating fire_abs[d_id] in-place via sort() below,
            # which would corrupt the fire list for subsequent drones in this loop.
            all_pool = list(fires_for_drone)
            def _fire_score(f: List[int]) -> float:
                density = sum(
                    1 for g in all_pool
                    if max(abs(g[0] - f[0]), abs(g[1] - f[1])) <= 3
                )
                prox = 1.0 / (1 + abs(f[0] - cx) + abs(f[1] - cy))
                return 0.6 * density + 0.4 * prox

            all_pool.sort(key=_fire_score, reverse=True)
            fx, fy = all_pool[0]
            nx = int(max(cx - 2, min(cx + 2, fx)))
            ny = int(max(cy - 2, min(cy + 2, fy)))

            # Pump when the drone is on the fire cell OR directly adjacent (Chebyshev ≤ 1).
            # The Gaussian footprint covers a 3×3 area so a drone 1 cell away still suppresses
            # the centre cell — waiting for exact overlap wastes at least one transit step.
            cheb_dist = max(abs(fx - cx), abs(fy - cy))
            if cheb_dist <= 1:
                pump_hint = "1.0 (ADJACENT TO FIRE — PUMP NOW)"
                nx, ny = cx, cy  # stay in place and pump
            else:
                pump_hint = "0.0 (TRANSIT)"

            bat_val = float(peer_telem.get(d_id, {}).get("battery", telem.get("avg_battery", 1.0)))
            bcast = f"{d_id}:{cx},{cy};bat:{bat_val:.2f};pay:{current_payload:.1f};tgt:{fx},{fy}"
            navigation_hints.append(
                f"  {d_id}: pos=({cx},{cy}) → cluster=({fx},{fy}) dist={cheb_dist}"
                f" | next_waypoint=[{nx},{ny},10.0] pump={pump_hint}"
                f" | broadcast_message=\"{bcast}\""
            )
        else:
            # ── PRIORITY 3: No fires visible — systematic sector sweep ───────────────────
            # Each drone is assigned a UNIQUE sector of the grid to search so the swarm
            # fans out rather than all converging on a single fixed anchor point.
            #
            # The fixed-anchor bug: when priority-3 sends every drone to the same
            # (fire_zone_row, grid_col_mid) = e.g. (16,12), all drones arrive there,
            # find "no fire" again, stay put, and hover in a zero-reward loiter for
            # 20+ steps while 1-4 residual fires burn elsewhere outside every FOV.
            #
            # Fix: divide the grid into a 2-row × N-col tile grid where N = number of
            # active drones. Each drone index gets its own column band to search.
            # Tile rows alternate between fire_zone_row (lower) and fire_zone_row - 4
            # (upper band) so all sectors cover the likely fire area and the drone
            # always has a distinct target to move toward.
            drone_idx  = list(active_ids).index(d_id) if d_id in active_ids else 0
            n_active   = max(1, len(active_ids))
            col_band   = grid_size // n_active
            # Column centre of this drone's sector, clamped to valid grid range.
            sector_col = int(max(0, min(grid_size - 1, col_band * drone_idx + col_band // 2)))
            # Alternate sector row between lower and upper fire zone per drone index
            # so adjacent drones cover different row bands.
            sector_row = fire_zone_row if drone_idx % 2 == 0 else max(0, fire_zone_row - 4)
            nx = int(max(cx - 2, min(cx + 2, sector_row)))
            ny = int(max(cy - 2, min(cy + 2, sector_col)))
            bat_val = float(peer_telem.get(d_id, {}).get("battery", telem.get("avg_battery", 1.0)))
            bcast = f"{d_id}:{cx},{cy};bat:{bat_val:.2f};pay:{current_payload:.1f};tgt:{sector_row},{sector_col}"
            navigation_hints.append(
                f"  {d_id}: no fire in FOV — sweep sector ({sector_row},{sector_col})"
                f" | next_waypoint=[{nx},{ny},10.0] pump=0.0"
                f" | broadcast_message=\"{bcast}\""
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
        f"  • altitude=0.0 ONLY when the waypoint IS exactly the dock corner — keep altitude=10.0 during all transit\n"
        f"  • QoS BEST_EFFORT always unless battery < 0.3 (RELIABLE triggers retx penalty)\n"
        f"\n{grids_str}"
    )


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

_VALID_QOS = {"BEST_EFFORT", "RELIABLE"}


def _nop_action(obs: SwarmObservation) -> SwarmAction:
    """Stay-in-place zero-pump fallback for LLM parse failures.

    Uses neighbor_states (GE-independent) for position so drones on a Bad
    channel are not sent to (0,0) — which would look like a move command and
    could trigger a loiter penalty or collision on the following step.
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
        bat = float(pt.get("battery", obs.drone_telemetry.get("avg_battery", 1.0)))
        pay = float(pt.get("payload", obs.drone_telemetry.get("avg_payload", 10.0)))
        bcast = f"{pid}:{int(pos[0])},{int(pos[1])};bat:{bat:.2f};pay:{pay:.1f};tgt:nop"
        node_actions.append(DroneNodeAction(
            agent_id          = pid,
            target_waypoint   = (float(pos[0]), float(pos[1]), 10.0),
            pump_activation   = 0.0,
            broadcast_message = bcast,
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

    Deduplicates pump targets so no two drones pump the same (x, y) cell,
    which would trigger the friendly-fire penalty and zero the step reward.
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

        # If two drones target the same cell with pump > 0.5, downgrade the
        # second one to pump=0.0 to avoid the friendly-fire reward penalty.
        key = (int(wp[0]), int(wp[1]))
        if pump > 0.5:
            if key in pumping_targets:
                pump = 0.0
            else:
                pumping_targets.add(key)

        agent_id = str(na.get("agent_id", "D0")) or "D0"

        # QoS guardrail — enforce system prompt Rule 6 at the code level.
        # The LLM occasionally sends RELIABLE even when battery is well above 0.3.
        # RELIABLE triggers GE retransmission penalties even on a Good channel, and
        # the extra retx_penalty subtracts from every step reward needlessly.
        # Override: only allow RELIABLE when this drone's battery is confirmed < 0.3.
        qos_str = str(na.get("qos_profile", "BEST_EFFORT")).upper()
        if qos_str == "RELIABLE":
            drone_bat = float(
                obs.dds_global_space.peer_telemetry.get(agent_id, {}).get("battery", 1.0)
            )
            qos = QoSProfile.RELIABLE if drone_bat < 0.30 else QoSProfile.BEST_EFFORT
        else:
            qos = QoSProfile.BEST_EFFORT

        # broadcast_message: use what the LLM returned; if null/missing build a
        # minimal status string so the DDS gossip layer is never empty.
        msg = na.get("broadcast_message")
        if msg:
            msg = str(msg)[:256]
        else:
            msg = f"{agent_id}:{int(wp[0])},{int(wp[1])};pump:{pump:.1f}"

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

    Falls back to a NOP action on any parse or API error so the episode
    continues rather than crashing mid-run.
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
# Stdout helpers — emit the [START] / [STEP] / [END] lines the validator reads
# ---------------------------------------------------------------------------

def _emit_start(task: str) -> None:
    """Emit the mandatory [START] line to stdout at episode begin."""
    print(f"[START] task={task} env=fire-swarm-simulator model={MODEL_NAME}", flush=True)


def _emit_step(step: int, action: SwarmAction, reward: float, done: bool, error: Optional[str]) -> None:
    """
    Emit one [STEP] line to stdout immediately after env.step() returns.

    action is serialised as a compact JSON string so it fits on one line.
    reward is formatted to 2 decimal places; done and success are lowercase booleans.
    """
    action_str = json.dumps(
        {"node_actions": [a.model_dump() for a in action.node_actions]},
        separators=(",", ":"),
    )
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str}"
        f" reward={reward:.2f} done={'true' if done else 'false'} error={error_str}",
        flush=True,
    )


def _emit_end(success: bool, steps: int, rewards: List[float]) -> None:
    """
    Emit the mandatory [END] line to stdout after the episode closes.

    Always emitted — even if the episode raised an exception — so the
    validator can parse a complete run record.
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'}"
        f" steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task: str, deadline: float) -> Dict[str, Any]:
    """
    Run one complete episode using the FireSwarmEnv WebSocket client.

    Emits [START], one [STEP] per tick, and [END] to stdout in the format
    required by the hackathon validator. The [END] line is guaranteed even
    when the episode terminates due to an exception or wall-clock timeout.
    """
    max_steps = MAX_STEPS_PER_TASK[task]
    log.info("=" * 60)
    log.info("TASK: %s  (max_steps=%d)", task.upper(), max_steps)
    log.info("=" * 60)

    _emit_start(task)

    t0                = time.time()
    cumulative_reward = 0.0
    step              = 0
    done              = False
    rewards: List[float] = []
    last_action: Optional[SwarmAction] = None
    obs = None

    try:
        with FireSwarmEnv(base_url=OPENENV_URL).sync() as env:
            # Fix the seed so the NOP baseline in the grader fights the identical fire.
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

                action      = get_llm_action(client, obs, task, step + 1)
                last_action = action
                result      = env.step(action)
                obs         = result.observation
                reward      = float(result.reward or 0.0)
                done        = bool(result.done)

                cumulative_reward += reward
                rewards.append(reward)
                step += 1

                _emit_step(step, action, reward, done, None)

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

    except Exception as exc:
        log.error("Episode exception in task %s: %s", task, exc)
        _emit_end(success=False, steps=step, rewards=rewards)
        return {
            "task":         task,
            "steps":        step,
            "total_reward": round(cumulative_reward, 4),
            "final_reward": round(rewards[-1], 4) if rewards else 0.0,
            "active_fires": -1,
            "done":         False,
            "elapsed_s":    round(time.time() - t0, 1),
            "score":        0.0,
        }

    active_fires = int(obs.drone_telemetry.get("active_fires", 0)) if obs else -1
    burned_area  = int(obs.drone_telemetry.get("burned_area",  0)) if obs else 0

    # ── Programmatic grader ────────────────────────────────────────────────
    score = grade_episode(
        task              = task,
        cumulative_reward = cumulative_reward,
        steps_taken       = step,
        episode_done      = done,
        active_fires      = active_fires,
        burned_area       = burned_area,
    )

    # success = all fires suppressed (active_fires == 0 and done == True)
    success = done and active_fires == 0
    _emit_end(success=success, steps=step, rewards=rewards)

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
        "final_reward": round(rewards[-1], 4) if rewards else 0.0,
        "active_fires": active_fires,
        "done":         done,
        "elapsed_s":    round(elapsed, 1),
        "score":        score,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # HF_TOKEN is the only mandatory credential with no permitted default.
    if not API_KEY:
        raise ValueError(
            "HF_TOKEN environment variable is required but not set.\n"
            "Export it before running: export HF_TOKEN=hf_..."
        )

    try:
        health = _get(f"{OPENENV_URL}/health", timeout=10)
        log.info("Server health: %s", health.get("status", "unknown"))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach FireSwarm server at {OPENENV_URL}: {exc}\n"
            "Start the server first: uvicorn server.app:app --port 7860"
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
            # Still emit the required [START] and [END] lines for skipped tasks
            # so the validator receives a complete three-task record.
            _emit_start(task)
            _emit_end(success=False, steps=0, rewards=[])
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
