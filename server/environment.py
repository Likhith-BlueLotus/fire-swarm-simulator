"""
FireSwarm physics engine and simulation loop.

Implements the full OpenEnv Environment interface:
  reset()  — initialise a new episode
  step()   — advance one simulation tick
  state    — property returning the current SwarmState

Physics subsystems
──────────────────
  Cellular Automata (CA)         Bernoulli ignition biased by wind, slope, and fuel load
  Ornstein-Uhlenbeck process     Continuous stochastic wind evolution
  Gilbert-Elliott (GE) model     Two-state Markov channel simulation for DDS telemetry
  Gaussian dispersion footprint  3×3 weighted payload drop centred on the pump target
  Battery cost model             E_HOVER + E_TRANSIT×dist + E_PUMP×throttle per tick

Reward structure
────────────────
  +W_EXTINGUISH  normalised agent-extinguishment signal (cells_out / initial_fire_seeds)
                 Natural burnout (BURNT_SCAR from t_burn expiry) is NOT rewarded —
                 only drone-pump extinguishments count.
  +W_PROXIMITY   normalised proximity bonus (dense navigation shaping)
  −W_BATTERY     energy efficiency cost
  −W_LOITER      anti-camping penalty after LOITER_MAX_TICKS idle ticks
  −W_COLLISION   per-drone collision cost
  −retx          RELIABLE QoS retransmission overhead
  =0             entire step zeroed on friendly-fire detection

Key physics invariants
──────────────────────
  - Drones that run out of battery freeze in place and are removed from
    all subsequent calculations (no phantom movement/pumping/collision).
  - Refill and pump are mutually exclusive in one tick: a drone docked at a
    corner station refills; it cannot also pump that step.
  - The CA grid is copied AFTER payload application so freshly extinguished
    cells are never mistakenly treated as fire sources in the same tick.
  - Each episode uses self.rng = np.random.default_rng(seed), fully isolated
    from the global NumPy RNG — concurrent sessions never corrupt each other.
"""

import json
import pathlib
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from openenv.core.env_server import Environment

try:
    from ..models import DDSDataSpace, QoSProfile, SwarmAction, SwarmObservation, SwarmState
except ImportError:
    from models import DDSDataSpace, QoSProfile, SwarmAction, SwarmObservation, SwarmState  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Cell-state constants
# ---------------------------------------------------------------------------
EMPTY        = 0
HEALTHY_FUEL = 1
ACTIVE_FIRE  = 2
EXTINGUISHED = 3
BURNT_SCAR   = 4

# ---------------------------------------------------------------------------
# Task difficulty registry
# ---------------------------------------------------------------------------
TASK_CONFIG: Dict[str, dict] = {
    "easy": {
        "grid_size":  15,
        "num_drones": 1,
        "fire_seeds": 3,
        "t_burn":     8,
        "wind_mult":  1.0,
        "max_steps":  30,
    },
    "medium": {
        "grid_size":  20,
        "num_drones": 3,
        "fire_seeds": 5,
        "t_burn":     6,
        "wind_mult":  1.5,
        "max_steps":  50,
    },
    "hard": {
        "grid_size":  25,
        "num_drones": 5,
        "fire_seeds": 8,
        "t_burn":     5,
        "wind_mult":  2.0,
        "max_steps":  70,
    },
}

# ---------------------------------------------------------------------------
# Egocentric FOV
# ---------------------------------------------------------------------------
CROP_RADIUS = 7

_yy, _xx = np.mgrid[-CROP_RADIUS:CROP_RADIUS + 1, -CROP_RADIUS:CROP_RADIUS + 1]
_FOV_MASK = (_xx ** 2 + _yy ** 2) > CROP_RADIUS ** 2

# ---------------------------------------------------------------------------
# CA fire spread parameters
# ---------------------------------------------------------------------------
BASE_IGNITE  = 0.08
WIND_SCALE   = 0.30
SLOPE_WEIGHT = 0.20

# ---------------------------------------------------------------------------
# Wind OU-process parameters
# ---------------------------------------------------------------------------
OU_THETA     = 0.15
OU_SIGMA     = 0.30
OU_SIGMA_MAG = 0.03

# ---------------------------------------------------------------------------
# DDS / Gilbert-Elliott parameters
# ---------------------------------------------------------------------------
DDS_TRANSITIONS: Dict[str, Tuple[float, float]] = {
    "BEST_EFFORT": (0.10, 0.40),
    "RELIABLE":    (0.02, 0.80),
}
RELIABLE_RETX_PENALTY = 0.02

# ---------------------------------------------------------------------------
# Payload / refill
# ---------------------------------------------------------------------------
MAX_PAYLOAD     = 10.0
EXTINGUISH_COST = 1.0   # cost per centre cell; adjacent weights (0.5×, 0.3×) cost less
REFILL_RATE     = 3.0   # kg refilled per tick while docked at a corner station
DOCKING_ALT     = 5.0

# ---------------------------------------------------------------------------
# Battery cost model
# ---------------------------------------------------------------------------
E_HOVER   = 0.002
E_TRANSIT = 0.008
E_PUMP    = 0.010

# ---------------------------------------------------------------------------
# Drone kinematic constraint
# Max cells a drone can move per step (Chebyshev clamp on each axis).
# Prevents instant teleportation to distant fires — forces genuine navigation.
# At MAX_SPEED=2, a drone that is 10 cells away (Chebyshev) needs ≥5 steps.
# ---------------------------------------------------------------------------
MAX_SPEED = 2

# ---------------------------------------------------------------------------
# Reward weights
# Matches the formula documented in README.md and openenv.yaml grader_formula:
#   R = clip(
#     +0.40 × cells_extinguished     primary suppression signal
#     +0.10 × proximity_bonus        dense navigation shaping
#     −0.05 × battery_drain          efficiency cost
#     −0.15 × loiter_penalty         anti-camping
#     −0.10 × collision_penalty      anti-crowding (transient crossings tolerated)
#     −retx_penalty,                 DDS comms overhead
#     0.0, 1.0)
# ---------------------------------------------------------------------------
W_EXTINGUISH = 0.40
W_PROXIMITY  = 0.10
W_BATTERY    = 0.05
W_LOITER     = 0.15
W_COLLISION  = 0.10  # reduced: transient path crossings shouldn't zero legitimate rewards

LOITER_RADIUS    = 3
LOITER_MAX_TICKS = 5

# ---------------------------------------------------------------------------
# Gaussian dispersion footprint (centre + 4 cardinal + 4 diagonal)
# ---------------------------------------------------------------------------
_GAUSS_OFFSETS: List[Tuple[int, int, float]] = [
    ( 0,  0, 1.0),
    (-1,  0, 0.5), ( 1,  0, 0.5),
    ( 0, -1, 0.5), ( 0,  1, 0.5),
    (-1, -1, 0.3), (-1,  1, 0.3),
    ( 1, -1, 0.3), ( 1,  1, 0.3),
]

_REPLAY_DIR = pathlib.Path("/tmp/fire_swarm_replays")


class FireSwarmEnvironment(Environment):
    """
    Decentralised firefighting UAV swarm environment.

    One instance per concurrent session; safe under
    SUPPORTS_CONCURRENT_SESSIONS = True because each session gets its own
    FireSwarmEnvironment object from the OpenEnv session pool.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._replay_file = None
        self._gossip_inbox: Dict[str, str] = {}
        # RELIABLE QoS retransmission buffer: messages are held here when their
        # drone's GE channel is Bad, then released the next tick the channel
        # recovers to Good. This implements the TCP-like hold-and-release
        # semantics advertised in the README, as opposed to simply charging a
        # penalty while permanently discarding the packet (BEST_EFFORT behaviour).
        self._retx_buffer: Dict[str, str] = {}
        self.reset()

    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        task:       str           = "medium",
        **kwargs,
    ) -> SwarmObservation:
        """Initialise all episode state. Safe to call between episodes."""
        # Each episode gets its own isolated RNG — fully thread-safe under
        # SUPPORTS_CONCURRENT_SESSIONS=True because no global np.random state
        # is touched. Two concurrent grading requests cannot corrupt each other.
        self.rng = np.random.default_rng(seed if seed is not None else 42)

        if self._replay_file is not None:
            try:
                self._replay_file.close()
            except OSError:
                pass

        cfg = TASK_CONFIG.get(task, TASK_CONFIG["medium"])
        gs  = cfg["grid_size"]

        self.task_cfg  = cfg
        self.grid_size = gs
        self.t_burn    = cfg["t_burn"]

        self.grid       = np.full((gs, gs), HEALTHY_FUEL, dtype=np.int8)
        self.fuel_timer = np.zeros((gs, gs), dtype=np.int16)
        self.fuel_grid  = self.rng.uniform(0.4, 1.0, (gs, gs)).astype(np.float32)
        self.elevation  = self.rng.uniform(0.0, 100.0, (gs, gs)).astype(np.float32)

        # Fire seeds placed in the lower two-thirds of the grid (rows gs//3 .. gs-2).
        # Drones spawn at row 0, so all fires are at least gs//3 rows away —
        # with MAX_SPEED=2 that forces ceil((gs//3)/2) ≥ 3 navigation steps minimum.
        # Seeds use ±4 offsets so min inter-seed distance ≥ 4 (no single-pump clears two).
        r = gs // 3 * 2   # lower-grid anchor row (~2/3 down)
        c = gs // 2
        seed_coords = [
            (r,         c),
            (r - 4,     c + 4),
            (r + 3,     c - 4),
            (r - 4,     c - 4),
            (r + 3,     c + 4),
            (r - 1,     c - 7),
            (r - 1,     c + 7),
            (r - 7,     c),
            (r + 3,     c),
            (r - 7,     c - 5),
        ]
        actual_seeds = 0
        for sx, sy in seed_coords[: cfg["fire_seeds"]]:
            sx = int(np.clip(sx, 1, gs - 2))
            sy = int(np.clip(sy, 1, gs - 2))
            self.grid[sx, sy] = ACTIVE_FIRE
            actual_seeds += 1

        self.refill_stations = {
            (0, 0), (0, gs - 1), (gs - 1, 0), (gs - 1, gs - 1),
        }

        # Drones spawn along the top edge of the grid (row 0), evenly spaced.
        # This guarantees:
        #   (a) no drone starts on a fire cell (fires are placed at c ± 4 rows, row ≥ 3)
        #   (b) drones don't collide on step 1 since they fan out from different x columns
        #   (c) each drone must travel several cells to reach its nearest fire
        n = cfg["num_drones"]
        _spawn_cols = [int(round(gs * (i + 1) / (n + 1))) for i in range(n)]
        self.drones: Dict[str, dict] = {
            f"D{i}": {
                "pos": (0, int(np.clip(_spawn_cols[i], 0, gs - 1))),
                "altitude":     10.0,
                "battery":      1.0,
                "payload":      MAX_PAYLOAD,
                "prev_pos":     None,
                "pump":         0.0,
                "qos":          "BEST_EFFORT",
                "loiter_ticks": 0,
                "broadcast":    None,
                "dead":         False,
            }
            for i in range(cfg["num_drones"])
        }

        self.dds_channel_state: Dict[str, int] = {d: 1 for d in self.drones}
        self._retx_count  = 0
        self._gossip_inbox = {}

        ep_id = episode_id or str(uuid.uuid4())
        self._state = SwarmState(
            episode_id=ep_id,
            step_count=0,
            active_fires=actual_seeds,
            total_burned_area=0,
            global_wind_vector=(float(cfg["wind_mult"]), 45.0),
            payload_levels={d: MAX_PAYLOAD for d in self.drones},
            drone_positions={
                d: [float(v["pos"][0]), float(v["pos"][1]), float(v["altitude"])]
                for d, v in self.drones.items()
            },
        )

        try:
            _REPLAY_DIR.mkdir(parents=True, exist_ok=True)
            self._replay_file = open(_REPLAY_DIR / f"{ep_id}.jsonl", "w")
        except OSError:
            self._replay_file = None

        return self._generate_observation()

    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------

    def step(self, action: SwarmAction, timeout_s=None, **kwargs) -> SwarmObservation:
        """
        Advance the simulation by one tick.

        Processing order:
          0. Apply QoS (must precede GE transitions)
          1. GE channel transitions
          2. Action dispatch: kinematic clamp, battery drain, mark dead
          3. Position commit (only for live drones)
          4. Anti-hacking checks: loiter / friendly-fire / collision
          5. Payload application (pump then refill — order matters)
          6. Gossip routing
          7. OU wind drift
          8. CA fire spread (copy grid AFTER payload so extinguished cells
             are not treated as fire sources this same tick)
          9. Global state counts
         10. Reward composition (normalised extinguishment signal)
         11. Terminal condition check + completion bonus + fd close
         12. SwarmState update
         13. Replay log write (before fd close so final step is captured)

        Key invariants enforced:
          - Dead drones never move, pump, refill, or affect collision.
          - A drone that runs out of battery this tick is frozen at its
            last known good position (pre-step pos), not at its target.
          - Refill and pump are mutually exclusive in one tick: a drone
            at a corner station refills instead of pumping.
          - fires_extinguished_by_agent tracks ONLY drone-caused suppressions;
            natural burnout (BURNT_SCAR) is intentionally NOT counted as
            agent extinguishment and does NOT grant W_EXTINGUISH reward.
          - W_EXTINGUISH reward is normalised by initial fire count so that
            extinguishing N cells never exceeds 1.0 by itself.
        """
        self._state.step_count += 1
        battery_drain_total = 0.0

        # ── 0. Apply QoS BEFORE GE transitions ──────────────────────────
        for na in action.node_actions:
            if na.agent_id in self.drones and not self.drones[na.agent_id]["dead"]:
                self.drones[na.agent_id]["qos"] = na.qos_profile.value

        # ── 1. GE channel transitions ───────────────────────────────────
        retx_drones = self._run_ge_transitions()

        # ── 2. Action dispatch: kinematic clamp + battery drain ──────────
        # pending holds the INTENDED new position for each LIVE drone.
        # A drone that runs out of battery this tick is NOT added to pending,
        # so its position is never updated (it stays where it was).
        pending:  Dict[str, Tuple[int, int]] = {}
        pump_map: Dict[str, float]           = {}
        alt_map:  Dict[str, float]           = {}

        for na in action.node_actions:
            drone_id = na.agent_id
            if drone_id not in self.drones:
                continue
            drone = self.drones[drone_id]
            if drone["dead"] or drone["battery"] <= 0.0:
                drone["dead"] = True
                continue

            drone["prev_pos"] = drone["pos"]

            px_cur, py_cur = drone["pos"]
            raw_tx = int(np.clip(na.target_waypoint[0], 0, self.grid_size - 1))
            raw_ty = int(np.clip(na.target_waypoint[1], 0, self.grid_size - 1))
            tx = int(np.clip(raw_tx, px_cur - MAX_SPEED, px_cur + MAX_SPEED))
            ty = int(np.clip(raw_ty, py_cur - MAX_SPEED, py_cur + MAX_SPEED))
            tz = float(np.clip(na.target_waypoint[2], 0.0, 200.0))

            dist  = float(np.sqrt((tx - px_cur) ** 2 + (ty - py_cur) ** 2))
            drain = E_HOVER + E_TRANSIT * dist + E_PUMP * float(na.pump_activation)

            new_bat          = float(max(0.0, drone["battery"] - drain))
            drone["battery"] = new_bat
            battery_drain_total += drain

            if new_bat <= 0.0:
                # Battery died this tick: freeze at current position, mark dead.
                # Do NOT add to pending — position is not updated.
                drone["dead"] = True
                continue

            pending[drone_id]  = (tx, ty)
            pump_map[drone_id] = float(na.pump_activation)
            alt_map[drone_id]  = tz
            drone["pump"]      = pump_map[drone_id]
            drone["broadcast"] = na.broadcast_message

        # ── 3. Position commit (live drones only) ───────────────────────
        for drone_id, (tx, ty) in pending.items():
            self.drones[drone_id]["pos"]      = (tx, ty)
            self.drones[drone_id]["altitude"] = alt_map.get(drone_id, 10.0)

        active_drone_count = max(1, sum(1 for d in self.drones.values() if not d["dead"]))

        # ── 4. Anti-hacking checks ──────────────────────────────────────
        loiter_violations = 0
        for drone_id, drone in self.drones.items():
            if drone["dead"] or drone["prev_pos"] is None:
                drone["loiter_ticks"] = 0
                continue
            px, py = drone["prev_pos"]
            cx, cy = drone["pos"]
            moved   = (cx != px or cy != py)
            pumping = pump_map.get(drone_id, 0.0) > 0.5
            if not moved and not pumping:
                drone["loiter_ticks"] += 1
            else:
                drone["loiter_ticks"] = 0
            if drone["loiter_ticks"] > LOITER_MAX_TICKS:
                loiter_violations += 1

        loiter_penalty = loiter_violations / active_drone_count

        # Collision: two or more LIVE drones at FLIGHT altitude sharing the same cell.
        # A drone docked at a refill station (altitude < DOCKING_ALT = 5 m) is on the
        # ground and does not occupy the airspace above its cell — a transit drone at
        # altitude 10 m flying over it is not a collision.
        active_positions: Dict[Tuple[int, int], List[str]] = {}
        for d_id, drone in self.drones.items():
            if not drone["dead"] and alt_map.get(d_id, 10.0) > DOCKING_ALT:
                active_positions.setdefault(drone["pos"], []).append(d_id)

        # Friendly fire: two or more LIVE drones pumping the SAME cell.
        pump_targets: Dict[Tuple[int, int], List[str]] = {}
        for d_id, (tx, ty) in pending.items():
            if not self.drones[d_id]["dead"] and pump_map.get(d_id, 0.0) > 0.5:
                pump_targets.setdefault((tx, ty), []).append(d_id)
        friendly_fire = any(len(pumpers) > 1 for pumpers in pump_targets.values())

        drones_in_collision = {
            d_id
            for occupants in active_positions.values()
            if len(occupants) > 1
            for d_id in occupants
        }
        collision_penalty = len(drones_in_collision) * W_COLLISION / active_drone_count

        # ── 5. Payload application ──────────────────────────────────────
        # Rule: refill and pump are MUTUALLY EXCLUSIVE per tick.
        #   A drone at a refill corner refills this tick; it cannot also pump.
        #   This prevents free refill+pump in a single step.
        # Rule: only LIVE drones that appear in pending can pump or refill.
        # Rule: pump is blocked when this drone is in a friendly-fire conflict.
        gossip_inbox:      Dict[str, str] = {}
        cells_extinguished = 0

        for drone_id, (tx, ty) in pending.items():
            drone = self.drones[drone_id]
            if drone["dead"]:
                continue

            at_station = (tx, ty) in self.refill_stations and alt_map.get(drone_id, 10.0) < DOCKING_ALT

            if at_station:
                # Refill only — no pump this tick.
                drone["payload"] = float(min(MAX_PAYLOAD, drone["payload"] + REFILL_RATE))
            elif pump_map[drone_id] > 0.5:
                # Pump only — skip if friendly-fire conflict on this cell.
                if len(pump_targets.get((tx, ty), [])) > 1:
                    pass  # blocked; no extinguishment, no payload cost
                else:
                    remaining = drone["payload"]
                    for dx, dy, weight in _GAUSS_OFFSETS:
                        cx_ = tx + dx
                        cy_ = ty + dy
                        cost = EXTINGUISH_COST * weight
                        # Always deduct payload cost — water dropped off-grid is
                        # wasted (evaporates), not double-pumped onto an edge cell.
                        # np.clip would silently redirect the off-grid drop back
                        # onto the boundary cell, charging payload twice for the
                        # same hit and draining the tank with no physical benefit.
                        if remaining >= cost:
                            remaining -= cost
                            if 0 <= cx_ < self.grid_size and 0 <= cy_ < self.grid_size:
                                if self.grid[cx_, cy_] == ACTIVE_FIRE:
                                    self.grid[cx_, cy_]       = EXTINGUISHED
                                    self.fuel_timer[cx_, cy_] = 0
                                    cells_extinguished       += 1
                    drone["payload"] = float(max(0.0, remaining))

            msg = drone.get("broadcast")
            qos = drone.get("qos", "BEST_EFFORT")
            ch  = self.dds_channel_state[drone_id]

            if msg:
                if ch == 1:
                    # Channel is Good: deliver immediately.
                    gossip_inbox[drone_id] = msg
                elif qos == "RELIABLE":
                    # Channel is Bad but QoS is RELIABLE: buffer the message.
                    # It will be delivered next tick the channel recovers.
                    # BEST_EFFORT packets on a Bad channel are silently dropped
                    # (no buffer), matching UDP fire-and-forget semantics.
                    self._retx_buffer[drone_id] = msg

            # Release any buffered RELIABLE message when channel recovers.
            # This models TCP retransmission: the packet was queued at the
            # sender, arrives one step late, but is not permanently lost.
            if ch == 1 and drone_id in self._retx_buffer:
                gossip_inbox[drone_id] = self._retx_buffer.pop(drone_id)

        # ── 6. Gossip ────────────────────────────────────────────────────
        self._gossip_inbox = gossip_inbox

        # ── 7. OU wind drift ────────────────────────────────────────────
        mag, angle = self._state.global_wind_vector
        base_mag   = float(self.task_cfg["wind_mult"])
        # Compute the shortest angular path to the mean-reversion target (45°).
        # Without this, when angle > 225° the naive (45 - angle) term is large
        # and negative, making the OU process spin the wrong direction around
        # the circle instead of taking the short arc back to 45°.
        angle_diff  = (45.0 - angle + 180.0) % 360.0 - 180.0
        angle      += OU_THETA * angle_diff + OU_SIGMA * float(self.rng.standard_normal())
        mag        += OU_THETA * (base_mag - mag) + OU_SIGMA_MAG * float(self.rng.standard_normal())
        mag         = float(np.clip(mag, 0.1 * base_mag, 3.0 * base_mag))
        angle       = float(angle % 360.0)
        self._state.global_wind_vector = (mag, angle)

        # ── 8. CA fire spread ────────────────────────────────────────────
        # CRITICAL: copy self.grid AFTER payload application so cells that
        # were just extinguished this tick are NOT treated as fire sources
        # by the CA this same tick. Without this, a freshly EXTINGUISHED
        # cell (now written into self.grid) would still appear as ACTIVE_FIRE
        # in new_grid (copied before payload write) and ignite its neighbours.
        new_grid  = self.grid.copy()
        new_timer = self.fuel_timer.copy()
        wind_rad  = np.radians(angle)
        wind_dx   = float(np.cos(wind_rad))
        wind_dy   = float(np.sin(wind_rad))
        gs        = self.grid_size

        for x in range(gs):
            for y in range(gs):
                cell = self.grid[x, y]

                if cell == HEALTHY_FUEL:
                    x0, x1 = max(0, x - 1), min(gs, x + 2)
                    y0, y1 = max(0, y - 1), min(gs, y + 2)
                    patch  = self.grid[x0:x1, y0:y1]
                    if not np.any(patch == ACTIVE_FIRE):
                        continue

                    fire_pos  = np.argwhere(patch == ACTIVE_FIRE)
                    mean_fire = fire_pos.mean(axis=0)
                    to_cell   = np.array([
                        x - (x0 + mean_fire[0]),
                        y - (y0 + mean_fire[1]),
                    ])
                    norm = float(np.linalg.norm(to_cell))

                    wind_factor = 1.0
                    if norm > 0:
                        cos_theta   = float(np.dot([wind_dx, wind_dy], to_cell / norm))
                        wind_factor = 1.0 + WIND_SCALE * mag * max(0.0, cos_theta)

                    fuel_factor = float(self.fuel_grid[x, y])

                    fxi         = int(np.clip(x0 + mean_fire[0], 0, gs - 1))
                    fyi         = int(np.clip(y0 + mean_fire[1], 0, gs - 1))
                    delta_h     = float(self.elevation[x, y] - self.elevation[fxi, fyi])
                    slope_rad   = float(np.arctan2(abs(delta_h), max(norm, 0.1)))
                    slope_sign  = 1.0 if delta_h > 0 else -0.5
                    slope_factor = float(np.clip(
                        1.0 + SLOPE_WEIGHT * float(np.sin(slope_rad)) * slope_sign,
                        0.5, 2.0,
                    ))

                    p_ignite = float(np.clip(
                        BASE_IGNITE * fuel_factor * wind_factor * slope_factor,
                        0.0, 1.0,
                    ))
                    if self.rng.random() < p_ignite:
                        new_grid[x, y]  = ACTIVE_FIRE
                        new_timer[x, y] = 0

                elif cell == ACTIVE_FIRE:
                    new_timer[x, y] += 1
                    if new_timer[x, y] >= self.t_burn:
                        new_grid[x, y]  = BURNT_SCAR
                        new_timer[x, y] = 0

        self.grid       = new_grid
        self.fuel_timer = new_timer

        # ── 9. Global state counts ──────────────────────────────────────
        self._state.active_fires      = int(np.sum(self.grid == ACTIVE_FIRE))
        self._state.total_burned_area = int(np.sum(self.grid == BURNT_SCAR))

        # ── 10. Reward composition ──────────────────────────────────────
        # W_EXTINGUISH contribution is normalised by initial fire count so
        # extinguishing all N initial seeds yields at most W_EXTINGUISH (0.40),
        # not N×0.40.  Gaussian splash hitting multiple cells never inflates
        # the reward above the per-step cap before clip.
        initial_fires   = max(1, self.task_cfg["fire_seeds"])
        # Cap at 1.0: fire naturally spreads beyond the initial seed count, so a
        # multi-drone simultaneous drop on a grown fire can extinguish more cells
        # than initial_fires (e.g. 15 cells / 8 seeds = 1.875 → W_EXTINGUISH×1.875
        # = 0.75 in a single tick, blowing past the documented ≤0.40 maximum and
        # overpowering all battery/loiter penalties). Clamping here ensures the
        # signal stays in [0,1] before the W_EXTINGUISH weight is applied.
        extinguish_signal = min(1.0, float(cells_extinguished) / initial_fires)

        fire_positions  = np.argwhere(self.grid == ACTIVE_FIRE)
        proximity_bonus = 0.0
        if len(fire_positions) > 0:
            max_possible = float(self.grid_size * np.sqrt(2))
            for drone in self.drones.values():
                if drone["dead"]:
                    continue
                dx = fire_positions[:, 0] - drone["pos"][0]
                dy = fire_positions[:, 1] - drone["pos"][1]
                min_dist = float(np.min(np.sqrt(dx ** 2 + dy ** 2)))
                proximity_bonus += 1.0 - min_dist / max_possible
            proximity_bonus /= active_drone_count

        battery_penalty = battery_drain_total / active_drone_count
        retx_penalty    = len(retx_drones) * RELIABLE_RETX_PENALTY / active_drone_count

        raw_reward = (
              W_EXTINGUISH * extinguish_signal
            + W_PROXIMITY  * proximity_bonus
            - W_BATTERY    * battery_penalty
            - W_LOITER     * loiter_penalty
            - W_COLLISION  * collision_penalty
            - retx_penalty
        )

        if friendly_fire:
            raw_reward = 0.0

        step_reward = float(np.clip(raw_reward, 0.0, 1.0))

        # ── 11. Terminal conditions + completion bonus ──────────────────
        all_out  = self._state.active_fires == 0
        timeout  = self._state.step_count >= self.task_cfg.get("max_steps", 70)
        all_dead = all(d["dead"] for d in self.drones.values())
        done     = all_out or timeout or all_dead

        # Completion bonus: small reward for clearing all fires (including
        # natural burnout — the swarm's containment strategy still wins).
        # Does NOT fire on timeout unless fires also happen to be out.
        if all_out:
            step_reward = float(np.clip(step_reward + 0.15, 0.0, 1.0))

        # ── 12. SwarmState update ───────────────────────────────────────
        self._state.payload_levels = {
            d_id: float(d["payload"]) for d_id, d in self.drones.items()
        }
        self._state.drone_positions = {
            d_id: [float(d["pos"][0]), float(d["pos"][1]), float(d["altitude"])]
            for d_id, d in self.drones.items()
        }

        obs        = self._generate_observation()
        obs.reward = step_reward
        obs.done   = done

        # ── 13. Replay log ──────────────────────────────────────────────
        # ORDERING: write BEFORE the fd-close block below so the terminal
        # step (done=True) is captured in the JSONL. Moving the close to
        # after the write was critical: previously the file was closed at
        # step 11, making _replay_file=None here and silently dropping the
        # final entry from every episode log.
        if self._replay_file is not None:
            try:
                self._replay_file.write(json.dumps({
                    "step":                    self._state.step_count,
                    "reward":                  step_reward,
                    "done":                    done,
                    "active_fires":            self._state.active_fires,
                    "burned_area":             self._state.total_burned_area,
                    "cells_extinguished":      cells_extinguished,
                    "extinguish_signal":       round(extinguish_signal, 4),
                    "payload_levels":          self._state.payload_levels,
                    "drone_positions":         self._state.drone_positions,
                    "retx_drones":             retx_drones,
                    "friendly_fire":           friendly_fire,
                    "collision_drones":        list(drones_in_collision),
                    "proximity_bonus":         round(proximity_bonus, 4),
                }) + "\n")
                self._replay_file.flush()
            except OSError:
                pass

        # Close replay fd after writing so the terminal step is not lost.
        # Prevents OS-level file descriptor leaks under high-concurrency
        # rollouts (openenv.yaml: max_concurrent_envs: 2048 — each worker
        # that completes without calling reset() again would hold an open
        # fd until container kill, triggering OS Error 24).
        if done and self._replay_file is not None:
            try:
                self._replay_file.close()
            except OSError:
                pass
            finally:
                self._replay_file = None

        return obs

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _run_ge_transitions(self) -> List[str]:
        """
        Advance Gilbert-Elliott channel states for all live drones.

        Returns the list of drone IDs that were in Bad state while using
        RELIABLE QoS — these incur a retransmission penalty this tick.
        """
        retx_drones: List[str] = []
        for drone_id, drone in self.drones.items():
            if drone["dead"]:
                continue
            qos      = drone.get("qos", "BEST_EFFORT")
            p_gb, p_bg = DDS_TRANSITIONS.get(qos, DDS_TRANSITIONS["BEST_EFFORT"])
            ch       = self.dds_channel_state[drone_id]
            if ch == 0 and qos == "RELIABLE":
                retx_drones.append(drone_id)
            if ch == 1 and self.rng.random() < p_gb:
                ch = 0
            elif ch == 0 and self.rng.random() < p_bg:
                ch = 1
            self.dds_channel_state[drone_id] = ch
        return retx_drones

    def _generate_observation(self) -> SwarmObservation:
        """Build a SwarmObservation from the current environment state."""
        gossip_inbox = getattr(self, "_gossip_inbox", {})

        peer_telemetry:   Dict[str, dict] = {}
        delivered_gossip: Dict[str, str]  = {}

        for drone_id, drone in self.drones.items():
            if drone["dead"]:
                continue
            ch  = self.dds_channel_state[drone_id]
            qos = drone.get("qos", "BEST_EFFORT")
            telemetry_entry = {
                "battery": float(drone["battery"]),
                "payload": float(drone["payload"]),
                "pos":     list(drone["pos"]),
                "alt":     float(drone["altitude"]),
                "loiter":  int(drone["loiter_ticks"]),
            }
            if ch == 1:
                peer_telemetry[drone_id] = telemetry_entry
                if drone_id in gossip_inbox:
                    delivered_gossip[drone_id] = gossip_inbox[drone_id]
            elif qos == "RELIABLE":
                peer_telemetry[drone_id] = {**telemetry_entry, "retx": True}

        dds_space = DDSDataSpace(
            active_peers=[d for d, v in self.drones.items() if not v["dead"]],
            peer_telemetry=peer_telemetry,
            gossip_messages=delivered_gossip,
        )

        # neighbor_states MUST iterate all live drones regardless of GE channel.
        # peer_telemetry already filters out Bad-channel drones, so using it here
        # would silently omit them and cause inference.py's ns_pos lookup to fall
        # back to [0,0] — exactly the stale-position bug the ns_pos fix was meant
        # to prevent. GE state is reported as "channel" so callers can act on it,
        # but the position/battery/payload fields are always ground-truth.
        neighbor_states: List[dict] = [
            {
                "id":      d_id,
                "pos":     list(self.drones[d_id]["pos"]),
                "alt":     float(self.drones[d_id]["altitude"]),
                "battery": float(self.drones[d_id]["battery"]),
                "payload": float(self.drones[d_id]["payload"]),
                "channel": "Good" if self.dds_channel_state[d_id] == 1 else "Bad",
            }
            for d_id in self.drones
            if not self.drones[d_id]["dead"]
        ]

        per_drone_grids: Dict[str, List[List[float]]] = {
            d_id: self._egocentric_crop(d_id)
            for d_id in self.drones
            if not self.drones[d_id]["dead"]
        }

        representative_id = next(
            (d for d in self.drones if not self.drones[d]["dead"]),
            next(iter(self.drones)),
        )
        egocentric = per_drone_grids.get(representative_id, [[]])

        alive     = [d for d in self.drones.values() if not d["dead"]]
        batteries = [d["battery"] for d in alive] or [0.0]
        payloads  = [d["payload"]  for d in alive] or [0.0]

        retx_count = sum(
            1 for d, ch in self.dds_channel_state.items()
            if ch == 0 and self.drones[d].get("qos") == "RELIABLE"
        )

        drone_telemetry: Dict[str, float] = {
            "avg_battery":  float(np.mean(batteries)),
            "min_battery":  float(np.min(batteries)),
            "avg_payload":  float(np.mean(payloads)),
            "min_payload":  float(np.min(payloads)),
            "active_fires": float(self._state.active_fires),
            "burned_area":  float(self._state.total_burned_area),
            "alive_drones": float(len(alive)),
            "retx_count":   float(retx_count),
        }

        wind_mag, wind_angle = self._state.global_wind_vector
        return SwarmObservation(
            local_grid_thermal=egocentric,
            per_drone_grids=per_drone_grids,
            drone_telemetry=drone_telemetry,
            neighbor_states=neighbor_states,
            dds_global_space=dds_space,
            wind_vector=(wind_mag, wind_angle),
            done=False,
            reward=0.0,
        )

    def _egocentric_crop(self, drone_id: str) -> List[List[float]]:
        """Return a 15×15 circular FOV grid centred on the given drone."""
        x, y   = self.drones[drone_id]["pos"]
        R      = CROP_RADIUS
        padded = np.pad(self.grid, R, constant_values=EMPTY)
        px, py = x + R, y + R
        crop   = padded[px - R: px + R + 1, py - R: py + R + 1].astype(np.float32).copy()
        crop[_FOV_MASK] = float(EMPTY)
        return crop.tolist()

    # -----------------------------------------------------------------------
    # OpenEnv required property
    # -----------------------------------------------------------------------

    @property
    def state(self) -> SwarmState:
        return self._state
