---
title: FireSwarm — Decentralised Firefighting UAV Swarm Environment
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
license: bsd-3-clause
short_description: MARL UAV firefighting sim with CA fire spread & DDS
tags:
  - reinforcement-learning
  - multi-agent
  - simulation
  - openenv
  - wildfire
  - uav
---

# FireSwarm — Decentralised Firefighting UAV Swarm Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible multi-agent reinforcement learning environment that simulates a real-world wildfire and infrastructure fire suppression mission using a fleet of autonomous UAVs.

---

## Motivation

The Middle East is experiencing an unprecedented convergence of threats that make autonomous aerial firefighting not a research curiosity but an operational necessity.

**Conflict-driven fire risk.** The ongoing Iran–Israel confrontation and the broader regional instability involving Yemen, Lebanon, and Gaza have introduced a new category of fire hazard: infrastructure fires triggered by missile strikes, drone swarms, and precision munitions hitting oil refineries, gas pipelines, electrical substations, and industrial zones. The 2019 Abqaiq–Khurais attack on Saudi Aramco — which briefly knocked out 5% of global oil supply — demonstrated how a single coordinated strike can ignite multiple simultaneous fires across a sprawling petrochemical facility. Human crews cannot safely enter active strike zones. Autonomous UAV swarms can.

**Gulf climate extremes.** The UAE, Saudi Arabia, Qatar, and Kuwait regularly record ambient temperatures above 50 °C in summer. At these temperatures, vegetation fires, electrical fires, and chemical storage fires spread faster than human response chains can coordinate. Dubai's dense urban canyon geometry — supertall towers, underground metro infrastructure, marina districts — creates wind tunnels that accelerate fire spread in ways that demand real-time aerial coordination, not manual dispatch.

**Communication-degraded environments.** Electronic warfare is now standard in Middle East conflict theatres. GPS jamming, RF spoofing, and 5G/LTE disruption are documented across Syria, Iraq, Lebanon, and the Red Sea corridor. Any firefighting drone fleet operating near a conflict zone must be designed from day one to work under degraded, intermittent, and actively jammed communications — exactly the failure mode our Gilbert-Elliott DDS model simulates.

**Why this simulator fills a real gap.** Existing wildfire RL environments (e.g. gym-cellular-automaton, FireGym) model California-style forest fires in benign electromagnetic environments with perfect inter-agent communication. None model:

- Simultaneous multi-ignition from strike patterns (4–8 fire seeds, not 1)
- High wind-multiplier scenarios matching Gulf shamal wind events (up to 80 km/h)
- Packet-loss-aware swarm coordination under RF-contested conditions
- Payload-constrained refuelling logistics (drones can't loiter indefinitely at 50 °C)

FireSwarm models all of these. An agent trained here learns behaviours directly transferable to real UAE Civil Defence UAV fleets, Saudi Aramco emergency response systems, and NATO-aligned Gulf partner force protection units.

---

## Real-World Deployment Context


| Scenario                      | Location               | Fire type                | Swarm challenge                       |
| ----------------------------- | ---------------------- | ------------------------ | ------------------------------------- |
| Oil refinery strike response  | Jubail, Saudi Arabia   | Petrochemical / BLEVE    | Multi-ignition, toxic exclusion zone  |
| Urban high-rise fire          | Dubai Marina / DIFC    | Structural + facade      | Wind tunnel effect, GPS canyon shadow |
| Gas pipeline rupture          | Qatar–UAE corridor     | Pressurised gas jet fire | Dynamic spread, no-fly enforcement    |
| Ammunition depot fire         | Conflict-adjacent zone | Explosive ordnance       | RF jamming, no human entry            |
| Shamal-driven vegetation fire | Al Ain / Oman border   | Dry scrub, 80 km/h wind  | Fast spread, limited water access     |


The `hard` task (5 drones, 8 fire seeds, windmult=2×, tburn=5 ticks, base_ignite=0.10, 25×25 grid) directly models the Abqaiq scenario: multiple simultaneous ignition points spread across all quadrants, high wind, fast burn-through, and a drone fleet that must coordinate multi-hop routing to suppress all fires within 70 steps.

---

## Environment Description

A swarm of UAVs operates over a discrete grid. Fire spreads via a **Cellular Automata** (CA) model driven by wind, fuel density, and terrain slope. Agents observe their local environment through an egocentric 15×15 circular field-of-view and communicate over a **Gilbert-Elliott two-state Markov** DDS network (simulating real-world packet loss from RF interference or smoke-induced 5G attenuation). Each drone carries a finite retardant payload and must refuel at ground stations.

### Key mechanics


| Feature       | Detail                                                                                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Grid topology | Discrete sector grid: easy=15×15, medium=20×20, hard=25×25                                                                                               |
| Fire physics  | CA Bernoulli ignition with wind bias (OU process), fuel load, slope factor; `base_ignite` is task-specific (0.05 easy / 0.08 medium / 0.10 hard)          |
| Drone fleet   | 1 / 3 / 5 drones for easy / medium / hard                                                                                                                |
| DDS network   | Gilbert-Elliott two-state Markov (BESTEFFORT and RELIABLE QoS profiles)                                                                                  |
| Observation   | 15×15 circular FOV per drone, neighbour telemetry, gossip messages, wind vector                                                                          |
| Payload       | 10 kg/drone; 1 kg centre, 0.5 kg cardinal, 0.3 kg diagonal (Gaussian); refill at corners (<5 m alt); **refill and pump are mutually exclusive per tick** |
| Battery       | Transit-distance model: EHOVER + ETRANSIT×dist + EPUMP×throttle; zero-battery drones freeze at last valid cell                                           |
| Max speed     | 2 cells per axis per step (Chebyshev clamp) — drones cannot teleport to distant fires                                                                    |
| Dead drone    | Battery=0 → position frozen, never pumps, never refills, excluded from reward and collision                                                              |


---

## Action Space

Each step the agent submits a `**SwarmAction`** containing one `DroneNodeAction` per active drone:

```json
{
  "node_actions": [
    {
      "agent_id":          "D0",
      "target_waypoint":   [10.0, 12.0, 15.0],
      "pump_activation":   0.8,
      "broadcast_message": "fire spotted at 10,12",
      "qos_profile":       "BEST_EFFORT"
    }
  ]
}
```


| Field               | Type                         | Description                                                             |
| ------------------- | ---------------------------- | ----------------------------------------------------------------------- |
| `agent_id`          | `str`                        | Drone identifier: `"D0"`, `"D1"`, …                                     |
| `target_waypoint`   | `[float, float, float]`      | `[x, y, altitude_m]`; x/y in `[0, grid_size-1]`, altitude in `[0, 200]` |
| `pump_activation`   | `float [0.0–1.0]`            | Retardant throttle. `>0.5` triggers extinguishing                       |
| `broadcast_message` | `str | null`                 | DDS gossip payload (max 256 chars); delivered only on Good channel      |
| `qos_profile`       | `"BEST_EFFORT" | "RELIABLE"` | DDS QoS contract governing this broadcast                               |


---

## Observation Space

Each step returns a `**SwarmObservation**`:


| Field                | Type                           | Description                                                                                                  |
| -------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `local_grid_thermal` | `List[List[float]]` 15×15      | Egocentric FOV for drone D0 (cells: 0=EMPTY, 1=FUEL, 2=FIRE, 3=EXTINGUISHED, 4=SCAR)                         |
| `per_drone_grids`    | `Dict[str, List[List[float]]]` | Per-drone 15×15 circular FOV grids                                                                           |
| `drone_telemetry`    | `Dict[str, float]`             | Fleet diagnostics: `avg_battery`, `min_battery`, `avg_payload`, `active_fires`, `burned_area`, `retx_count`  |
| `neighbor_states`    | `List[dict]`                   | Peer snapshots filtered through Gilbert-Elliott channel: `id`, `pos`, `alt`, `battery`, `payload`, `channel` |
| `dds_global_space`   | `DDSDataSpace`                 | `active_peers`, `peer_telemetry`, `gossip_messages`                                                          |
| `wind_vector`        | `[float, float]`               | `(magnitude_m_s, bearing_degrees)`                                                                           |
| `reward`             | `float [0.0–1.0]`              | Normalised step reward                                                                                       |
| `done`               | `bool`                         | Episode terminal flag                                                                                        |


---

## Reward Function

```
R = clip(
  + 0.40 × (cells_extinguished_by_drones / initial_fire_seeds)   ← normalised suppression signal
  + 0.10 × proximity_to_nearest_fire                              ← dense navigation shaping
  − 0.05 × battery_drain_per_drone                                ← efficiency cost
  − 0.15 × loiter_penalty                                         ← anti-camping
  − 0.10 × collision_penalty                                      ← anti-crowding
  − retx_penalty,                                                  ← 0.02 per RELIABLE drone on Bad channel
  0.0, 1.0)
```

- **Suppression signal**: normalised by `initial_fire_seeds` so extinguishing the entire swarm in one step yields at most `+0.40`, never a clipped-to-1.0 shortcut
- **Agent-only credit**: only drone-pump extinguishments are counted; natural burnout (BURNTSCAR after `t_burn` ticks) does **not** grant reward
- **proximitybonus**: average `(1 − dist/max_dist)` over all live drones to their nearest fire — non-zero during transit
- **BatteryDrain**: per-active-drone `E_HOVER + E_TRANSIT×dist + E_PUMP×throttle`
- **LoiterPenalty**: fraction of drones idle >5 ticks **and** not pumping (pumping-in-place is legal)
- **CollisionPenalty**: fraction of drones sharing the same (x, y) cell
- **RetxPenalty**: 0.02 per RELIABLE-QoS drone on a Bad channel
- **FriendlyFire**: two or more drones pumping the same cell → reward zeroed **and** extinguishment blocked
- **Refill**: drones at a corner station refill payload (+3 kg/tick); refill and pump are **mutually exclusive** per tick
- **Dead drones**: zero-battery drones freeze in place, never pump, never refill, and are excluded from all reward calculations
- **Completion bonus**: `+0.15` added when all fires are out (includes natural burnout — containment still wins)

---

## RL Agent Compatibility (PPO / MAPPO)

The LLM-based `inference.py` is the **hackathon baseline**. The environment is deliberately designed so that a traditional deep-RL agent — particularly PPO or its multi-agent variant MAPPO — can be dropped in with minimal glue code. This section documents the exact mapping for any team that wants to train a policy network from scratch.

### Observation vector (per drone)

The raw `SwarmObservation` must be flattened into a fixed-size tensor. A practical encoding:

| Component | Shape | Notes |
|---|---|---|
| Egocentric FOV grid | `(225,)` | 15×15 = 225 cells, values ∈ {0,1,2,3,4}, normalise ÷ 4 |
| Wind vector | `(2,)` | `(magnitude/3.0, angle/360.0)` → [0,1] |
| Own battery | `(1,)` | already ∈ [0,1] |
| Own payload | `(1,)` | ÷ 10.0 → [0,1] |
| Peer positions (N–1 drones) | `(N-1, 3)` | `(x/gs, y/gs, battery)` — zero-pad for dead/lost peers |
| Active fire count | `(1,)` | ÷ `grid_size²` → [0,1] |
| **Total (hard, N=5)** | **≈ 240** | Fully continuous, no discrete tokens |

The observation is **fully continuous and bounded**, making it directly compatible with any actor-critic that accepts a flat `Box` space — no embedding layers required.

### Action head

Each drone's action is a **4-dimensional continuous vector**:

```
[Δx, Δy, altitude, pump_activation]
   ∈ [−2, +2] × [−2, +2] × [0, 200] × [0, 1]
```

The server enforces the `MAX_SPEED=2` Chebyshev clamp, so out-of-range waypoints are silently clamped — the policy can output raw deltas without a hard `tanh` squash. `pump_activation > 0.5` triggers the Gaussian retardant drop.

For **MAPPO** (Multi-Agent PPO with a shared centralised critic), the global state fed to the critic can be constructed from `SwarmState`:

```python
global_state = np.concatenate([
    self_obs_flat,                      # own FOV + telemetry
    peer_positions.flatten(),           # all drone (x, y, battery, payload)
    [active_fires / grid_size**2],      # fire pressure scalar
    [wind_mag / 3.0, wind_angle / 360], # global wind
])  # shape ≈ (260,) for hard task
```

### Reward shaping compatibility

The per-step reward `R ∈ [0.0, 1.0]` is already dense and shaped for RL:

- `W_PROXIMITY = 0.10` gives a non-zero gradient signal during the entire transit phase (before the first pump), preventing the sparse-reward cold-start problem that kills vanilla PPO on navigation tasks.
- `W_EXTINGUISH = 0.40` is normalised by `initial_fire_seeds` — the gradient magnitude is stable across all three task difficulties without reward rescaling.
- The loiter penalty (`W_LOITER = 0.15`) and collision penalty (`W_COLLISION = 0.10`) act as implicit regularisers that discourage the mode collapse behaviour (all drones converging on one fire) common in cooperative MARL without communication.

### Training sketch (MAPPO, ~500k steps)

```python
from stable_baselines3 import PPO          # single-agent baseline
# or: from marllib import MAPPO            # full multi-agent

env = FireSwarmEnv(base_url="http://localhost:7860")

# Observation: flat vector ~240-dim Box
# Action:      flat vector 4-dim Box per drone
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.995,          # long horizon — fires spread over 30–70 steps
    gae_lambda=0.97,
    ent_coef=0.01,        # encourage exploration of fire-zone entry
    learning_rate=3e-4,
    clip_range=0.2,
    verbose=1,
)
model.learn(total_timesteps=500_000)
```

Key hyperparameter notes:
- `gamma=0.995` (not 0.99): the completion bonus arrives at step 30–70, so a higher discount is critical to propagate the terminal reward back to early transit decisions.
- `ent_coef=0.01`: the loiter penalty already discourages hovering, but entropy regularisation ensures the policy explores different quadrant orderings instead of always attacking the nearest seed first.
- For MAPPO, the **centralised critic** eliminates the non-stationarity problem (each agent's reward depends on other agents' pumping decisions) that causes vanilla independent PPO to diverge on this task.

### LLM agent vs. trained PPO — expected trade-offs

| Criterion | LLM baseline (`gpt-4o-mini`) | Trained PPO |
|---|---|---|
| Zero-shot generalisation | Strong — handles unseen seed layouts | Requires retraining for new configs |
| Reaction latency | ~2–3 s/step (API round-trip) | <1 ms/step (local inference) |
| Navigation precision | Limited by token-level arithmetic | Exact — policy outputs continuous Δx, Δy |
| Communication modelling | Rules-based QoS switching; gossip-seeded residual fire search | Can learn optimal QoS-switching policy end-to-end |
| Sample efficiency | 0 environment steps needed | ~500k steps to match or surpass LLM |
| Interpretability | Prompt-readable decision trace | Opaque weight tensor |

The LLM baseline scores **0.81 overall** (easy=0.86, medium=0.80, hard=0.78) without any environment interaction during training. All three tasks clear fires to zero well within their step budgets, demonstrating genuine navigation and coordination under communication noise and communication-degraded conditions. The easy task clears in 2 steps; medium in 37/50; hard in 35/70 with 5 coordinating drones. A trained PPO policy would achieve even higher scores by learning optimal pump timing and tighter routing — the dense reward signal (proximity bonus + extinguishment signal) supports fast convergence.

---


| Task     | Grid  | Drones | Fire seeds | Burn timer | Wind mult | Base ignite | Max steps | Analogue scenario                                |
| -------- | ----- | ------ | ---------- | ---------- | --------- | ----------- | --------- | ------------------------------------------------ |
| `easy`   | 15×15 | 1      | 1          | 10 ticks   | 0.8×      | 0.05        | 40        | Single ignition, single-drone response           |
| `medium` | 20×20 | 3      | 5          | 6 ticks    | 1.5×      | 0.08        | 50        | Multi-ignition industrial district, gusty wind   |
| `hard`   | 25×25 | 5      | 8          | 5 ticks    | 2×        | 0.10        | 70        | Multi-strike petrochemical facility, shamal wind |


**Easy task placement**: The single fire seed is placed at row `gs//4 = 3` (near drone spawn at row 0), reachable in 2 steps. With slow fire spread (`base_ignite=0.05`, `wind_mult=0.8`), the drone can arrive and suppress before significant secondary ignitions occur.

**Medium/hard placement**: Fires are placed in the lower two-thirds of the grid, forcing genuine multi-step navigation and swarm coordination across all quadrants. Minimum inter-seed distance ≥ 4 cells — no single Gaussian drop clears two fires simultaneously.

### Grader criteria (score 0.0–1.0)

```
score = 0.35 × fire_suppression_ratio       (vs. uncontrolled NOP baseline)
      + 0.25 × (1 − burned_area / grid_cells)
      + 0.20 × normalised_cumulative_reward
      + 0.20 × completion_bonus (1.0 if all fires out, 0.5 if beat baseline, 0.0 otherwise)
```

Scores are computed relative to an uncontrolled NOP baseline (drones stationary, pump=0) run for the same number of steps. An agent that does nothing scores ≈ 0.25. Meaningful scores require multi-step active fire suppression across all quadrants.

---

## Baseline Scores

Measured with `gpt-4o-mini` (temperature=0.2).


| Task             | Steps taken | Fires left | Score      |
| ---------------- | ----------- | ---------- | ---------- |
| `easy`           | 2 / 40      | 0          | **0.8644** |
| `medium`         | 37 / 50     | 0          | **0.7975** |
| `hard`           | 35 / 70     | 0          | **0.7799** |
| **Overall mean** | —           | —          | **0.8139** |


```
JSON_SCORES: {"easy": 0.8644, "medium": 0.7975, "hard": 0.7799}
```

*A NOP agent (drones stationary, pump=0) scores ≈ 0.25 on all tasks. All three tasks are cleared well within their step budgets. The easy task clears in 2 steps: the single drone transits from (0,8) to (2,7) in 1 step, then pumps; the fire at (3,7) falls within the Gaussian cardinal footprint (weight 0.5), extinguishing it immediately. Medium and hard require multi-drone coordination across fire zones in the lower grid. Total runtime: ~8 min for all 3 tasks — well within the 20-minute cap.*

### Hard task — fire dynamics (why it genuinely challenges frontier models)

Starting from 8 seeds, the CA fire spread races ahead of the drones during the 5-step approach phase:

| Step | Active fires | Event |
|------|-------------|-------|
| 0 | 8 | Reset — fires seeded across all quadrants |
| 1 | 14 | CA spread begins; drones still in transit (MAX_SPEED=2) |
| 2 | 22 | Fires grow 75% — **drones still have not fired a single drop** |
| 3 | 29 | Peak growth; 3 drones converge on centre cluster |
| 4 | 33 | **4.1× initial fire count** — 5.3% of 25×25 grid burning |
| 5 | 31 | First pumps at steps 5&7; D1+D2 suppress central cluster |
| 7 | 29 | Two coordinated drops; GE packet loss degrades 40% of comms |
| 10 | 30 | D0 pumps at (14,13); D1+D2 refuelling (both at corner 0,0) |
| 19 | 20 | Coordinated quad-sector sweep begins |
| 33 | 4 | Final phase; D1 refuels while fires relight in periphery |
| 35 | **0** | All fires out — 35/70 steps used; avg battery 38% |

Key difficulty factors that separate this from a trivial task:
- **CA spread races ahead**: fire count **4×** before any drone can pump
- **Refuelling bottleneck**: drones carry only 10 kg retardant; multi-hop reload required while fires relapse
- **GE communication degradation**: ~40% of BEST_EFFORT broadcasts lost; drones must act on stale peer telemetry
- **Kinematic constraint**: MAX_SPEED=2 cells/step means no teleportation to distant fires
- **Friendly-fire lock**: two drones pumping the same cell block each other's extinguishment

---

## Setup & Usage

### Prerequisites

- Python ≥ 3.10
- Docker (for containerised deployment)

### Local installation

```bash
git clone https://github.com/Likhith-BlueLotus/fire-swarm-simulator.git
cd fire_swarm_simulator

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Verify it's up:

```bash
curl http://localhost:7860/health
```

### Run inference (all 3 tasks)

The inference script uses three environment variables and targets any OpenAI-compatible API endpoint, including Hugging Face Inference Providers:

```bash
# ── OpenAI ──────────────────────────────────────────────────────────────
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="<openai-api-key>"

# ── Hugging Face Inference Providers (Nemotron, Llama, Qwen …) ──────────
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
export HF_TOKEN="<hf-token>"

export OPENENV_ENDPOINT="http://localhost:7860"   # or your HF Space URL

python inference.py
```

### Supported models (any OpenAI-compatible provider)

| Key | Provider | Tested |
|---|---|---|
| `gpt-4o-mini` | OpenAI | ✅ (baseline, 0.8139 mean) |
| `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | HF Inference Providers | ✅ compatible |
| `meta-llama/Llama-3.3-70B-Instruct` | HF Inference Providers | ✅ compatible |
| `Qwen/Qwen2.5-72B-Instruct` | HF Inference Providers | ✅ compatible |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | HF Inference Providers | ✅ compatible |

Any model accessible via an OpenAI-compatible chat-completions endpoint works — just change `API_BASE_URL` and `MODEL_NAME`.

### Docker

```bash
# Build (run from fire_swarm_simulator/ — the repo root)
docker build -t fire-swarm .

# Run (maps HF Spaces port 7860)
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=<your-api-key> \
  fire-swarm

# Verify
curl http://localhost:7860/health
```

### Validate submission (full automated gate check)

```bash
# Runs the official 3-step gate check (HF Space ping, docker build, openenv validate)
chmod +x validate-submission.sh
./validate-submission.sh https://le0atis-fire-swarm-simulator.hf.space .

# Expected output:
# [PASSED] -- HF Space is live and responds to /reset
# [PASSED] -- Docker build succeeded
# [PASSED] -- openenv validate passed
# ========================================
#   All 3/3 checks passed!
#   Your submission is ready to submit.
# ========================================
```

### Run the test suite

```bash
# 183 tests: physics engine, reward shaping, API endpoints, async client
python -m pytest tests/ -q
# → 183 passed in 0.82s
```

---

## API Reference


| Endpoint        | Method    | Description                                                                 |
| --------------- | --------- | --------------------------------------------------------------------------- |
| `/health`       | GET       | Readiness probe — returns `{"status":"healthy", ...}`                       |
| `/reset`        | POST      | Start new episode. Body: `{"task": "easy"|"medium"|"hard"}`                 |
| `/step`         | POST      | Advance one tick. Body: `{"action": {...}, "session_id": "..."}`            |
| `/state`        | GET       | Current `SwarmState` (episode_id, step_count, fire counts, drone positions) |
| `/tasks`        | GET       | List all 3 graded tasks with difficulty metadata                            |
| `/grade/{task}` | POST      | Run programmatic grader; returns score vs. NOP baseline                     |
| `/schema`       | GET       | Action/observation JSON schemas                                             |
| `/ws`           | WebSocket | High-frequency real-time agents                                             |
| `/docs`         | GET       | Interactive Swagger UI                                                      |


---

## Project Structure

```
fire_swarm_simulator/           ← repo root (this directory is uploaded to HF Spaces)
├── Dockerfile                  # Container build — HF Spaces & validator pick this up
├── .env.example                # Environment variable template (copy to .env locally)
├── LICENSE                     # BSD-3-Clause
├── README.md                   # This file
├── inference.py                # Baseline inference script (hackathon spec requirement)
├── openenv.yaml                # OpenEnv manifest — tasks, hardware tier, env vars
├── requirements.txt            # Python dependencies
├── uv.lock                     # Locked dependency versions for reproducible installs
├── healthcheck.py              # Docker HEALTHCHECK script (polls /health)
├── models.py                   # Pydantic types: SwarmAction, SwarmObservation, SwarmState
├── client.py                   # Async OpenEnv client: FireSwarmEnv
├── server/
│   ├── app.py                  # FastAPI entrypoint + /grade programmatic grader
│   └── environment.py          # FireSwarmEnvironment physics engine (CA + GE + OU)
└── tests/
    ├── conftest.py             # Shared pytest fixtures
    ├── test_models.py          # Pydantic model validation tests
    ├── test_environment.py     # CA physics, reward function, grader tests
    ├── test_api.py             # FastAPI endpoint integration tests
    └── test_client.py          # Async FireSwarmEnv client tests
```

---

## Environment Variables


| Variable           | Required | Description                                             |
| ------------------ | -------- | ------------------------------------------------------- |
| `API_BASE_URL`     | Yes      | OpenAI-compatible LLM base URL                          |
| `MODEL_NAME`       | Yes      | Model identifier string                                 |
| `HF_TOKEN`         | Yes      | Bearer token for LLM endpoint                           |
| `OPENENV_ENDPOINT` | No       | FireSwarm server URL (default: `http://localhost:7860`) |
| `WORKERS`          | No       | Uvicorn worker count (default: 1; set 2 on cpu-upgrade) |


---

## OpenEnv Compliance

- ✅ `openenv.yaml` with `spec_version`, `name`, `app`, `port`, `hardware_tier`, full `tasks` block with `grader_formula` for each task
- ✅ Typed `Action`, `Observation`, `State` Pydantic models inheriting from OpenEnv base classes
- ✅ `step()` / `reset()` / `state` property on `FireSwarmEnvironment`
- ✅ `SUPPORTS_CONCURRENT_SESSIONS = True`
- ✅ `ConcurrencyConfig(max_concurrent_envs=4, session_timeout=300)`
- ✅ Rewards normalised to `[0.0, 1.0]`
- ✅ `Dockerfile` at repo root — `docker build -t fire-swarm .` works from root
- ✅ Docker `HEALTHCHECK` with `/health` readiness probe
- ✅ `inference.py` at repo root using `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- ✅ 3 tasks (`easy`, `medium`, `hard`) with programmatic graders scored against NOP baseline
- ✅ Grader scores reflect genuine agent performance (not exploitable with NOP actions)
- ✅ Anti-exploit measures: loitering penalty, friendly-fire block, NOP-baseline comparison
- ✅ **183 tests passing** — `pytest tests/` with 100% coverage of physics, rewards, API, and client
- ✅ Deterministic and reproducible: isolated `np.random.default_rng(seed)` per episode instance

---

## Why FireSwarm for the OpenEnv Ecosystem

| Property | FireSwarm | Typical visual simulator |
|---|---|---|
| Agent count | 1 / 3 / 5 (MARL) | Single agent |
| Reward signal | Dense, shaped, [0,1] | Sparse or terminal-only |
| Hardware | **CPU-only** (cpu-upgrade tier) | GPU required (T4+) |
| Evaluation cost | ~8 min / 3 tasks | 30–90 min setup + GPU hours |
| Reproducibility | Seeded RNG, deterministic | Rendering non-determinism |
| Physics | Original CA+GE+OU — novel | CARLA/Unity engine |
| Originality | **Fully original** implementation | Adapter over existing simulator |
| Test suite | **183 tests** | Variable |
| openenv.yaml | **109 lines** with task specs | 5–7 lines (minimal) |

FireSwarm is purpose-built for the OpenEnv goal: an environment where RL agents and LLM agents can be trained, evaluated, and compared on a task with real-world transfer value, without requiring specialised hardware.

---

## Citation

```bibtex
@misc{fireswarm2026,
  author       = {Likhith M},
  title        = {FireSwarm: Decentralised Firefighting UAV Swarm Environment for OpenEnv},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/spaces/Le0AtiS/fire-swarm-simulator}},
  note         = {OpenEnv-compatible MARL environment with Cellular Automata fire spread,
                  Gilbert-Elliott DDS communication model, and Ornstein-Uhlenbeck wind evolution.}
}
```

---

## License

BSD-3-Clause. See `LICENSE` for details.