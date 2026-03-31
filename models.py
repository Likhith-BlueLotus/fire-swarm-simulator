"""
Pydantic data contracts for the FireSwarm MARL environment.

Three top-level schemas implement the OpenEnv typed interface:
  SwarmAction      — per-tick vectorised action for the full drone fleet
  SwarmObservation — decentralised, G-E-filtered observation per step
  SwarmState       — global ground-truth used by graders and replay loggers
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from openenv.core.env_server import Action, Observation, State

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class QoSProfile(str, Enum):
    """
    DDS Quality-of-Service profile.

    Controls which Gilbert-Elliott transition matrix governs packet delivery
    for a drone's broadcast_message this tick.

    BEST_EFFORT  P(G→B)=0.10, P(B→G)=0.40  — low overhead, lossy
    RELIABLE     P(G→B)=0.02, P(B→G)=0.80  — near-lossless, ARQ retx cost
    """
    BEST_EFFORT = "BEST_EFFORT"
    RELIABLE    = "RELIABLE"


class DroneNodeAction(BaseModel):
    """Single-drone action submitted as part of a vectorised SwarmAction."""

    agent_id: str = Field(
        ...,
        min_length=1,
        description="Drone identifier, e.g. 'D0'. Must match an active drone in the episode.",
    )
    target_waypoint: Tuple[float, float, float] = Field(
        ...,
        description=(
            "Destination (x, y, altitude_m) in discrete sector coordinates. "
            "x and y are clipped to [0, grid_size-1]. altitude < 5 m triggers docking."
        ),
    )
    pump_activation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Retardant throttle. Values > 0.5 trigger a Gaussian payload drop.",
    )
    broadcast_message: Optional[str] = Field(
        None,
        max_length=256,
        description="Gossip payload routed to peers via the DDS Good-channel filter.",
    )
    qos_profile: QoSProfile = Field(
        ...,
        description="DDS QoS contract governing this broadcast.",
    )


class DDSDataSpace(BaseModel):
    """
    Simulated DDS shared-memory snapshot for one tick.

    peer_telemetry is sparse: drones whose channel is in Bad state and using
    BEST_EFFORT QoS are silently absent. RELIABLE drones appear with a 'retx'
    flag indicating stale, retransmitted data.
    """

    active_peers: List[str] = Field(
        default_factory=list,
        description="IDs of all living drones this episode.",
    )
    peer_telemetry: Dict[str, dict] = Field(
        default_factory=dict,
        description="G-E-filtered per-drone telemetry snapshots.",
    )
    gossip_messages: Dict[str, str] = Field(
        default_factory=dict,
        description="Broadcast payloads delivered this tick, keyed by sender drone_id.",
    )


class SwarmAction(Action):
    """Vectorised action — one DroneNodeAction per active drone per step."""

    node_actions: List[DroneNodeAction] = Field(
        ...,
        description="Ordered list of per-drone actions for this tick.",
    )


class SwarmObservation(Observation):
    """
    Decentralised observation returned after each step.

    Inherits done (bool) and reward (float) from the OpenEnv Observation base.
    reward is re-declared here with a [0, 1] constraint to enforce the
    anti-hacking normalisation guarantee at the schema layer.
    """

    local_grid_thermal: List[List[float]] = Field(
        default_factory=list,
        description=(
            "15×15 egocentric thermal grid for the representative drone (D0). "
            "Cell values: 0=EMPTY 1=FUEL 2=FIRE 3=EXTINGUISHED 4=SCAR."
        ),
    )
    per_drone_grids: Dict[str, List[List[float]]] = Field(
        default_factory=dict,
        description="Individual 15×15 circular-FOV grids keyed by drone_id.",
    )
    drone_telemetry: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Fleet-wide scalar diagnostics: avg_battery, min_battery, "
            "avg_payload, min_payload, active_fires, burned_area, alive_drones."
        ),
    )
    neighbor_states: List[dict] = Field(
        default_factory=list,
        description="Peer snapshots (id, pos, battery, payload, channel) after G-E filtering.",
    )
    dds_global_space: DDSDataSpace = Field(
        default_factory=DDSDataSpace,
        description="Full DDS state including telemetry and gossip for this tick.",
    )
    wind_vector: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Current wind state (magnitude m/s, bearing degrees).",
    )
    reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Step reward normalised to [0.0, 1.0].",
    )


class SwarmState(State):
    """
    Global ground-truth snapshot.

    Used by the programmatic grader, replay logger, and the inference script's
    /state polling loop. Agents do not receive this during rollouts.

    Inherits episode_id (str, UUIDv4) and step_count (int, ge=0) from State.
    """

    total_burned_area: int = Field(
        default=0,
        ge=0,
        description="Cumulative count of BURNT_SCAR cells.",
    )
    active_fires: int = Field(
        default=0,
        ge=0,
        description="Current ACTIVE_FIRE cell count.",
    )
    global_wind_vector: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Ground-truth wind (magnitude m/s, bearing degrees).",
    )
    payload_levels: Dict[str, float] = Field(
        default_factory=dict,
        description="Remaining retardant payload (kg) per drone_id.",
    )
    drone_positions: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Current [x, y, altitude_m] per drone_id.",
    )

    @field_validator("episode_id", mode="before")
    @classmethod
    def _validate_uuid4(cls, v: str) -> str:
        if v is not None and not _UUID4_RE.match(str(v)):
            raise ValueError(f"episode_id must be UUIDv4; got {v!r}")
        return str(v)
