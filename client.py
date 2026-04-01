"""
OpenEnv client adapter for the FireSwarm environment.

Bridges the OpenEnv HTTP/WebSocket wire protocol (raw JSON dicts) and the
typed Pydantic models defined in models.py. Inherit from the generic
EnvClient base so callers can use the standard `.sync()` context manager
pattern without knowing the transport details.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import DDSDataSpace, SwarmAction, SwarmObservation, SwarmState
except ImportError:
    from models import DDSDataSpace, SwarmAction, SwarmObservation, SwarmState  # type: ignore[no-redef]


class FireSwarmEnv(EnvClient[SwarmAction, SwarmObservation, SwarmState]):
    """
    Typed OpenEnv client for the FireSwarm environment.

    Overrides three EnvClient hooks to convert between the wire format and
    the FireSwarm Pydantic schemas:
      _step_payload  — serialise SwarmAction to the JSON dict the server expects
      _parse_result  — deserialise a /step response into StepResult[SwarmObservation]
      _parse_state   — deserialise a /state response into SwarmState
    """

    def _step_payload(self, action: SwarmAction) -> dict:
        """Serialise a SwarmAction to the wire format expected by POST /step."""
        return {"node_actions": [a.model_dump() for a in action.node_actions]}

    def _parse_result(self, payload: dict) -> StepResult:
        """
        Deserialise a /step response envelope into a typed StepResult.

        The server wraps the observation inside an "observation" key; reward
        and done are also present at the top level of the envelope for
        compatibility with the base EnvClient consumer.
        """
        obs_data = payload.get("observation", {})

        raw_dds = obs_data.get(
            "dds_global_space",
            {"active_peers": [], "peer_telemetry": {}, "gossip_messages": {}},
        )
        dds_space = DDSDataSpace(
            active_peers=raw_dds.get("active_peers", []),
            peer_telemetry=raw_dds.get("peer_telemetry", {}),
            gossip_messages=raw_dds.get("gossip_messages", {}),
        )

        observation = SwarmObservation(
            done=bool(payload.get("done", False)),
            reward=float(payload.get("reward", 0.0)),
            local_grid_thermal=obs_data.get("local_grid_thermal", []),
            per_drone_grids=obs_data.get("per_drone_grids", {}),
            drone_telemetry=obs_data.get("drone_telemetry", {}),
            neighbor_states=obs_data.get("neighbor_states", []),
            dds_global_space=dds_space,
            wind_vector=tuple(obs_data.get("wind_vector", [0.0, 0.0])),
        )

        return StepResult(
            observation=observation,
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> SwarmState:
        """Deserialise a /state response into the ground-truth SwarmState model."""
        return SwarmState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            total_burned_area=int(payload.get("total_burned_area", 0)),
            active_fires=int(payload.get("active_fires", 0)),
            global_wind_vector=tuple(payload.get("global_wind_vector", [0.0, 0.0])),
            payload_levels=payload.get("payload_levels", {}),
            drone_positions=payload.get("drone_positions", {}),
        )
