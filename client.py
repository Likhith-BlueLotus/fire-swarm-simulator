"""
Async OpenEnv client for the FireSwarm environment.

Converts between the OpenEnv HTTP/WebSocket wire format (raw JSON dicts)
and the typed Pydantic models defined in models.py.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import DDSDataSpace, SwarmAction, SwarmObservation, SwarmState
except ImportError:
    from models import DDSDataSpace, SwarmAction, SwarmObservation, SwarmState  # type: ignore[no-redef]


class FireSwarmEnv(EnvClient[SwarmAction, SwarmObservation, SwarmState]):
    """
    Typed OpenEnv client for FireSwarm.

    Wraps EnvClient's HTTP/WebSocket machinery and provides two overrides:
      _step_payload  — serialise SwarmAction → wire dict
      _parse_result  — deserialise wire dict → StepResult[SwarmObservation]
      _parse_state   — deserialise wire dict → SwarmState
    """

    def _step_payload(self, action: SwarmAction) -> dict:
        return {"node_actions": [a.model_dump() for a in action.node_actions]}

    def _parse_result(self, payload: dict) -> StepResult:
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
        return SwarmState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            total_burned_area=int(payload.get("total_burned_area", 0)),
            active_fires=int(payload.get("active_fires", 0)),
            global_wind_vector=tuple(payload.get("global_wind_vector", [0.0, 0.0])),
            payload_levels=payload.get("payload_levels", {}),
            drone_positions=payload.get("drone_positions", {}),
        )
