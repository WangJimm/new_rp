from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


TRAINING_STATE_PICKLE_MAGIC = b"RP_TRAINING_STATE_PICKLE_V1\n"


def load_state_payload(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        prefix = f.read(len(TRAINING_STATE_PICKLE_MAGIC))
        if prefix == TRAINING_STATE_PICKLE_MAGIC:
            return pickle.load(f)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This state is not the refactored pickle format, so torch is required to read it."
        ) from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_policy_weights(payload: dict[str, Any]) -> Any:
    agent_state = payload.get("agent_state")
    if isinstance(agent_state, dict):
        if "policy_weights" in agent_state:
            return agent_state["policy_weights"]
        nested = agent_state.get("agent_state")
        if isinstance(nested, dict) and "policy_weights" in nested:
            return nested["policy_weights"]
    if "policy_weights" in payload:
        return payload["policy_weights"]
    raise KeyError("No PPO policy_weights found in the input state.")


def save_warmstart(path: Path, policy_weights: Any, source: Path, source_episode: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to write the warm-start checkpoint.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": 0,
            "agent_type": "ppo_policy_warm_start",
            "policy_weights": policy_weights,
            "source_state": str(source),
            "source_episode": source_episode,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PPO policy weights from a training_state_latest.pt file for warm-start training."
    )
    parser.add_argument("--input", required=True, type=Path, help="Old PPO training_state_latest.pt")
    parser.add_argument("--output", required=True, type=Path, help="Output warm-start .pt checkpoint")
    args = parser.parse_args()

    payload = load_state_payload(args.input)
    policy_weights = extract_policy_weights(payload)
    save_warmstart(args.output, policy_weights, args.input, payload.get("episode"))
    print(f"saved PPO policy warm-start checkpoint: {args.output}")


if __name__ == "__main__":
    main()
