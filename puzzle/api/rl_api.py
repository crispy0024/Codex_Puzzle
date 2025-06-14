import json
import os
import logging
from fastapi import APIRouter, BackgroundTasks
import numpy as np
try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - optional dependency
    PPO = None

from ..rl_env import PuzzleEnv

router = APIRouter()

USE_RL_MODEL = os.environ.get("USE_RL_MODEL", "1") == "1"
_policy = None
_rl_env = None
_state_to_idx: dict[str, int] = {}

logger = logging.getLogger(__name__)


def _load_rl_policy():
    global _policy, _rl_env, _state_to_idx
    if not USE_RL_MODEL:
        logger.info("RL policy disabled via USE_RL_MODEL=0")
        return
    if PPO is None:
        logger.info("stable-baselines3 not installed, RL policy disabled")
        return
    model_path = "rl_model.zip"
    if not os.path.exists(model_path):
        logger.info("RL model %s not found", model_path)
        return
    try:
        feedback: list[dict] = []
        if os.path.exists("feedback.jsonl"):
            with open("feedback.jsonl") as f:
                feedback = [json.loads(line) for line in f if line.strip()]
        if feedback:
            _rl_env = PuzzleEnv(feedback)
            _state_to_idx = {s: i for i, s in enumerate(_rl_env._states)}
            _policy = PPO.load("rl_model", env=_rl_env)
        else:
            _policy = PPO.load("rl_model")
        logger.info("Loaded RL model from %s", model_path)
    except Exception:
        logger.exception("Failed to load RL model")
        _policy = None
        _rl_env = None


def rank_with_policy(state: dict, candidates: list[dict]) -> list[dict]:
    if _policy is None or _rl_env is None:
        return candidates
    key = json.dumps(state, sort_keys=True)
    idx = _state_to_idx.get(key)
    if idx is None:
        logger.info("No policy state for %s", key)
        return candidates
    try:
        action_idx, _ = _policy.predict(np.array([idx]), deterministic=True)
        if isinstance(action_idx, np.ndarray):
            action_idx = int(action_idx[0])
        if 0 <= action_idx < len(_rl_env._actions):
            chosen = _rl_env._actions[action_idx]
            logger.info("Policy chose action %s", chosen)
            for i, cand in enumerate(candidates):
                cand_id = {
                    "piece_id": cand["piece_id"],
                    "edge_index": cand["edge_index"],
                }
                if cand_id == chosen:
                    if i != 0:
                        candidates = [cand] + candidates[:i] + candidates[i + 1 :]
                    break
    except Exception:
        logger.exception("Policy ranking failed")
    return candidates


_load_rl_policy()

# Background RL training
try:
    from .. import train_rl
except Exception:  # pragma: no cover - optional dependency
    train_rl = None


def _train_model(timesteps: int = 1000):
    if train_rl is None:
        logger.info("RL training module not available")
        return
    try:
        train_rl.main(total_timesteps=timesteps)
    except Exception:
        logger.exception("RL training failed")


@router.post("/train_rl")
def train_rl_endpoint(background: BackgroundTasks, timesteps: int = 1000):
    """Start RL training in the background."""
    background.add_task(_train_model, timesteps)
    return {"status": "training"}
