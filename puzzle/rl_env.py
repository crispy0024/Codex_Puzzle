import json
import random
from typing import Any, Dict, List

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional dependency
    gym = None
    spaces = None


class PuzzleEnv(gym.Env if gym is not None else object):
    """Minimal environment using logged feedback as transitions."""

    metadata = {"render.modes": []}

    def __init__(self, feedback: List[Dict[str, Any]]):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for PuzzleEnv")
        super().__init__()
        self.feedback = feedback

        # Group rewards by serialized state
        self._state_actions: Dict[str, Dict[Any, float]] = {}
        states = []
        for entry in feedback:
            key = json.dumps(entry["state"], sort_keys=True)
            states.append(key)
            acts = self._state_actions.setdefault(key, {})
            acts[entry["action"]] = entry.get("reward", 0.0)

        self._states = sorted(set(states))
        self._actions = sorted({e["action"] for e in feedback})

        self.observation_space = spaces.Discrete(len(self._states))
        self.action_space = spaces.Discrete(len(self._actions))
        self.state_index = 0

    @property
    def current_state(self) -> Dict[str, Any]:
        return json.loads(self._states[self.state_index])

    def available_actions(self) -> List[Any]:
        key = self._states[self.state_index]
        return list(self._state_actions.get(key, {}).keys())

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.state_index = random.randrange(len(self._states))
        observation = self.state_index
        info = {}
        return observation, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        action_value = self._actions[action]
        key = self._states[self.state_index]
        reward = float(self._state_actions.get(key, {}).get(action_value, 0.0))
        terminated = True
        truncated = False
        info: Dict[str, Any] = {}
        # sample next state randomly
        self.state_index = random.randrange(len(self._states))
        observation = self.state_index
        return observation, reward, terminated, truncated, info
