import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from puzzle.rl_env import PuzzleEnv


FEEDBACK_FILE = "feedback.jsonl"
MODEL_FILE = "rl_model"


def load_feedback(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(total_timesteps: int = 1000):
    feedback = load_feedback(FEEDBACK_FILE)
    if not feedback:
        print("No feedback data found.")
        return

    def _make_env():
        return PuzzleEnv(feedback)

    env = DummyVecEnv([_make_env])

    if os.path.exists(MODEL_FILE + ".zip"):
        model = PPO.load(MODEL_FILE, env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}.zip")


if __name__ == "__main__":
    main()
