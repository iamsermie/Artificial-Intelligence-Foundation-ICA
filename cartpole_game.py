"""
cartpole_game.py

A small, runnable CartPole demo that supports three modes:
- auto: a simple reflex heuristic using pole angle
- random: random actions
- render-only: steps through environment without actions (uses random)

The script tries to import `gymnasium` first, then falls back to `gym` for compatibility.

Usage:
    python cartpole_game.py --mode auto --episodes 5

"""
import argparse
import time
import numpy as np

# Try gymnasium then gym for compatibility
try:
    import gymnasium as gym
    GYMNAME = "gymnasium"
except Exception:
    import gym
    GYMNAME = "gym"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["auto", "random", "render-only"], default="auto")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def auto_policy(obs):
    """Very simple heuristic: use pole angle (obs[2]) to decide action.
    If the pole is leaning right (angle > 0) push right (action 1), else push left (action 0).
    This is not a learned policy but often keeps the pole upright for some time.
    """
    angle = float(obs[2])
    return 1 if angle > 0 else 0


def run_episode(env, mode, seed=None, max_steps=1000):
    # Reset handling for gymnasium vs gym
    try:
        obs, info = env.reset(seed=seed)
    except Exception:
        obs = env.reset()
        info = {}

    total_reward = 0.0
    steps = 0

    while True:
        if mode == "random":
            action = env.action_space.sample()
        elif mode == "auto":
            action = auto_policy(obs)
        else:  # render-only fallback to random actions
            action = env.action_space.sample()

        # Step and handle gym vs gymnasium return signatures
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        total_reward += reward
        steps += 1

        # render (human)
        try:
            env.render()
        except Exception:
            pass

        time.sleep(0.01)

        if done or steps >= max_steps:
            break

    return total_reward, steps


def main():
    args = parse_args()

    # Create environment. Use render_mode='human' where supported.
    env_kwargs = {}
    try:
        env = gym.make("CartPole-v1", render_mode="human")
    except Exception:
        # older gym versions accept render_mode at step/render time
        env = gym.make("CartPole-v1")

    print(f"Using {GYMNAME}, mode={args.mode}, episodes={args.episodes}")

    for ep in range(1, args.episodes + 1):
        r, s = run_episode(env, args.mode, seed=(args.seed + ep))
        print(f"Episode {ep}: reward={r:.2f}, steps={s}")

    env.close()


if __name__ == "__main__":
    main()
