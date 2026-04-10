#!/usr/bin/env python
"""Smoke test for the new parallel-style MaCA multi-agent wrapper."""

import argparse
import random

from marl_env import MaCAParallelEnv
from marl_env.maca_parallel_env import EnvConfig


def sample_valid_action(mask):
    valid = [idx for idx, flag in enumerate(mask.tolist()) if flag]
    return random.choice(valid) if valid else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--opponent", default="fix_rule")
    parser.add_argument("--max-step", type=int, default=650)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = MaCAParallelEnv(
        EnvConfig(
            opponent=args.opponent,
            max_step=args.max_step,
            render=args.render,
        )
    )
    observations, infos = env.reset(seed=42)
    print("agents:", env.agents)
    print("sample obs keys:", list(observations[env.agents[0]].keys()))
    print("global_state_shape:", infos[env.agents[0]]["global_state"].shape)

    for step_idx in range(args.steps):
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = sample_valid_action(obs["action_mask"])
        observations, rewards, terminations, truncations, infos = env.step(actions)
        sample_agent = env.agents[0]
        print(
            f"step={step_idx + 1} "
            f"reward={rewards[sample_agent]:.1f} "
            f"active={infos[sample_agent]['is_active']} "
            f"terminated={terminations[sample_agent]} "
            f"truncated={truncations[sample_agent]}"
        )
        if any(terminations.values()) or any(truncations.values()):
            break


if __name__ == "__main__":
    main()
