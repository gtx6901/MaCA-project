#!/usr/bin/env python
"""Behavior cloning warm start for MaCA MAPPO actor."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from marl_env.mappo_model import MAPPOModelConfig, TeamActorCritic


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="mappo_maca_bc_warmstart")
    parser.add_argument("--train_dir", type=str, default="train_dir/mappo")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--role_embed_dim", type=int, default=8)
    parser.add_argument("--course_embed_dim", type=int, default=16)

    parser.add_argument("--save_actor_only", type=str2bool, default=True)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device_arg: str):
    if device_arg == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def flatten_dataset(data: dict):
    local_obs = np.asarray(data["local_obs"], dtype=np.float32)
    agent_ids = np.asarray(data["agent_ids"], dtype=np.int64)
    attack_masks = np.asarray(data["attack_masks"], dtype=np.bool_)
    alive_mask = np.asarray(data["alive_mask"], dtype=np.float32)
    course_action = np.asarray(data["course_action"], dtype=np.int64)
    attack_action = np.asarray(data["attack_action"], dtype=np.int64)

    t, a, d = local_obs.shape
    flat_local_obs = local_obs.reshape(t * a, d)
    flat_agent_ids = agent_ids.reshape(t * a)
    flat_attack_masks = attack_masks.reshape(t * a, attack_masks.shape[-1])
    flat_alive_mask = alive_mask.reshape(t * a)
    flat_course_action = course_action.reshape(t * a)
    flat_attack_action = attack_action.reshape(t * a)

    active = flat_alive_mask > 0.5
    return {
        "local_obs": flat_local_obs[active],
        "agent_ids": flat_agent_ids[active],
        "attack_masks": flat_attack_masks[active],
        "course_action": flat_course_action[active],
        "attack_action": flat_attack_action[active],
        "num_agents": int(np.max(agent_ids) + 1),
        "local_obs_dim": int(d),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset not found: %s" % dataset_path)

    raw_data = dict(np.load(str(dataset_path), allow_pickle=False))
    data = flatten_dataset(raw_data)
    num_samples = int(data["local_obs"].shape[0])
    if num_samples <= 0:
        raise RuntimeError("No alive samples in dataset")

    model = TeamActorCritic(
        MAPPOModelConfig(
            local_obs_dim=int(data["local_obs_dim"]),
            global_state_dim=1,
            num_agents=int(data["num_agents"]),
            hidden_size=args.hidden_size,
            role_embed_dim=args.role_embed_dim,
            course_embed_dim=args.course_embed_dim,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    local_obs_t = torch.as_tensor(data["local_obs"], dtype=torch.float32, device=device)
    agent_ids_t = torch.as_tensor(data["agent_ids"], dtype=torch.long, device=device)
    attack_masks_t = torch.as_tensor(data["attack_masks"], dtype=torch.bool, device=device)
    course_t = torch.as_tensor(data["course_action"], dtype=torch.long, device=device)
    attack_t = torch.as_tensor(data["attack_action"], dtype=torch.long, device=device)

    actor_h_zeros = torch.zeros((num_samples, model.actor_hidden_dim), dtype=torch.float32, device=device)

    steps_per_epoch = max(1, int(np.ceil(float(num_samples) / float(max(args.batch_size, 1)))))
    start_time = time.time()

    for epoch in range(args.epochs):
        order = np.random.permutation(num_samples)
        epoch_loss = 0.0
        epoch_course_loss = 0.0
        epoch_attack_loss = 0.0

        for step_idx in range(steps_per_epoch):
            left = step_idx * args.batch_size
            right = min(num_samples, left + args.batch_size)
            batch_idx = order[left:right]
            if batch_idx.size == 0:
                continue

            obs_b = local_obs_t[batch_idx]
            ids_b = agent_ids_t[batch_idx]
            masks_b = attack_masks_t[batch_idx]
            course_b = course_t[batch_idx]
            attack_b = attack_t[batch_idx]
            h_b = actor_h_zeros[batch_idx]

            course_logits, _attack_logits_unused, next_h = model.actor_step(obs_b, ids_b, h_b, course_actions=course_b)
            attack_logits = model.attack_logits(next_h, course_b)
            invalid_logit = torch.finfo(attack_logits.dtype).min
            attack_logits = attack_logits.masked_fill(~masks_b, invalid_logit)

            course_loss = F.cross_entropy(course_logits, course_b)
            attack_loss = F.cross_entropy(attack_logits, attack_b)
            loss = course_loss + attack_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_course_loss += float(course_loss.item())
            epoch_attack_loss += float(attack_loss.item())

        denom = float(max(steps_per_epoch, 1))
        print(
            "[bc] epoch=%d/%d loss=%.4f course=%.4f attack=%.4f"
            % (
                epoch + 1,
                args.epochs,
                epoch_loss / denom,
                epoch_course_loss / denom,
                epoch_attack_loss / denom,
            ),
            flush=True,
        )

    exp_dir = Path(args.train_dir) / args.experiment
    ckpt_dir = exp_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_state = model.state_dict()
    if args.save_actor_only:
        model_state = {k: v for k, v in model_state.items() if not k.startswith("critic.")}

    checkpoint_path = ckpt_dir / "checkpoint_000000000_0.pt"
    torch.save(
        {
            "model": model_state,
            "env_steps": 0,
            "update_idx": 0,
            "bc_epochs": int(args.epochs),
            "dataset_path": str(dataset_path),
        },
        checkpoint_path,
    )

    cfg = {
        "experiment": args.experiment,
        "train_dir": args.train_dir,
        "hidden_size": int(args.hidden_size),
        "role_embed_dim": int(args.role_embed_dim),
        "course_embed_dim": int(args.course_embed_dim),
        "local_obs_dim": int(data["local_obs_dim"]),
        "num_agents": int(data["num_agents"]),
        "bc_warmstart": True,
        "bc_dataset_path": str(dataset_path),
        "bc_wall_time_sec": float(time.time() - start_time),
    }
    (exp_dir / "cfg.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "samples": num_samples,
                "epochs": int(args.epochs),
                "actor_only": bool(args.save_actor_only),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
