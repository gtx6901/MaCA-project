#!/usr/bin/env python
"""Behavior cloning warm start for the MaCA MAPPO actor."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = ROOT_DIR / "environment"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

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
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--chunk_len", type=int, default=32)
    parser.add_argument("--burn_in", type=int, default=8)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--role_embed_dim", type=int, default=8)
    parser.add_argument("--course_embed_dim", type=int, default=16)

    parser.add_argument("--bc_stage", type=str, default="both", choices=["both", "mode", "action"])
    parser.add_argument("--mode_loss_weight", type=float, default=1.0)

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


def load_dataset(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset not found: %s" % dataset_path)

    data = dict(np.load(str(dataset_path), allow_pickle=False))
    required = [
        "local_obs",
        "agent_ids",
        "attack_masks",
        "alive_mask",
        "course_action",
        "attack_action",
    ]
    for key in required:
        if key not in data:
            raise KeyError("Dataset missing key: %s" % key)
    return data


def build_episode_offsets(episode_lengths: np.ndarray, total_steps: int):
    offsets = []
    cursor = 0
    for length in episode_lengths.tolist():
        length = int(length)
        if length <= 0:
            continue
        end = min(total_steps, cursor + length)
        if end > cursor:
            offsets.append((cursor, end))
        cursor = end
        if cursor >= total_steps:
            break
    if not offsets:
        offsets.append((0, total_steps))
    return offsets


def build_chunks(episode_offsets, num_agents: int, chunk_len: int):
    chunks = []
    for ep_start, ep_end in episode_offsets:
        for agent_idx in range(num_agents):
            start = ep_start
            while start < ep_end:
                end = min(ep_end, start + chunk_len)
                chunks.append((agent_idx, start, end, ep_start))
                start = end
    return chunks


def pack_chunk_batch(
    dataset,
    chunk_indices,
    chunks,
    burn_in: int,
    device: torch.device,
):
    local_obs = dataset["local_obs"]
    agent_ids = dataset["agent_ids"]
    attack_masks = dataset["attack_masks"]
    alive_mask = dataset["alive_mask"]
    course_action = dataset["course_action"]
    attack_action = dataset["attack_action"]
    mode_label = dataset.get("mode_label", None)

    batch_size = len(chunk_indices)
    hidden_dim = 0
    obs_dim = local_obs.shape[-1]
    attack_dim = attack_masks.shape[-1]
    max_total_len = 0
    meta = []

    for chunk_idx in chunk_indices:
        agent_idx, start, end, ep_start = chunks[int(chunk_idx)]
        burn_start = max(ep_start, start - burn_in)
        total_len = end - burn_start
        train_offset = start - burn_start
        train_len = end - start
        meta.append((agent_idx, start, end, burn_start, total_len, train_offset, train_len))
        max_total_len = max(max_total_len, total_len)

    local_batch = torch.zeros((max_total_len, batch_size, obs_dim), dtype=torch.float32, device=device)
    agent_id_batch = torch.zeros((max_total_len, batch_size), dtype=torch.long, device=device)
    attack_mask_batch = torch.zeros((max_total_len, batch_size, attack_dim), dtype=torch.bool, device=device)
    course_batch = torch.zeros((max_total_len, batch_size), dtype=torch.long, device=device)
    attack_batch = torch.zeros((max_total_len, batch_size), dtype=torch.long, device=device)
    mode_batch = torch.zeros((max_total_len, batch_size), dtype=torch.long, device=device)
    train_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=device)
    active_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=device)
    seq_valid_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=device)

    for batch_col, chunk_meta in enumerate(meta):
        agent_idx, start, end, burn_start, total_len, train_offset, train_len = chunk_meta
        seq_valid_mask[:total_len, batch_col] = True
        local_batch[:total_len, batch_col] = torch.as_tensor(
            local_obs[burn_start:end, agent_idx],
            dtype=torch.float32,
            device=device,
        )
        agent_id_batch[:total_len, batch_col] = torch.as_tensor(
            agent_ids[burn_start:end, agent_idx],
            dtype=torch.long,
            device=device,
        )
        attack_mask_batch[:total_len, batch_col] = torch.as_tensor(
            attack_masks[burn_start:end, agent_idx],
            dtype=torch.bool,
            device=device,
        )
        course_batch[:total_len, batch_col] = torch.as_tensor(
            course_action[burn_start:end, agent_idx],
            dtype=torch.long,
            device=device,
        )
        attack_batch[:total_len, batch_col] = torch.as_tensor(
            attack_action[burn_start:end, agent_idx],
            dtype=torch.long,
            device=device,
        )
        if mode_label is not None:
            mode_batch[:total_len, batch_col] = torch.as_tensor(
                mode_label[burn_start:end, agent_idx],
                dtype=torch.long,
                device=device,
            )
        train_slice = slice(train_offset, train_offset + train_len)
        train_mask[train_slice, batch_col] = True
        active_mask[train_slice, batch_col] = torch.as_tensor(
            alive_mask[start:end, agent_idx] > 0.5,
            dtype=torch.bool,
            device=device,
        )

    return {
        "local_obs": local_batch,
        "agent_ids": agent_id_batch,
        "attack_masks": attack_mask_batch,
        "course_action": course_batch,
        "attack_action": attack_batch,
        "mode_label": mode_batch,
        "train_mask": train_mask,
        "active_mask": active_mask,
        "seq_valid_mask": seq_valid_mask,
        "hidden_dim": hidden_dim,
    }


def train_epoch(model, optimizer, dataset, chunks, args, device):
    if not chunks:
        raise RuntimeError("No chunks available for BC training")

    order = np.random.permutation(len(chunks))
    batch_size = max(1, int(args.batch_size))
    total_loss = 0.0
    total_course_loss = 0.0
    total_attack_loss = 0.0
    total_mode_loss = 0.0
    total_active = 0
    total_batches = 0

    for left in range(0, len(order), batch_size):
        right = min(len(order), left + batch_size)
        chunk_indices = order[left:right]
        packed = pack_chunk_batch(dataset, chunk_indices, chunks, args.burn_in, device)
        hidden = torch.zeros((len(chunk_indices), model.actor_hidden_dim), dtype=torch.float32, device=device)

        course_losses = []
        attack_losses = []
        mode_losses = []
        max_total_len = packed["local_obs"].shape[0]

        for seq_idx in range(max_total_len):
            seq_valid = packed["seq_valid_mask"][seq_idx]
            if not torch.any(seq_valid):
                continue

            seq_local = packed["local_obs"][seq_idx, seq_valid]
            seq_ids = packed["agent_ids"][seq_idx, seq_valid]
            seq_hidden = hidden[seq_valid]
            seq_course = packed["course_action"][seq_idx, seq_valid]
            seq_attack = packed["attack_action"][seq_idx, seq_valid]
            seq_mode = packed["mode_label"][seq_idx, seq_valid]
            seq_attack_mask = packed["attack_masks"][seq_idx, seq_valid]

            course_logits, attack_logits, next_hidden = model.actor_step(
                seq_local,
                seq_ids,
                seq_hidden,
                course_actions=seq_course,
            )
            hidden = hidden.clone()
            hidden[seq_valid] = next_hidden

            seq_train = packed["train_mask"][seq_idx, seq_valid]
            seq_active = packed["active_mask"][seq_idx, seq_valid]
            use_mask = seq_train & seq_active
            if not torch.any(use_mask):
                continue

            course_logits = course_logits[use_mask]
            attack_logits = attack_logits[use_mask]
            seq_course = seq_course[use_mask]
            seq_attack = seq_attack[use_mask]
            seq_mode = seq_mode[use_mask]
            seq_attack_mask = seq_attack_mask[use_mask]

            invalid_logit = torch.finfo(attack_logits.dtype).min
            attack_logits = attack_logits.masked_fill(~seq_attack_mask, invalid_logit)

            if args.bc_stage in {"both", "action"}:
                course_losses.append(F.cross_entropy(course_logits, seq_course))
                attack_losses.append(F.cross_entropy(attack_logits, seq_attack))
            if args.bc_stage in {"both", "mode"}:
                mode_logits = model.mode_head(next_hidden[use_mask])
                mode_losses.append(F.cross_entropy(mode_logits, seq_mode))
            total_active += int(seq_course.shape[0])

        if not course_losses and not mode_losses:
            continue

        if course_losses:
            course_loss = torch.stack(course_losses).mean()
        else:
            course_loss = torch.tensor(0.0, device=device)

        if attack_losses:
            attack_loss = torch.stack(attack_losses).mean()
        else:
            attack_loss = torch.tensor(0.0, device=device)

        if mode_losses:
            mode_loss = torch.stack(mode_losses).mean()
        else:
            mode_loss = torch.tensor(0.0, device=device)

        if args.bc_stage == "mode":
            loss = args.mode_loss_weight * mode_loss
        elif args.bc_stage == "action":
            loss = course_loss + attack_loss
        else:
            loss = course_loss + attack_loss + args.mode_loss_weight * mode_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_course_loss += float(course_loss.item())
        total_attack_loss += float(attack_loss.item())
        total_mode_loss += float(mode_loss.item())
        total_batches += 1

    denom = float(max(total_batches, 1))
    return {
        "loss": total_loss / denom,
        "course_loss": total_course_loss / denom,
        "attack_loss": total_attack_loss / denom,
        "mode_loss": total_mode_loss / denom,
        "active_samples": total_active,
        "batches": total_batches,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)

    dataset_path = Path(args.dataset_path)
    raw_data = load_dataset(dataset_path)
    local_obs = np.asarray(raw_data["local_obs"], dtype=np.float32)
    agent_ids = np.asarray(raw_data["agent_ids"], dtype=np.int64)
    attack_masks = np.asarray(raw_data["attack_masks"], dtype=np.bool_)
    alive_mask = np.asarray(raw_data["alive_mask"], dtype=np.float32)
    course_action = np.asarray(raw_data["course_action"], dtype=np.int64)
    attack_action = np.asarray(raw_data["attack_action"], dtype=np.int64)
    mode_label = np.asarray(raw_data.get("mode_label", np.zeros_like(attack_action)), dtype=np.int64)
    episode_lengths = np.asarray(raw_data.get("episode_lengths", np.asarray([local_obs.shape[0]], dtype=np.int32)))

    total_steps, num_agents, local_obs_dim = local_obs.shape
    active_samples = int(np.sum(alive_mask > 0.5))
    if total_steps <= 0 or active_samples <= 0:
        raise RuntimeError("Dataset has no usable alive samples")

    dataset = {
        "local_obs": local_obs,
        "agent_ids": agent_ids,
        "attack_masks": attack_masks,
        "alive_mask": alive_mask,
        "course_action": course_action,
        "attack_action": attack_action,
        "mode_label": mode_label,
    }

    episode_offsets = build_episode_offsets(episode_lengths, total_steps)
    chunks = build_chunks(episode_offsets, num_agents, max(1, int(args.chunk_len)))

    model = TeamActorCritic(
        MAPPOModelConfig(
            local_obs_dim=int(local_obs_dim),
            global_state_dim=1,
            num_agents=int(num_agents),
            hidden_size=args.hidden_size,
            role_embed_dim=args.role_embed_dim,
            course_embed_dim=args.course_embed_dim,
            attack_dim=int(attack_masks.shape[-1]),
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_time = time.time()
    for epoch in range(args.epochs):
        stats = train_epoch(model, optimizer, dataset, chunks, args, device)
        print(
            "[bc] epoch=%d/%d stage=%s loss=%.4f mode=%.4f course=%.4f attack=%.4f active=%d batches=%d"
            % (
                epoch + 1,
                args.epochs,
                args.bc_stage,
                stats["loss"],
                stats["mode_loss"],
                stats["course_loss"],
                stats["attack_loss"],
                stats["active_samples"],
                stats["batches"],
            ),
            flush=True,
        )

    exp_dir = Path(args.train_dir) / args.experiment
    ckpt_dir = exp_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_state = model.state_dict()
    if args.save_actor_only:
        model_state = {k: v for k, v in model_state.items() if not k.startswith("critic")}

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
        "local_obs_dim": int(local_obs_dim),
        "num_agents": int(num_agents),
        "bc_warmstart": True,
        "bc_dataset_path": str(dataset_path),
        "bc_wall_time_sec": float(time.time() - start_time),
        "bc_stage": args.bc_stage,
        "bc_chunk_len": int(args.chunk_len),
        "bc_burn_in": int(args.burn_in),
        "bc_total_steps": int(total_steps),
        "bc_active_samples": int(active_samples),
    }
    (exp_dir / "cfg.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "steps": int(total_steps),
                "active_samples": int(active_samples),
                "episodes": int(len(episode_offsets)),
                "chunks": int(len(chunks)),
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
