"""Structured actor-critic models for the lightweight MaCA MAPPO lane."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MAPPOModelConfig:
    local_obs_dim: int
    global_state_dim: int
    num_agents: int
    hidden_size: int = 256
    role_embed_dim: int = 8
    course_dim: int = 16
    attack_dim: int = 21


def _mlp(input_dim: int, hidden_size: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim),
    )


class TeamActorCritic(nn.Module):
    def __init__(self, cfg: MAPPOModelConfig):
        super().__init__()
        self.cfg = cfg
        self.role_embedding = nn.Embedding(cfg.num_agents, cfg.role_embed_dim)
        self.actor_backbone = nn.Sequential(
            nn.Linear(cfg.local_obs_dim + cfg.role_embed_dim, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
        )
        self.actor_rnn = nn.GRUCell(cfg.hidden_size, cfg.hidden_size)
        self.course_head = nn.Linear(cfg.hidden_size, cfg.course_dim)
        self.attack_head = nn.Linear(cfg.hidden_size, cfg.attack_dim)
        self.critic = _mlp(cfg.global_state_dim, cfg.hidden_size, 1)

    @property
    def actor_hidden_dim(self) -> int:
        return int(self.cfg.hidden_size)

    def actor_step(self, local_obs: torch.Tensor, agent_ids: torch.Tensor, actor_h: torch.Tensor):
        role_embed = self.role_embedding(agent_ids)
        x = torch.cat([local_obs, role_embed], dim=-1)
        x = self.actor_backbone(x)
        next_h = self.actor_rnn(x, actor_h)
        return self.course_head(next_h), self.attack_head(next_h), next_h

    def actor(self, local_obs: torch.Tensor, agent_ids: torch.Tensor):
        actor_h = torch.zeros(
            (local_obs.shape[0], self.actor_hidden_dim),
            dtype=local_obs.dtype,
            device=local_obs.device,
        )
        course_logits, attack_logits, _ = self.actor_step(local_obs, agent_ids, actor_h)
        return course_logits, attack_logits

    def value(self, global_state: torch.Tensor):
        return self.critic(global_state).squeeze(-1)
