"""Structured actor-critic models for the lightweight MaCA MAPPO lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class MAPPOModelConfig:
    local_obs_dim: int
    local_screen_shape: Tuple[int, int, int]
    global_state_dim: int
    num_agents: int
    hidden_size: int = 256
    mode_dim: int = 4
    mode_embed_dim: int = 8
    course_embed_dim: int = 16
    screen_embed_dim: int = 64
    course_dim: int = 16
    attack_dim: int = 21
    priority_grid_h: int = 4
    priority_grid_w: int = 4
    priority_top_k: int = 2


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
        self.mode_embedding = nn.Embedding(cfg.mode_dim, cfg.mode_embed_dim)

        if len(cfg.local_screen_shape) != 3:
            raise ValueError("local_screen_shape must be (H, W, C)")
        screen_h, screen_w, screen_c = cfg.local_screen_shape
        if min(int(screen_h), int(screen_w), int(screen_c)) <= 0:
            raise ValueError("local_screen_shape entries must be positive")

        self.priority_grid_h = max(2, int(cfg.priority_grid_h))
        self.priority_grid_w = max(2, int(cfg.priority_grid_w))
        self.priority_grid_size = int(self.priority_grid_h * self.priority_grid_w)
        self.priority_top_k = max(1, min(int(cfg.priority_top_k), self.priority_grid_size))
        self.priority_feature_dim = int(self.priority_top_k * 4 + 1)

        centers = []
        for gy in range(self.priority_grid_h):
            for gx in range(self.priority_grid_w):
                centers.append(
                    [
                        float(gx + 0.5) / float(self.priority_grid_w),
                        float(gy + 0.5) / float(self.priority_grid_h),
                    ]
                )
        self.register_buffer("priority_cell_centers", torch.as_tensor(centers, dtype=torch.float32))

        self.screen_backbone = nn.Sequential(
            nn.Conv2d(int(screen_c), 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.screen_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, cfg.screen_embed_dim),
            nn.ReLU(),
        )
        self.priority_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.priority_grid_h, self.priority_grid_w)),
            nn.Flatten(),
            nn.Linear(16 * self.priority_grid_size, self.priority_grid_size),
        )

        self.actor_backbone = nn.Sequential(
            nn.Linear(cfg.local_obs_dim + cfg.screen_embed_dim + self.priority_feature_dim, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
        )
        self.actor_rnn = nn.GRUCell(cfg.hidden_size, cfg.hidden_size)

        # High-level mode policy branch (currently structural; trainer can ignore for now).
        self.mode_head = nn.Linear(cfg.hidden_size, cfg.mode_dim)
        self.worker = nn.Sequential(
            nn.Linear(cfg.hidden_size + cfg.mode_embed_dim, cfg.hidden_size),
            nn.ReLU(),
        )

        self.course_head = nn.Linear(cfg.hidden_size, cfg.course_dim)
        self.course_embedding = nn.Embedding(cfg.course_dim, cfg.course_embed_dim)

        # Split low-level attack into target selection and fire decision.
        self.target_dim = max(1, cfg.attack_dim - 1)
        attack_input_dim = cfg.hidden_size + cfg.course_embed_dim
        self.target_head = nn.Linear(attack_input_dim, self.target_dim)
        self.fire_head = nn.Linear(attack_input_dim, 2)

        # Keep legacy alias for state_dict compatibility and external assumptions.
        self.attack_head = self.target_head
        self.critic = nn.ModuleDict(
            {
                "input_proj": nn.Linear(cfg.global_state_dim + cfg.mode_embed_dim, cfg.hidden_size),
                "rnn": nn.GRU(cfg.hidden_size, cfg.hidden_size, num_layers=1),
                "trunk": nn.Sequential(
                    nn.Linear(cfg.hidden_size, cfg.hidden_size),
                    nn.ReLU(),
                ),
                "value_team": nn.Linear(cfg.hidden_size, 1),
                "value_contact": nn.Linear(cfg.hidden_size, 1),
                "value_opportunity": nn.Linear(cfg.hidden_size, 1),
                "value_survival": nn.Linear(cfg.hidden_size, 1),
            }
        )

    @property
    def actor_hidden_dim(self) -> int:
        return int(self.cfg.hidden_size)

    def _screen_features(self, local_screen: torch.Tensor) -> torch.Tensor:
        # local_screen: [N, H, W, C] uint8/float32
        if local_screen.dim() != 4:
            raise ValueError("Expected local_screen rank 4 [N,H,W,C], got %d" % local_screen.dim())
        x = local_screen.to(dtype=torch.float32)
        if torch.max(x) > 1.0:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.screen_backbone(x)

    def _encode_screen(self, local_screen: torch.Tensor) -> torch.Tensor:
        return self.screen_encoder(self._screen_features(local_screen))

    def _priority_logits_from_features(self, screen_features: torch.Tensor) -> torch.Tensor:
        return self.priority_head(screen_features)

    def priority_logits(self, local_screen: torch.Tensor) -> torch.Tensor:
        screen_features = self._screen_features(local_screen)
        return self._priority_logits_from_features(screen_features)

    def _priority_features(self, priority_logits: torch.Tensor, local_obs: torch.Tensor) -> torch.Tensor:
        priority_probs = torch.softmax(priority_logits, dim=-1)
        top_values, top_indices = torch.topk(priority_probs, k=self.priority_top_k, dim=-1)
        top_centers = self.priority_cell_centers[top_indices]

        if local_obs.shape[-1] >= 6:
            own_xy = torch.clamp(local_obs[:, 4:6], 0.0, 1.0)
        else:
            own_xy = torch.zeros((local_obs.shape[0], 2), dtype=local_obs.dtype, device=local_obs.device)

        rel_xy = top_centers - own_xy.unsqueeze(1)
        uncertainty_proxy = 1.0 - top_values
        topk_struct = torch.cat(
            [
                rel_xy,
                top_values.unsqueeze(-1),
                uncertainty_proxy.unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(priority_logits.shape[0], -1)

        entropy = -(priority_probs * torch.log(torch.clamp(priority_probs, min=1e-8))).sum(dim=-1, keepdim=True)
        entropy_norm = entropy / max(math.log(float(max(self.priority_grid_size, 2))), 1e-6)
        return torch.cat([topk_struct, entropy_norm], dim=-1)

    def _course_context(self, course_logits: torch.Tensor, course_actions: Optional[torch.Tensor] = None):
        if course_actions is None:
            # Use expected course embedding to keep forward differentiable when
            # course action is not explicitly provided.
            probs = torch.softmax(course_logits, dim=-1)
            return torch.matmul(probs, self.course_embedding.weight)
        return self.course_embedding(course_actions)

    def _mode_context(self, mode_logits: torch.Tensor, mode_actions: Optional[torch.Tensor] = None):
        if mode_actions is None:
            # Phase2 constraint: structure only. Use deterministic dummy mode=0.
            mode_actions = torch.zeros(
                (mode_logits.shape[0],),
                dtype=torch.long,
                device=mode_logits.device,
            )
        return self.mode_embedding(mode_actions)

    @staticmethod
    def _compose_attack_logits(fire_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        # Map split heads to legacy MaCA attack space:
        # index 0 -> no-fire; indices 1..N -> fire at target/missile bucket.
        no_fire_logit = fire_logits[:, 0:1]
        fire_logit = fire_logits[:, 1:2]
        fire_target_logits = fire_logit + target_logits
        return torch.cat([no_fire_logit, fire_target_logits], dim=-1)

    def attack_logits(
        self,
        actor_h: torch.Tensor,
        course_actions: torch.Tensor,
        mode_actions: Optional[torch.Tensor] = None,
    ):
        # Build worker representation from recurrent state so attack policy stays
        # mode-conditioned while recurrent hidden remains a pure GRU state.
        mode_logits = self.mode_head(actor_h)
        mode_ctx = self._mode_context(mode_logits, mode_actions)
        worker_h = self.worker(torch.cat([actor_h, mode_ctx], dim=-1))
        course_ctx = self.course_embedding(course_actions)
        attack_input = torch.cat([worker_h, course_ctx], dim=-1)
        target_logits = self.target_head(attack_input)
        fire_logits = self.fire_head(attack_input)
        return self._compose_attack_logits(fire_logits, target_logits)

    def actor_step(
        self,
        local_obs: torch.Tensor,
        local_screen: torch.Tensor,
        actor_h: torch.Tensor,
        course_actions: Optional[torch.Tensor] = None,
        mode_actions: Optional[torch.Tensor] = None,
    ):
        screen_features = self._screen_features(local_screen)
        screen_embed = self.screen_encoder(screen_features)
        priority_logits = self._priority_logits_from_features(screen_features)
        priority_features = self._priority_features(priority_logits, local_obs)

        x = torch.cat([local_obs, screen_embed, priority_features], dim=-1)
        x = self.actor_backbone(x)
        next_h = self.actor_rnn(x, actor_h)

        mode_logits = self.mode_head(next_h)
        mode_ctx = self._mode_context(mode_logits, mode_actions)
        worker_h = self.worker(torch.cat([next_h, mode_ctx], dim=-1))

        course_logits = self.course_head(worker_h)
        course_ctx = self._course_context(course_logits, course_actions)
        attack_input = torch.cat([worker_h, course_ctx], dim=-1)
        target_logits = self.target_head(attack_input)
        fire_logits = self.fire_head(attack_input)
        attack_logits = self._compose_attack_logits(fire_logits, target_logits)
        # Return recurrent hidden state for temporal credit assignment.
        return course_logits, attack_logits, next_h

    def actor(self, local_obs: torch.Tensor, local_screen: torch.Tensor):
        actor_h = torch.zeros(
            (local_obs.shape[0], self.actor_hidden_dim),
            dtype=local_obs.dtype,
            device=local_obs.device,
        )
        course_logits, attack_logits, _ = self.actor_step(local_obs, local_screen, actor_h)
        return course_logits, attack_logits

    def _critic_mode_context(
        self,
        mode_actions: Optional[torch.Tensor],
        leading_shape,
        device,
    ) -> torch.Tensor:
        if mode_actions is None:
            zeros = torch.zeros(leading_shape, dtype=torch.long, device=device)
            return self.mode_embedding(zeros)

        mode_actions = mode_actions.to(device=device, dtype=torch.long)
        expected_ndim = len(leading_shape)
        if mode_actions.dim() == expected_ndim:
            return self.mode_embedding(mode_actions)
        if mode_actions.dim() == expected_ndim + 1:
            # Aggregate per-agent modes for centralized critic conditioning.
            return self.mode_embedding(mode_actions).mean(dim=-2)

        raise ValueError(
            "mode_actions rank mismatch: got %d expected %d or %d"
            % (mode_actions.dim(), expected_ndim, expected_ndim + 1)
        )

    def value_heads(self, global_state: torch.Tensor, mode_actions: Optional[torch.Tensor] = None):
        if global_state.dim() not in (2, 3):
            raise ValueError("Expected global_state rank 2 or 3, got %d" % global_state.dim())

        if global_state.dim() == 2:
            n, _d = global_state.shape
            mode_ctx = self._critic_mode_context(mode_actions, (n,), global_state.device)
            x = torch.cat([global_state, mode_ctx], dim=-1)
            x = torch.relu(self.critic["input_proj"](x))
            x_seq = x.unsqueeze(0)
            h0 = torch.zeros((1, n, self.cfg.hidden_size), dtype=x.dtype, device=x.device)
            out_seq, _h_n = self.critic["rnn"](x_seq, h0)
            h = out_seq[0]
            trunk = self.critic["trunk"](h)
            return {
                "team": self.critic["value_team"](trunk).squeeze(-1),
                "contact": self.critic["value_contact"](trunk).squeeze(-1),
                "opportunity": self.critic["value_opportunity"](trunk).squeeze(-1),
                "survival": self.critic["value_survival"](trunk).squeeze(-1),
            }

        t, b, _d = global_state.shape
        mode_ctx = self._critic_mode_context(mode_actions, (t, b), global_state.device)
        x = torch.cat([global_state, mode_ctx], dim=-1)
        x = torch.relu(self.critic["input_proj"](x))

        h0 = torch.zeros((1, b, self.cfg.hidden_size), dtype=x.dtype, device=x.device)
        out_seq, _h_n = self.critic["rnn"](x, h0)
        trunk = self.critic["trunk"](out_seq.reshape(t * b, self.cfg.hidden_size))

        team_values = self.critic["value_team"](trunk).reshape(t, b)
        contact_values = self.critic["value_contact"](trunk).reshape(t, b)
        opportunity_values = self.critic["value_opportunity"](trunk).reshape(t, b)
        survival_values = self.critic["value_survival"](trunk).reshape(t, b)

        return {
            "team": team_values,
            "contact": contact_values,
            "opportunity": opportunity_values,
            "survival": survival_values,
        }

    def value(self, global_state: torch.Tensor, mode_actions: Optional[torch.Tensor] = None):
        return self.value_heads(global_state, mode_actions=mode_actions)["team"]
