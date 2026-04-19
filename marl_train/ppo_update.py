from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def masked_categorical(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    invalid_logit = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~masks, invalid_logit)
    return Categorical(logits=masked_logits)


def fire_decision_logits_from_attack_logits(attack_logits: torch.Tensor, attack_masks: torch.Tensor) -> torch.Tensor:
    invalid_logit = torch.finfo(attack_logits.dtype).min
    masked_logits = attack_logits.masked_fill(~attack_masks, invalid_logit)
    no_fire_logit = masked_logits[:, 0]
    fire_logit = torch.logsumexp(masked_logits[:, 1:], dim=-1)
    has_legal_nonzero = torch.any(attack_masks[:, 1:], dim=-1)
    fire_logit = torch.where(has_legal_nonzero, fire_logit, torch.full_like(fire_logit, invalid_logit))
    return torch.stack([no_fire_logit, fire_logit], dim=-1)


def tensor_explained_variance(returns_t: torch.Tensor, values_t: torch.Tensor) -> float:
    returns_flat = returns_t.reshape(-1)
    values_flat = values_t.reshape(-1)
    valid = torch.isfinite(returns_flat) & torch.isfinite(values_flat)
    if not torch.any(valid):
        return 0.0
    returns_flat = returns_flat[valid]
    values_flat = values_flat[valid]
    if returns_flat.numel() <= 1:
        return 0.0
    var_returns = torch.var(returns_flat, unbiased=False)
    if float(var_returns.item()) < 1e-8:
        return 0.0
    var_res = torch.var(returns_flat - values_flat, unbiased=False)
    return float((1.0 - var_res / var_returns).item())


def global_grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad_norm = p.grad.data.norm(2)
        total += float(grad_norm.item()) ** 2
    return float(total ** 0.5)


def sanitize_tensor(t: torch.Tensor, clamp_abs: float = 1e6) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
    return torch.clamp(t, -clamp_abs, clamp_abs)


def sanitize_logits(logits: torch.Tensor, clamp_abs: float = 30.0) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
    return torch.clamp(logits, -clamp_abs, clamp_abs)


def ensure_valid_action_mask(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() <= 0:
        return masks
    has_any = torch.any(masks, dim=-1, keepdim=True)
    safe_masks = masks.clone()
    safe_masks[..., 0] = safe_masks[..., 0] | (~has_any.squeeze(-1))
    return safe_masks


def repair_non_finite_parameters(model: nn.Module, clamp_abs: float = 1e3) -> int:
    repaired = 0
    with torch.no_grad():
        for p in model.parameters():
            if p is None:
                continue
            bad = ~torch.isfinite(p)
            bad_count = int(bad.sum().item())
            if bad_count > 0:
                repaired += bad_count
                fixed = torch.nan_to_num(p, nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
                p.copy_(torch.clamp(fixed, -clamp_abs, clamp_abs))
    return repaired


def has_non_finite_gradients(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.all(torch.isfinite(p.grad)):
            return True
    return False


def build_recurrent_chunks(rollout_steps: int, num_envs: int, num_agents: int, chunk_len: int):
    chunks = []
    for env_idx in range(num_envs):
        for agent_idx in range(num_agents):
            start = 0
            while start < rollout_steps:
                end = min(rollout_steps, start + chunk_len)
                chunks.append((env_idx, agent_idx, start, end))
                start = end
    return chunks


def pack_recurrent_minibatch(
    chunk_indices,
    chunks,
    local_obs,
    local_screen,
    global_state,
    attack_masks,
    alive_mask,
    assigned_region_obs,
    priority_map_teacher,
    actor_h,
    old_log_prob,
    mode_action,
    mode_log_prob,
    mode_decision,
    mode_duration,
    course_action,
    attack_action,
    policy_attack_action,
    contact_signal,
    opportunity_signal,
    survival_reward,
    aux_reward_mean,
    returns_t,
    combined_adv,
    mode_adv,
    burn_in: int,
):
    batch_size = len(chunk_indices)
    hidden_dim = actor_h.shape[-1]
    obs_dim = local_obs.shape[-1]
    screen_h = local_screen.shape[-3]
    screen_w = local_screen.shape[-2]
    screen_c = local_screen.shape[-1]
    global_dim = global_state.shape[-1]
    attack_dim = attack_masks.shape[-1]
    assigned_region_dim = assigned_region_obs.shape[-1]
    priority_dim = priority_map_teacher.shape[-1]

    chunk_meta = []
    max_total_len = 0
    for chunk_idx in chunk_indices:
        env_idx, agent_idx, start, end = chunks[int(chunk_idx)]
        burn_start = max(0, start - burn_in)
        total_len = end - burn_start
        train_len = end - start
        train_offset = start - burn_start
        chunk_meta.append((env_idx, agent_idx, start, end, burn_start, total_len, train_len, train_offset))
        max_total_len = max(max_total_len, total_len)

    local_batch = local_obs.new_zeros((max_total_len, batch_size, obs_dim))
    local_screen_batch = local_screen.new_zeros((max_total_len, batch_size, screen_h, screen_w, screen_c))
    global_batch = global_state.new_zeros((max_total_len, batch_size, global_dim))
    attack_mask_batch = attack_masks.new_zeros((max_total_len, batch_size, attack_dim))
    assigned_region_batch = assigned_region_obs.new_zeros((max_total_len, batch_size, assigned_region_dim))
    priority_teacher_batch = priority_map_teacher.new_zeros((max_total_len, batch_size, priority_dim))
    actor_h_batch = actor_h.new_zeros((max_total_len, batch_size, hidden_dim))
    course_batch = course_action.new_zeros((max_total_len, batch_size))
    attack_batch = attack_action.new_zeros((max_total_len, batch_size))
    policy_attack_batch = policy_attack_action.new_zeros((max_total_len, batch_size))
    mode_batch = mode_action.new_zeros((max_total_len, batch_size))
    old_log_prob_batch = old_log_prob.new_zeros((max_total_len, batch_size))
    old_mode_log_prob_batch = mode_log_prob.new_zeros((max_total_len, batch_size))
    mode_decision_batch = mode_decision.new_zeros((max_total_len, batch_size))
    mode_duration_batch = mode_duration.new_zeros((max_total_len, batch_size))
    contact_signal_batch = contact_signal.new_zeros((max_total_len, batch_size))
    opportunity_signal_batch = opportunity_signal.new_zeros((max_total_len, batch_size))
    survival_reward_batch = survival_reward.new_zeros((max_total_len, batch_size))
    aux_reward_mean_batch = aux_reward_mean.new_zeros((max_total_len, batch_size))
    adv_batch = combined_adv.new_zeros((max_total_len, batch_size))
    mode_adv_batch = mode_adv.new_zeros((max_total_len, batch_size))
    return_batch = returns_t.new_zeros((max_total_len, batch_size))
    train_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=local_obs.device)
    active_policy_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=local_obs.device)
    seq_valid_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=local_obs.device)
    init_h = actor_h.new_zeros((batch_size, hidden_dim))
    hidden_state_init_abs_error = actor_h.new_zeros((batch_size,))

    for batch_col, meta in enumerate(chunk_meta):
        env_idx, agent_idx, start, end, burn_start, total_len, train_len, train_offset = meta
        init_h[batch_col] = actor_h[burn_start, env_idx, agent_idx]
        hidden_state_init_abs_error[batch_col] = torch.max(
            torch.abs(init_h[batch_col] - actor_h[burn_start, env_idx, agent_idx])
        )
        seq_valid_mask[:total_len, batch_col] = True

        local_batch[:total_len, batch_col] = local_obs[burn_start:end, env_idx, agent_idx]
        local_screen_batch[:total_len, batch_col] = local_screen[burn_start:end, env_idx, agent_idx]
        global_batch[:total_len, batch_col] = global_state[burn_start:end, env_idx]
        attack_mask_batch[:total_len, batch_col] = attack_masks[burn_start:end, env_idx, agent_idx]
        assigned_region_batch[:total_len, batch_col] = assigned_region_obs[burn_start:end, env_idx, agent_idx]
        priority_teacher_batch[:total_len, batch_col] = priority_map_teacher[burn_start:end, env_idx, agent_idx]
        actor_h_batch[:total_len, batch_col] = actor_h[burn_start:end, env_idx, agent_idx]
        course_batch[:total_len, batch_col] = course_action[burn_start:end, env_idx, agent_idx]
        attack_batch[:total_len, batch_col] = attack_action[burn_start:end, env_idx, agent_idx]
        policy_attack_batch[:total_len, batch_col] = policy_attack_action[burn_start:end, env_idx, agent_idx]
        mode_batch[:total_len, batch_col] = mode_action[burn_start:end, env_idx, agent_idx]
        mode_decision_batch[:total_len, batch_col] = mode_decision[burn_start:end, env_idx, agent_idx]
        mode_duration_batch[:total_len, batch_col] = mode_duration[burn_start:end, env_idx, agent_idx]
        contact_signal_batch[:total_len, batch_col] = contact_signal[burn_start:end, env_idx]
        opportunity_signal_batch[:total_len, batch_col] = opportunity_signal[burn_start:end, env_idx]
        survival_reward_batch[:total_len, batch_col] = survival_reward[burn_start:end, env_idx]
        aux_reward_mean_batch[:total_len, batch_col] = aux_reward_mean[burn_start:end, env_idx]

        train_slice = slice(train_offset, train_offset + train_len)
        train_mask[train_slice, batch_col] = True
        old_log_prob_batch[train_slice, batch_col] = old_log_prob[start:end, env_idx, agent_idx]
        old_mode_log_prob_batch[train_slice, batch_col] = mode_log_prob[start:end, env_idx, agent_idx]
        adv_batch[train_slice, batch_col] = combined_adv[start:end, env_idx, agent_idx]
        mode_adv_batch[train_slice, batch_col] = mode_adv[start:end, env_idx, agent_idx]
        return_batch[train_slice, batch_col] = returns_t[start:end, env_idx]
        active_policy_mask[train_slice, batch_col] = alive_mask[start:end, env_idx, agent_idx] > 0.5

    return {
        "init_h": init_h.detach(),
        "local_obs": local_batch,
        "local_screen": local_screen_batch,
        "global_state": global_batch,
        "attack_masks": attack_mask_batch,
        "assigned_region_obs": assigned_region_batch,
        "priority_map_teacher": priority_teacher_batch,
        "actor_h": actor_h_batch,
        "course_action": course_batch,
        "attack_action": attack_batch,
        "policy_attack_action": policy_attack_batch,
        "mode_action": mode_batch,
        "mode_decision": mode_decision_batch,
        "mode_duration": mode_duration_batch,
        "contact_signal": contact_signal_batch,
        "opportunity_signal": opportunity_signal_batch,
        "survival_reward": survival_reward_batch,
        "aux_reward_mean": aux_reward_mean_batch,
        "old_log_prob": old_log_prob_batch,
        "old_mode_log_prob": old_mode_log_prob_batch,
        "advantages": adv_batch,
        "mode_advantages": mode_adv_batch,
        "returns": return_batch,
        "train_mask": train_mask,
        "active_policy_mask": active_policy_mask,
        "seq_valid_mask": seq_valid_mask,
        "hidden_state_init_abs_error": hidden_state_init_abs_error,
    }


def ppo_update(
    model,
    optimizer,
    buffer,
    advantages_team,
    advantages_aux,
    returns,
    device,
    args,
    value_normalizer=None,
    teacher_model=None,
    update_idx: int = 0,
):
    local_obs = torch.as_tensor(buffer["local_obs"], dtype=torch.float32, device=device)
    local_screen = torch.as_tensor(buffer["local_screen"], dtype=torch.uint8, device=device)
    global_state = torch.as_tensor(buffer["global_state"], dtype=torch.float32, device=device)
    attack_masks = torch.as_tensor(buffer["attack_masks"], dtype=torch.bool, device=device)
    alive_mask = torch.as_tensor(buffer["alive_mask"], dtype=torch.float32, device=device)
    assigned_region_obs = torch.as_tensor(buffer["assigned_region_obs"], dtype=torch.float32, device=device)
    priority_map_teacher = torch.as_tensor(buffer["priority_map_teacher"], dtype=torch.float32, device=device)
    actor_h = torch.as_tensor(buffer["actor_h"], dtype=torch.float32, device=device)
    old_log_prob = torch.as_tensor(buffer["log_prob"], dtype=torch.float32, device=device)
    old_mode_log_prob = torch.as_tensor(buffer["mode_log_prob"], dtype=torch.float32, device=device)
    course_action = torch.as_tensor(buffer["course_action"], dtype=torch.long, device=device)
    attack_action = torch.as_tensor(buffer["attack_action"], dtype=torch.long, device=device)
    policy_attack_action = torch.as_tensor(buffer.get("policy_attack_action", buffer["attack_action"]), dtype=torch.long, device=device)
    mode_action = torch.as_tensor(buffer["mode_action"], dtype=torch.long, device=device)
    mode_decision = torch.as_tensor(buffer["mode_decision"], dtype=torch.float32, device=device)
    mode_duration = torch.as_tensor(buffer["mode_duration"], dtype=torch.float32, device=device)
    contact_signal_t = torch.as_tensor(buffer["contact_signal"], dtype=torch.float32, device=device)
    opportunity_signal_t = torch.as_tensor(buffer["opportunity_signal"], dtype=torch.float32, device=device)
    survival_reward_t = torch.as_tensor(buffer["survival_reward"], dtype=torch.float32, device=device)
    aux_reward_mean_t = torch.as_tensor(
        np.mean(buffer["agent_aux_reward"], axis=-1),
        dtype=torch.float32,
        device=device,
    )

    adv_team_t = torch.as_tensor(advantages_team, dtype=torch.float32, device=device)
    adv_aux_t = torch.as_tensor(advantages_aux, dtype=torch.float32, device=device)

    if value_normalizer is not None:
        norm_returns = value_normalizer.normalize(returns)
        returns_t = torch.as_tensor(norm_returns, dtype=torch.float32, device=device)
    else:
        norm_returns = returns
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    returns_denorm_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    combined_adv = args.team_adv_weight * adv_team_t.unsqueeze(-1) + args.aux_adv_weight * adv_aux_t
    combined_adv = sanitize_tensor(combined_adv)
    active_mask_bool = alive_mask > 0.5
    active_adv = combined_adv[active_mask_bool]
    if active_adv.numel() > 0:
        adv_mean = active_adv.mean()
        adv_std = active_adv.std(unbiased=False)
        combined_adv = (combined_adv - adv_mean) / torch.clamp(adv_std, min=1e-6)
    combined_adv = combined_adv * alive_mask

    mode_adv = adv_team_t.unsqueeze(-1) * torch.clamp(mode_duration, min=1.0)
    mode_adv = mode_adv * mode_decision * alive_mask
    mode_adv = sanitize_tensor(mode_adv)
    active_mode_adv = mode_adv[mode_decision > 0.5]
    if active_mode_adv.numel() > 0:
        mode_adv_mean = active_mode_adv.mean()
        mode_adv_std = active_mode_adv.std(unbiased=False)
        mode_adv = (mode_adv - mode_adv_mean) / torch.clamp(mode_adv_std, min=1e-6)
        mode_adv = mode_adv * mode_decision * alive_mask

    imitation_coef = float(max(args.imitation_coef, 0.0))
    if imitation_coef > 0.0 and int(args.imitation_warmup_updates) > 0:
        warmup = float(max(1, int(args.imitation_warmup_updates)))
        decay = max(0.0, 1.0 - float(max(update_idx, 0)) / warmup)
        imitation_coef = imitation_coef * decay

    with torch.no_grad():
        explained_var = tensor_explained_variance(
            returns_denorm_t,
            torch.as_tensor(buffer["value"], dtype=torch.float32, device=device),
        )

    rollout_steps, num_envs, num_agents = local_obs.shape[0], local_obs.shape[1], local_obs.shape[2]
    chunks = build_recurrent_chunks(rollout_steps, num_envs, num_agents, args.chunk_len)
    num_chunks = len(chunks)

    total_policy_low = 0.0
    total_policy_high = 0.0
    total_value = 0.0
    total_entropy_low = 0.0
    total_entropy_high = 0.0
    total_imitation = 0.0
    total_minibatches = 0
    total_active = 0
    total_grad_steps = 0
    total_actor_grad_norm = 0.0
    total_critic_grad_norm = 0.0
    total_train_mask_count = 0
    total_active_mask_count = 0
    total_hidden_init_err = 0.0
    total_hidden_init_count = 0
    total_skipped_non_finite = 0
    total_repaired_params = 0
    total_value_team_loss = 0.0
    total_value_contact_loss = 0.0
    total_value_opportunity_loss = 0.0
    total_value_survival_loss = 0.0
    total_priority_aux_loss = 0.0

    actor_params = [p for name, p in model.named_parameters() if not name.startswith("critic")]
    critic_params = [p for name, p in model.named_parameters() if name.startswith("critic")]
    attack_policy_mode = str(getattr(args, "attack_policy_mode", "full_discrete")).lower()
    disable_aux_value_heads = bool(getattr(args, "disable_aux_value_heads", False))
    aux_value_loss_coeff = float(max(getattr(args, "aux_value_loss_coeff", 0.25), 0.0))
    priority_aux_loss_coeff = float(max(getattr(args, "priority_aux_loss_coeff", 0.05), 0.0))
    if disable_aux_value_heads:
        aux_value_loss_coeff = 0.0

    for _ in range(args.ppo_epochs):
        order = np.random.permutation(num_chunks)
        mini_batches = np.array_split(order, max(1, args.num_mini_batches))

        for mini_batch_indices in mini_batches:
            if len(mini_batch_indices) <= 0:
                continue

            repaired = repair_non_finite_parameters(model)
            total_repaired_params += int(repaired)

            packed = pack_recurrent_minibatch(
                mini_batch_indices,
                chunks,
                local_obs,
                local_screen,
                global_state,
                attack_masks,
                alive_mask,
                assigned_region_obs,
                priority_map_teacher,
                actor_h,
                old_log_prob,
                mode_action,
                old_mode_log_prob,
                mode_decision,
                mode_duration,
                course_action,
                attack_action,
                policy_attack_action,
                contact_signal_t,
                opportunity_signal_t,
                survival_reward_t,
                aux_reward_mean_t,
                returns_t,
                combined_adv,
                mode_adv,
                args.burn_in,
            )

            train_mask = packed["train_mask"]
            active_train_mask = train_mask & packed["active_policy_mask"]
            train_returns = packed["returns"][active_train_mask]
            if train_returns.numel() <= 0:
                continue

            train_returns = sanitize_tensor(train_returns)

            total_train_mask_count += int(train_mask.sum().item())
            total_active_mask_count += int(active_train_mask.sum().item())
            init_err = packed["hidden_state_init_abs_error"]
            total_hidden_init_err += float(init_err.sum().item())
            total_hidden_init_count += int(init_err.numel())

            value_heads_t = model.value_heads(
                sanitize_tensor(packed["global_state"]),
                mode_actions=packed["mode_action"],
            )
            value_team_pred = sanitize_tensor(value_heads_t["team"][active_train_mask])
            value_contact_pred = sanitize_tensor(value_heads_t["contact"][active_train_mask])
            value_opportunity_pred = sanitize_tensor(value_heads_t["opportunity"][active_train_mask])
            value_survival_pred = sanitize_tensor(value_heads_t["survival"][active_train_mask])

            contact_target = sanitize_tensor(packed["contact_signal"][active_train_mask])
            opportunity_target = sanitize_tensor(packed["opportunity_signal"][active_train_mask])
            survival_target = sanitize_tensor(packed["survival_reward"][active_train_mask])

            def _norm_target(x: torch.Tensor) -> torch.Tensor:
                if x.numel() <= 1:
                    return x
                mean = x.mean()
                std = x.std(unbiased=False)
                return (x - mean) / torch.clamp(std, min=1e-6)

            value_team_loss = 0.5 * (value_team_pred - train_returns).pow(2).mean()
            value_contact_loss = 0.5 * (value_contact_pred - _norm_target(contact_target)).pow(2).mean()
            value_opportunity_loss = 0.5 * (value_opportunity_pred - _norm_target(opportunity_target)).pow(2).mean()
            value_survival_loss = 0.5 * (value_survival_pred - _norm_target(survival_target)).pow(2).mean()
            value_aux_loss = value_contact_loss + value_opportunity_loss + value_survival_loss
            value_loss = (
                value_team_loss
                + aux_value_loss_coeff * value_aux_loss
            ) * args.value_loss_coeff

            h = packed["init_h"]
            new_log_probs = []
            old_log_probs = []
            entropies = []
            advantages = []
            high_new_log_probs = []
            high_old_log_probs = []
            high_entropies = []
            high_advantages = []
            imitation_losses = []
            priority_aux_losses = []

            max_total_len = packed["local_obs"].shape[0]
            for seq_idx in range(max_total_len):
                seq_valid = packed["seq_valid_mask"][seq_idx]
                if not torch.any(seq_valid):
                    continue

                seq_local = packed["local_obs"][seq_idx, seq_valid]
                seq_screen = packed["local_screen"][seq_idx, seq_valid]
                seq_h = h[seq_valid]
                seq_course = packed["course_action"][seq_idx, seq_valid]
                seq_mode = packed["mode_action"][seq_idx, seq_valid]
                seq_assigned_region = packed["assigned_region_obs"][seq_idx, seq_valid]

                course_logits, attack_logits, next_h_valid = model.actor_step(
                    seq_local,
                    seq_screen,
                    seq_h,
                    course_actions=seq_course,
                    mode_actions=seq_mode,
                    assigned_region_obs=seq_assigned_region,
                )
                course_logits = sanitize_logits(course_logits)
                attack_logits = sanitize_logits(attack_logits)
                next_h_valid = sanitize_tensor(next_h_valid)
                h = h.clone()
                h[seq_valid] = next_h_valid

                seq_train = packed["train_mask"][seq_idx, seq_valid]
                if not torch.any(seq_train):
                    continue

                course_logits = course_logits[seq_train]
                attack_logits = attack_logits[seq_train]
                seq_course = seq_course[seq_train]
                seq_attack_mask = packed["attack_masks"][seq_idx, seq_valid][seq_train]
                seq_attack_mask = ensure_valid_action_mask(seq_attack_mask)
                seq_old_log_prob = packed["old_log_prob"][seq_idx, seq_valid][seq_train]
                seq_adv = packed["advantages"][seq_idx, seq_valid][seq_train]
                seq_active = packed["active_policy_mask"][seq_idx, seq_valid][seq_train]
                seq_mode = packed["mode_action"][seq_idx, seq_valid][seq_train]
                seq_mode_old_log_prob = packed["old_mode_log_prob"][seq_idx, seq_valid][seq_train]
                seq_mode_decision = packed["mode_decision"][seq_idx, seq_valid][seq_train]
                seq_mode_adv = packed["mode_advantages"][seq_idx, seq_valid][seq_train]
                seq_actor_h = packed["actor_h"][seq_idx, seq_valid][seq_train]
                seq_old_log_prob = sanitize_tensor(seq_old_log_prob)
                seq_adv = sanitize_tensor(seq_adv)
                seq_mode_old_log_prob = sanitize_tensor(seq_mode_old_log_prob)
                seq_mode_adv = sanitize_tensor(seq_mode_adv)
                seq_actor_h = sanitize_tensor(seq_actor_h)

                if course_logits.shape[0] <= 0:
                    continue

                if priority_aux_loss_coeff > 0.0:
                    seq_priority_teacher = packed["priority_map_teacher"][seq_idx, seq_valid][seq_train]
                    seq_priority_logits = sanitize_logits(model.priority_logits(seq_screen[seq_train]))
                    if torch.any(seq_active):
                        active_priority_logits = seq_priority_logits[seq_active]
                        active_priority_teacher = sanitize_tensor(seq_priority_teacher[seq_active])
                        active_priority_teacher = torch.clamp(active_priority_teacher, min=0.0)
                        teacher_sum = torch.clamp(active_priority_teacher.sum(dim=-1, keepdim=True), min=1e-6)
                        active_priority_teacher = active_priority_teacher / teacher_sum
                        log_prob = torch.log_softmax(active_priority_logits, dim=-1)
                        priority_aux_losses.append(-(active_priority_teacher * log_prob).sum(dim=-1))

                course_dist = Categorical(logits=course_logits)
                if attack_policy_mode == "fire_or_not":
                    seq_policy_attack = packed["policy_attack_action"][seq_idx, seq_valid][seq_train]
                    fire_logits = fire_decision_logits_from_attack_logits(attack_logits, seq_attack_mask)
                    attack_dist = Categorical(logits=fire_logits)
                    seq_log_prob = course_dist.log_prob(seq_course) + attack_dist.log_prob(seq_policy_attack)
                    seq_entropy = course_dist.entropy() + attack_dist.entropy()
                else:
                    seq_attack = packed["attack_action"][seq_idx, seq_valid][seq_train]
                    attack_dist = masked_categorical(attack_logits, seq_attack_mask)
                    seq_log_prob = course_dist.log_prob(seq_course) + attack_dist.log_prob(seq_attack)
                    seq_entropy = course_dist.entropy() + attack_dist.entropy()

                if torch.any(seq_active):
                    new_log_probs.append(seq_log_prob[seq_active])
                    old_log_probs.append(seq_old_log_prob[seq_active])
                    entropies.append(seq_entropy[seq_active])
                    advantages.append(seq_adv[seq_active])

                    if teacher_model is not None and imitation_coef > 0.0:
                        with torch.no_grad():
                            teacher_course_logits, teacher_attack_logits, _teacher_next_h = teacher_model.actor_step(
                                seq_local,
                                seq_screen,
                                seq_h.detach(),
                                course_actions=packed["course_action"][seq_idx, seq_valid],
                                mode_actions=packed["mode_action"][seq_idx, seq_valid],
                                assigned_region_obs=seq_assigned_region,
                            )
                            teacher_course_logits = teacher_course_logits[seq_train]
                            teacher_attack_logits = teacher_attack_logits[seq_train]
                            teacher_course_logits = sanitize_logits(teacher_course_logits)
                            teacher_attack_logits = sanitize_logits(teacher_attack_logits)
                            teacher_attack_logits = teacher_attack_logits.masked_fill(
                                ~seq_attack_mask,
                                torch.finfo(teacher_attack_logits.dtype).min,
                            )

                        student_attack_logits = sanitize_logits(attack_logits).masked_fill(
                            ~seq_attack_mask,
                            torch.finfo(attack_logits.dtype).min,
                        )
                        student_course = Categorical(logits=sanitize_logits(course_logits))
                        teacher_course = Categorical(logits=teacher_course_logits)
                        student_attack = Categorical(logits=student_attack_logits)
                        teacher_attack = Categorical(logits=teacher_attack_logits)
                        kl_course = torch.distributions.kl_divergence(student_course, teacher_course)
                        kl_attack = torch.distributions.kl_divergence(student_attack, teacher_attack)
                        imitation_losses.append((kl_course + kl_attack)[seq_active])

                    mode_logits = sanitize_logits(model.mode_head(seq_actor_h))
                    mode_dist = Categorical(logits=mode_logits)
                    seq_mode_log_prob = mode_dist.log_prob(seq_mode)
                    seq_mode_entropy = mode_dist.entropy()
                    mode_active = seq_active & (seq_mode_decision > 0.5)
                    if torch.any(mode_active):
                        high_new_log_probs.append(seq_mode_log_prob[mode_active])
                        high_old_log_probs.append(seq_mode_old_log_prob[mode_active])
                        high_entropies.append(seq_mode_entropy[mode_active])
                        high_advantages.append(seq_mode_adv[mode_active])

            if new_log_probs:
                new_log_prob_t = torch.cat(new_log_probs, dim=0)
                old_log_prob_t = torch.cat(old_log_probs, dim=0)
                adv_t = torch.cat(advantages, dim=0)
                entropy_t = torch.cat(entropies, dim=0)
                new_log_prob_t = sanitize_tensor(new_log_prob_t)
                old_log_prob_t = sanitize_tensor(old_log_prob_t)
                adv_t = sanitize_tensor(adv_t)
                entropy_t = sanitize_tensor(entropy_t)

                ratio = torch.exp(new_log_prob_t - old_log_prob_t)
                clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                policy_loss_low = -torch.min(ratio * adv_t, clipped_ratio * adv_t).mean()
                entropy_loss_low = -args.entropy_coeff * entropy_t.mean()
                entropy_mean_low = float(entropy_t.mean().item())
                total_active += int(adv_t.numel())
            else:
                policy_loss_low = value_loss * 0.0
                entropy_loss_low = value_loss * 0.0
                entropy_mean_low = 0.0

            if high_new_log_probs:
                high_new_log_prob_t = torch.cat(high_new_log_probs, dim=0)
                high_old_log_prob_t = torch.cat(high_old_log_probs, dim=0)
                high_adv_t = torch.cat(high_advantages, dim=0)
                high_entropy_t = torch.cat(high_entropies, dim=0)
                high_new_log_prob_t = sanitize_tensor(high_new_log_prob_t)
                high_old_log_prob_t = sanitize_tensor(high_old_log_prob_t)
                high_adv_t = sanitize_tensor(high_adv_t)
                high_entropy_t = sanitize_tensor(high_entropy_t)

                high_ratio = torch.exp(high_new_log_prob_t - high_old_log_prob_t)
                high_clipped_ratio = torch.clamp(high_ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                policy_loss_high = -torch.min(high_ratio * high_adv_t, high_clipped_ratio * high_adv_t).mean()
                entropy_loss_high = -args.high_level_entropy_coeff * high_entropy_t.mean()
                entropy_mean_high = float(high_entropy_t.mean().item())
            else:
                policy_loss_high = value_loss * 0.0
                entropy_loss_high = value_loss * 0.0
                entropy_mean_high = 0.0

            policy_loss = policy_loss_low + args.high_level_loss_coeff * policy_loss_high
            entropy_loss = entropy_loss_low + entropy_loss_high

            if imitation_losses:
                imitation_loss = sanitize_tensor(torch.cat(imitation_losses, dim=0)).mean()
            else:
                imitation_loss = value_loss * 0.0

            if priority_aux_losses:
                priority_aux_loss = sanitize_tensor(torch.cat(priority_aux_losses, dim=0)).mean()
            else:
                priority_aux_loss = value_loss * 0.0

            loss = (
                policy_loss
                + value_loss
                + entropy_loss
                + imitation_coef * imitation_loss
                + priority_aux_loss_coeff * priority_aux_loss
            )
            if not torch.isfinite(loss):
                total_skipped_non_finite += 1
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            if has_non_finite_gradients(model):
                total_skipped_non_finite += 1
                optimizer.zero_grad()
                repaired = repair_non_finite_parameters(model)
                total_repaired_params += int(repaired)
                continue
            actor_grad_norm = global_grad_norm(actor_params)
            critic_grad_norm = global_grad_norm(critic_params)
            max_actor_grad_norm = float(getattr(args, "max_actor_grad_norm", 0.0) or 0.0)
            max_critic_grad_norm = float(getattr(args, "max_critic_grad_norm", 0.0) or 0.0)
            max_total_grad_norm = float(getattr(args, "max_grad_norm", 0.0) or 0.0)
            if max_actor_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(actor_params, max_actor_grad_norm)
            if max_critic_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(critic_params, max_critic_grad_norm)
            if max_total_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_total_grad_norm)
            optimizer.step()

            repaired = repair_non_finite_parameters(model)
            total_repaired_params += int(repaired)

            total_policy_low += float(policy_loss_low.item())
            total_policy_high += float(policy_loss_high.item())
            total_value += float(value_loss.item())
            total_value_team_loss += float(value_team_loss.item())
            total_value_contact_loss += float(value_contact_loss.item())
            total_value_opportunity_loss += float(value_opportunity_loss.item())
            total_value_survival_loss += float(value_survival_loss.item())
            total_entropy_low += entropy_mean_low
            total_entropy_high += entropy_mean_high
            total_imitation += float(imitation_loss.item())
            total_priority_aux_loss += float(priority_aux_loss.item())
            total_minibatches += 1
            total_grad_steps += 1
            total_actor_grad_norm += actor_grad_norm
            total_critic_grad_norm += critic_grad_norm

    if total_minibatches <= 0:
        return {
            "policy_loss": 0.0,
            "policy_loss_low": 0.0,
            "policy_loss_high": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "entropy_low": 0.0,
            "entropy_high": 0.0,
            "imitation_loss": 0.0,
            "imitation_coef": float(imitation_coef),
            "active_samples": 0,
            "grad_steps": 0,
            "explained_variance": explained_var,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "active_mask_ratio": 0.0,
            "hidden_state_init_abs_error": 0.0,
            "skipped_non_finite_batches": int(total_skipped_non_finite),
            "repaired_non_finite_params": int(total_repaired_params),
            "value_target_mean": float(np.mean(returns)),
            "value_target_std": float(np.std(returns)),
            "value_target_norm_mean": float(np.mean(norm_returns)) if value_normalizer is not None else float(np.mean(returns)),
            "value_target_norm_std": float(np.std(norm_returns)) if value_normalizer is not None else float(np.std(returns)),
            "value_team_loss": 0.0,
            "value_contact_loss": 0.0,
            "value_opportunity_loss": 0.0,
            "value_survival_loss": 0.0,
            "value_aux_loss": 0.0,
            "priority_aux_loss": 0.0,
            "priority_aux_loss_coeff": float(priority_aux_loss_coeff),
            "aux_value_loss_coeff": float(aux_value_loss_coeff),
            "disable_aux_value_heads": 1.0 if disable_aux_value_heads else 0.0,
        }

    active_mask_ratio = 0.0
    if total_train_mask_count > 0:
        active_mask_ratio = float(total_active_mask_count) / float(total_train_mask_count)

    hidden_state_init_abs_error = 0.0
    if total_hidden_init_count > 0:
        hidden_state_init_abs_error = total_hidden_init_err / float(total_hidden_init_count)

    return {
        "policy_loss": (total_policy_low + args.high_level_loss_coeff * total_policy_high) / float(total_minibatches),
        "policy_loss_low": total_policy_low / float(total_minibatches),
        "policy_loss_high": total_policy_high / float(total_minibatches),
        "value_loss": total_value / float(total_minibatches),
        "entropy": (total_entropy_low + total_entropy_high) / float(total_minibatches),
        "entropy_low": total_entropy_low / float(total_minibatches),
        "entropy_high": total_entropy_high / float(total_minibatches),
        "imitation_loss": total_imitation / float(total_minibatches),
        "imitation_coef": float(imitation_coef),
        "active_samples": total_active,
        "grad_steps": total_grad_steps,
        "explained_variance": explained_var,
        "actor_grad_norm": total_actor_grad_norm / float(total_minibatches),
        "critic_grad_norm": total_critic_grad_norm / float(total_minibatches),
        "active_mask_ratio": active_mask_ratio,
        "hidden_state_init_abs_error": hidden_state_init_abs_error,
        "skipped_non_finite_batches": int(total_skipped_non_finite),
        "repaired_non_finite_params": int(total_repaired_params),
        "value_target_mean": float(np.mean(returns)),
        "value_target_std": float(np.std(returns)),
        "value_target_norm_mean": float(np.mean(norm_returns)) if value_normalizer is not None else float(np.mean(returns)),
        "value_target_norm_std": float(np.std(norm_returns)) if value_normalizer is not None else float(np.std(returns)),
        "value_team_loss": total_value_team_loss / float(total_minibatches),
        "value_contact_loss": total_value_contact_loss / float(total_minibatches),
        "value_opportunity_loss": total_value_opportunity_loss / float(total_minibatches),
        "value_survival_loss": total_value_survival_loss / float(total_minibatches),
        "value_aux_loss": (total_value_contact_loss + total_value_opportunity_loss + total_value_survival_loss)
        / float(total_minibatches),
        "priority_aux_loss": total_priority_aux_loss / float(total_minibatches),
        "priority_aux_loss_coeff": float(priority_aux_loss_coeff),
        "aux_value_loss_coeff": float(aux_value_loss_coeff),
        "disable_aux_value_heads": 1.0 if disable_aux_value_heads else 0.0,
    }


def action_distribution_stats(
    course_actions: np.ndarray,
    attack_actions: np.ndarray,
    alive_mask: np.ndarray,
    attack_masks: np.ndarray,
    course_dim: int,
    attack_dim: int,
    policy_attack_actions: Optional[np.ndarray] = None,
    attack_policy_mode: str = "full_discrete",
    attack_rule_mode: str = "none",
) -> Dict[str, float]:
    alive = alive_mask > 0.5
    stats: Dict[str, float] = {}
    attack_policy_mode = str(attack_policy_mode).lower()
    attack_rule_mode = str(attack_rule_mode).lower()

    stats["attack_policy_mode_full_discrete"] = 1.0 if attack_policy_mode == "full_discrete" else 0.0
    stats["attack_policy_mode_fire_or_not"] = 1.0 if attack_policy_mode == "fire_or_not" else 0.0
    stats["attack_rule_mode_none"] = 1.0 if attack_rule_mode == "none" else 0.0
    stats["attack_rule_mode_nearest_target"] = 1.0 if attack_rule_mode == "nearest_target" else 0.0

    legal_nonzero = attack_masks[..., 1:] & alive[..., None]
    legal_opportunity = np.any(legal_nonzero, axis=-1) & alive
    legal_count = int(np.sum(legal_opportunity))
    alive_count = int(np.sum(alive))
    executed_nonzero = (attack_actions > 0) & alive
    executed_fire_on_legal = executed_nonzero & legal_opportunity

    stats["attack_opportunity_frac"] = float(legal_count / max(alive_count, 1))
    stats["executed_fire_action_frac"] = float(np.sum(executed_nonzero) / max(alive_count, 1))
    stats["no_fire_when_legal_frac"] = float(np.sum(legal_opportunity & (~executed_nonzero)) / max(legal_count, 1))
    stats["opportunity_to_fire_ratio"] = float(np.sum(executed_fire_on_legal) / max(legal_count, 1))

    if attack_policy_mode == "fire_or_not" and policy_attack_actions is not None:
        fire_decision = (np.asarray(policy_attack_actions) > 0) & alive
        no_fire_decision = (~fire_decision) & alive
        stats["fire_decision_freq_00"] = float(np.sum(no_fire_decision) / max(alive_count, 1))
        stats["fire_decision_freq_01"] = float(np.sum(fire_decision) / max(alive_count, 1))
        if attack_rule_mode == "nearest_target":
            stats["rule_selected_attack_nonzero_freq"] = float(np.sum(fire_decision & executed_nonzero) / max(alive_count, 1))
        else:
            stats["rule_selected_attack_nonzero_freq"] = float(np.sum(executed_nonzero) / max(alive_count, 1))
    else:
        stats["fire_decision_freq_00"] = 0.0
        stats["fire_decision_freq_01"] = 0.0
        stats["rule_selected_attack_nonzero_freq"] = 0.0

    if np.any(alive):
        selected_course = course_actions[alive]
        selected_attack = attack_actions[alive]
        total = float(selected_course.size)
        for action in range(course_dim):
            stats["course_action_freq_%02d" % action] = float(np.mean(selected_course == action))
        for action in range(attack_dim):
            stats["attack_action_freq_%02d" % action] = float(np.mean(selected_attack == action))

        legal_non_zero = attack_masks[..., 1:] & alive[..., None]
        if np.any(legal_non_zero):
            legal_count = np.sum(legal_non_zero, axis=(0, 1, 2)).astype(np.float64)
            pick_count = np.array(
                [np.sum((attack_actions == (a + 1)) & alive) for a in range(attack_dim - 1)],
                dtype=np.float64,
            )
            valid_idx = legal_count > 0
            if np.any(valid_idx):
                usage = np.zeros_like(legal_count)
                usage[valid_idx] = pick_count[valid_idx] / legal_count[valid_idx]
                stats["attack_legal_usage_min_nonzero"] = float(np.min(usage[valid_idx]))
                stats["attack_legal_usage_mean_nonzero"] = float(np.mean(usage[valid_idx]))
            else:
                stats["attack_legal_usage_min_nonzero"] = 0.0
                stats["attack_legal_usage_mean_nonzero"] = 0.0
        else:
            stats["attack_legal_usage_min_nonzero"] = 0.0
            stats["attack_legal_usage_mean_nonzero"] = 0.0

        stats["alive_sample_count"] = total
    else:
        for action in range(course_dim):
            stats["course_action_freq_%02d" % action] = 0.0
        for action in range(attack_dim):
            stats["attack_action_freq_%02d" % action] = 0.0
        stats["attack_legal_usage_min_nonzero"] = 0.0
        stats["attack_legal_usage_mean_nonzero"] = 0.0
        stats["alive_sample_count"] = 0.0

    return stats
