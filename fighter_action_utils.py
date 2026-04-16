import os

import numpy as np


FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = FIGHTER_NUM * 2 + 1
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM
COURSE_BASE_OFFSETS = np.arange(COURSE_NUM, dtype=np.int64) * ATTACK_IND_NUM

RADAR_POINT_NUM = 10
DEFAULT_DISTURB_POINT = 11
LONG_ATTACK_MAX_DISTANCE = 120.0
SHORT_ATTACK_MAX_DISTANCE = 50.0


def get_support_action(step_cnt, fighter_idx):
    """Use a deterministic but optionally slower radar sweep."""
    sweep_interval = max(1, int(os.getenv("MACA_SUPPORT_SWEEP_INTERVAL", "1")))
    radar_phase = int(step_cnt // sweep_interval)
    radar_point = int((radar_phase + fighter_idx) % RADAR_POINT_NUM) + 1
    return radar_point, DEFAULT_DISTURB_POINT


def get_valid_attack_indices(info_obs):
    info = np.asarray(info_obs, dtype=np.float32).reshape(-1)
    valid = [0]
    if info.size < 5:
        return valid

    long_missile_left = int(max(info[1], 0))
    short_missile_left = int(max(info[2], 0))
    target_distance = float(info[3])
    target_id = int(info[4])

    if target_id <= 0 or target_id > FIGHTER_NUM or target_distance <= 0:
        return valid

    if long_missile_left > 0 and target_distance <= LONG_ATTACK_MAX_DISTANCE:
        valid.append(target_id)
    if short_missile_left > 0 and target_distance <= SHORT_ATTACK_MAX_DISTANCE:
        valid.append(target_id + FIGHTER_NUM)
    return valid


def build_valid_action_masks(info_obs_batch):
    batch = np.asarray(info_obs_batch, dtype=np.float32)
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)

    mask = np.zeros((batch.shape[0], ACTION_NUM), dtype=np.bool_)
    for row_idx, info_obs in enumerate(batch):
        valid_attack_indices = np.asarray(get_valid_attack_indices(info_obs), dtype=np.int64)
        mask[row_idx, (COURSE_BASE_OFFSETS[:, None] + valid_attack_indices[None, :]).reshape(-1)] = True
    return mask


def get_valid_attack_indices_from_raw(fighter_raw_obs):
    """Build valid target-specific attack indices from raw visible targets.

    Unlike the legacy `simple` obs path, this does not collapse attack
    legality to only the nearest visible target. Every visible fighter target
    in range gets its corresponding long/short missile action enabled.
    """

    valid = [0]
    if fighter_raw_obs is None or not fighter_raw_obs.get("alive", False):
        return valid

    long_missile_left = int(max(fighter_raw_obs.get("l_missile_left", 0), 0))
    short_missile_left = int(max(fighter_raw_obs.get("s_missile_left", 0), 0))
    own_x = float(fighter_raw_obs.get("pos_x", 0.0))
    own_y = float(fighter_raw_obs.get("pos_y", 0.0))
    visible_list = list(fighter_raw_obs.get("r_visible_list", []))

    for target in visible_list:
        target_id = int(target.get("id", 0))
        if target_id <= 0 or target_id > FIGHTER_NUM:
            continue

        dx = own_x - float(target.get("pos_x", own_x))
        dy = own_y - float(target.get("pos_y", own_y))
        target_distance = float(np.sqrt(dx * dx + dy * dy))

        if long_missile_left > 0 and target_distance <= LONG_ATTACK_MAX_DISTANCE:
            valid.append(target_id)
        if short_missile_left > 0 and target_distance <= SHORT_ATTACK_MAX_DISTANCE:
            valid.append(target_id + FIGHTER_NUM)

    return sorted(set(valid))


def build_attack_mask_from_raw(fighter_raw_obs):
    attack_mask = np.zeros((ATTACK_IND_NUM,), dtype=np.bool_)
    valid_attack_indices = np.asarray(get_valid_attack_indices_from_raw(fighter_raw_obs), dtype=np.int64)
    attack_mask[valid_attack_indices] = True
    return attack_mask


def build_decoupled_action_mask_from_raw(fighter_raw_obs):
    course_mask = np.ones((COURSE_NUM,), dtype=np.bool_)
    attack_mask = build_attack_mask_from_raw(fighter_raw_obs)

    if fighter_raw_obs is None or not fighter_raw_obs.get("alive", False):
        course_mask[:] = False
        course_mask[0] = True
        attack_mask[:] = False
        attack_mask[0] = True

    return np.concatenate([course_mask, attack_mask], axis=0)
