import numpy as np


FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = FIGHTER_NUM * 2 + 1
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM

RADAR_POINT_NUM = 10
DEFAULT_DISTURB_POINT = 11
LONG_ATTACK_MAX_DISTANCE = 120.0
SHORT_ATTACK_MAX_DISTANCE = 50.0


def get_support_action(step_cnt, fighter_idx):
    """Use a deterministic radar sweep so training and evaluation match."""
    radar_point = int((step_cnt + fighter_idx) % RADAR_POINT_NUM) + 1
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
        valid_attack_indices = get_valid_attack_indices(info_obs)
        for course_idx in range(COURSE_NUM):
            base = course_idx * ATTACK_IND_NUM
            mask[row_idx, base + np.asarray(valid_attack_indices, dtype=np.int64)] = True
    return mask
