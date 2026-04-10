#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
import glob
import importlib
import json
import os
import random
import re
import shutil
import time

import numpy as np
import torch

import dqn
from fighter_action_utils import get_support_action
from interface import Environment

DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_opponent(agent_name):
    module = importlib.import_module(f'agent.{agent_name}.agent')
    return module.Agent()


def ensure_latest_checkpoint_link(model_dir):
    pattern = os.path.join(model_dir, 'model_*.pkl')
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    # Choose checkpoint by mtime first (latest produced file), then numeric step.
    # This avoids lexicographic mistakes and also works when historical checkpoints
    # have larger step ids than a fresh run's first checkpoint.
    def checkpoint_step(path):
        name = os.path.basename(path)
        match = re.match(r'^model_(\d+)\.pkl$', name)
        return int(match.group(1)) if match else -1

    latest = max(candidates, key=lambda path: (os.path.getmtime(path), checkpoint_step(path)))
    fixed_model = os.path.join(model_dir, 'model.pkl')
    shutil.copy2(latest, fixed_model)
    return latest


def infer_resume_step(resume_path, model_dir):
    name = os.path.basename(resume_path)
    match = re.match(r'^model_(\d+)\.pkl$', name)
    if match:
        return int(match.group(1))

    # If resuming from model.pkl, try to infer step from latest numbered checkpoint.
    latest = ensure_latest_checkpoint_link(model_dir)
    if not latest:
        return 0
    latest_name = os.path.basename(latest)
    latest_match = re.match(r'^model_(\d+)\.pkl$', latest_name)
    return int(latest_match.group(1)) if latest_match else 0


def main():
    parser = argparse.ArgumentParser(description='Train MaCA DQN pipeline with configurable settings.')
    parser.add_argument('--map', default='maps/1000_1000_fighter10v10.map')
    parser.add_argument('--opponent', default='fix_rule_no_att', help='agent folder name under agent/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_step', type=int, default=1500)
    parser.add_argument('--learn_interval', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_increment', type=float, default=-0.00005)
    parser.add_argument('--target_replace_iter', type=int, default=100)
    parser.add_argument('--memory_size', type=int, default=2000)
    parser.add_argument('--min_replay_size', type=int, default=2000)
    parser.add_argument('--random_pos', action='store_true')
    parser.add_argument('--render', action='store_true', help='enable pygame render')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metrics_csv', default='log/train_dqn_metrics.csv')
    parser.add_argument('--summary_json', default='log/train_dqn_summary.json')
    parser.add_argument('--model_dir', default='model/simple')
    parser.add_argument('--resume', default=None, help='checkpoint path to resume from')
    parser.add_argument('--fresh_start', action='store_true', help='start from random init even if model.pkl exists')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    blue_agent = build_opponent(args.opponent)
    blue_obs_ind = blue_agent.get_obs_ind()

    env = Environment(
        args.map,
        'simple',
        blue_obs_ind,
        max_step=args.max_step,
        render=args.render,
        random_pos=args.random_pos,
    )
    size_x, size_y = env.get_map_size()
    _, red_fighter_num, _, blue_fighter_num = env.get_unit_num()
    blue_agent.set_map_info(size_x, size_y, 0, blue_fighter_num)

    red_detector_action = []
    fighter_model = dqn.RLFighter(
        ACTION_NUM,
        learning_rate=args.lr,
        reward_decay=args.gamma,
        e_greedy=args.epsilon,
        replace_target_iter=args.target_replace_iter,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        e_greedy_increment=args.epsilon_increment,
        model_dir=args.model_dir,
    )
    min_replay_size = max(1, min(args.min_replay_size, args.memory_size))

    resume_path = args.resume
    if (resume_path is None) and (not args.fresh_start):
        auto_resume = os.path.join(args.model_dir, 'model.pkl')
        if os.path.exists(auto_resume):
            resume_path = auto_resume

    if resume_path:
        fighter_model.load(resume_path)
        # Keep target network aligned when resuming.
        fighter_model.target_net.load_state_dict(fighter_model.eval_net.state_dict())
        fighter_model.learn_step_counter = infer_resume_step(resume_path, args.model_dir)
        print(f'Resume training from: {resume_path} (learn_step_counter={fighter_model.learn_step_counter})')
    else:
        print('Start training from scratch.')

    with open(args.metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'steps', 'total_reward', 'red_round_reward', 'blue_round_reward', 'red_win', 'learn_calls', 'elapsed_sec'])

        train_start = time.time()
        for epoch in range(1, args.epochs + 1):
            env.reset()
            step_cnt = 0
            total_reward = 0.0
            learn_calls = 0
            epoch_start = time.time()
            red_obs_dict, blue_obs_dict = env.get_obs()

            while True:
                red_fighter_action = np.zeros((red_fighter_num, 4), dtype=np.int32)
                for idx in range(red_fighter_num):
                    radar_point, disturb_point = get_support_action(step_cnt, idx)
                    red_fighter_action[idx][1] = radar_point
                    red_fighter_action[idx][2] = disturb_point
                alive_indices = []
                alive_img_obs = []
                alive_info_obs = []

                blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)

                for idx in range(red_fighter_num):
                    if red_obs_dict['fighter'][idx]['alive']:
                        tmp_img_obs = red_obs_dict['fighter'][idx]['screen'].transpose(2, 0, 1)
                        tmp_info_obs = red_obs_dict['fighter'][idx]['info']

                        alive_indices.append(idx)
                        alive_img_obs.append(tmp_img_obs)
                        alive_info_obs.append(tmp_info_obs)

                if alive_indices:
                    batch_actions = fighter_model.choose_action_batch(alive_img_obs, alive_info_obs)
                    for local_i, idx in enumerate(alive_indices):
                        act = int(batch_actions[local_i])
                        red_fighter_action[idx][0] = int(360 / COURSE_NUM * int(act / ATTACK_IND_NUM))
                        red_fighter_action[idx][3] = int(act % ATTACK_IND_NUM)

                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

                _, red_fighter_reward, red_game_reward, _, _, blue_game_reward = env.get_reward()
                fighter_reward = red_fighter_reward + red_game_reward
                total_reward += float(fighter_reward.sum())
                done = env.get_done()

                next_red_obs_dict, next_blue_obs_dict = env.get_obs()
                for local_i, idx in enumerate(alive_indices):
                    next_img_obs = next_red_obs_dict['fighter'][idx]['screen'].transpose(2, 0, 1)
                    next_info_obs = next_red_obs_dict['fighter'][idx]['info']
                    next_alive = bool(next_red_obs_dict['fighter'][idx]['alive'])
                    fighter_model.store_transition(
                        {'screen': alive_img_obs[local_i], 'info': alive_info_obs[local_i]},
                        int(batch_actions[local_i]),
                        fighter_reward[idx],
                        {'screen': next_img_obs, 'info': next_info_obs},
                        done=(done or (not next_alive)),
                    )

                if done:
                    if fighter_model.memory_counter >= min_replay_size:
                        fighter_model.learn()
                        learn_calls += 1
                    elapsed = time.time() - epoch_start
                    red_win = int(red_game_reward > blue_game_reward)
                    writer.writerow([epoch, step_cnt + 1, total_reward, int(red_game_reward), int(blue_game_reward), red_win, learn_calls, round(elapsed, 3)])
                    f.flush()
                    print(
                        f'[Epoch {epoch}/{args.epochs}] '
                        f'steps={step_cnt + 1} total_reward={int(total_reward)} '
                        f'round_reward=({int(red_game_reward)},{int(blue_game_reward)}) '
                        f'learn_calls={learn_calls}'
                    )
                    break

                if step_cnt > 0 and step_cnt % args.learn_interval == 0 and fighter_model.memory_counter >= min_replay_size:
                    fighter_model.learn()
                    learn_calls += 1

                red_obs_dict, blue_obs_dict = next_red_obs_dict, next_blue_obs_dict
                step_cnt += 1

        latest_model = ensure_latest_checkpoint_link(args.model_dir)
        summary = {
            'epochs': args.epochs,
            'map': args.map,
            'opponent': args.opponent,
            'seed': args.seed,
            'metrics_csv': args.metrics_csv,
            'latest_checkpoint': latest_model,
            'fixed_model_path': os.path.join(args.model_dir, 'model.pkl') if latest_model else None,
            'elapsed_sec': round(time.time() - train_start, 3),
        }
        with open(args.summary_json, 'w') as sf:
            json.dump(summary, sf, indent=2)

    print('\nTraining finished.')
    print('Metrics:', args.metrics_csv)
    print('Summary:', args.summary_json)
    if latest_model:
        print('Latest checkpoint:', latest_model)
        print('Synced model:', os.path.join(args.model_dir, 'model.pkl'))
    else:
        print('Warning: no model_*.pkl checkpoint produced.')


if __name__ == '__main__':
    main()
