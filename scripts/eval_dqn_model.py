#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
import importlib
import json
import os
import random
import time

import numpy as np
import torch

from interface import Environment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_agent(agent_name):
    module = importlib.import_module(f'agent.{agent_name}.agent')
    return module.Agent()


def resolve_map_path(map_arg):
    if map_arg.endswith('.map'):
        return map_arg
    return os.path.join('maps', f'{map_arg}.map')


def alive_count(obs_raw):
    detector_alive = sum(1 for item in obs_raw['detector_obs_list'] if item['alive'])
    fighter_alive = sum(1 for item in obs_raw['fighter_obs_list'] if item['alive'])
    return detector_alive, fighter_alive


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation for MaCA agents.')
    parser.add_argument('--map', default='1000_1000_fighter10v10')
    parser.add_argument('--agent1', default='simple', help='left side (red) agent folder under agent/')
    parser.add_argument('--agent2', default='fix_rule_no_att', help='right side (blue) agent folder under agent/')
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--max_step', type=int, default=1500)
    parser.add_argument('--random_pos', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metrics_csv', default='log/eval_metrics.csv')
    parser.add_argument('--summary_json', default='log/eval_summary.json')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)

    map_path = resolve_map_path(args.map)
    if not os.path.exists(map_path):
        raise FileNotFoundError(f'map not found: {map_path}')

    agent1 = build_agent(args.agent1)
    agent2 = build_agent(args.agent2)

    env = Environment(
        map_path,
        agent1.get_obs_ind(),
        agent2.get_obs_ind(),
        max_step=args.max_step,
        render=args.render,
        random_pos=args.random_pos,
    )

    size_x, size_y = env.get_map_size()
    side1_detector_num, side1_fighter_num, side2_detector_num, side2_fighter_num = env.get_unit_num()
    agent1.set_map_info(size_x, size_y, side1_detector_num, side1_fighter_num)
    agent2.set_map_info(size_x, size_y, side2_detector_num, side2_fighter_num)

    side1_win = 0
    side2_win = 0
    draw = 0
    total_steps = 0
    total_s1_step_reward = 0.0
    total_s2_step_reward = 0.0
    total_s1_det_alive = 0
    total_s1_fig_alive = 0
    total_s2_det_alive = 0
    total_s2_fig_alive = 0

    start = time.time()
    with open(args.metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'round', 'steps',
            'side1_step_reward', 'side2_step_reward',
            'side1_round_reward', 'side2_round_reward',
            'winner',
            'side1_alive_detector', 'side1_alive_fighter',
            'side2_alive_detector', 'side2_alive_fighter',
        ])

        for round_idx in range(1, args.rounds + 1):
            if round_idx > 1:
                env.reset()

            step_cnt = 0
            side1_total_reward = 0.0
            side2_total_reward = 0.0
            side1_round_reward = 0
            side2_round_reward = 0

            while True:
                side1_obs, side2_obs = env.get_obs()
                side1_detector_action, side1_fighter_action = agent1.get_action(side1_obs, step_cnt)
                side2_detector_action, side2_fighter_action = agent2.get_action(side2_obs, step_cnt)

                env.step(side1_detector_action, side1_fighter_action, side2_detector_action, side2_fighter_action)

                o_detector_reward, o_fighter_reward, o_game_reward, e_detector_reward, e_fighter_reward, e_game_reward = env.get_reward()
                side1_step_reward = float(np.sum(o_detector_reward) + np.sum(o_fighter_reward))
                side2_step_reward = float(np.sum(e_detector_reward) + np.sum(e_fighter_reward))
                side1_total_reward += side1_step_reward
                side2_total_reward += side2_step_reward

                if env.get_done():
                    side1_round_reward = int(o_game_reward)
                    side2_round_reward = int(e_game_reward)
                    if o_game_reward > e_game_reward:
                        winner = 'side1'
                        side1_win += 1
                    elif o_game_reward < e_game_reward:
                        winner = 'side2'
                        side2_win += 1
                    else:
                        winner = 'draw'
                        draw += 1

                    side1_obs_raw, side2_obs_raw = env.get_obs_raw()
                    s1_det_alive, s1_fig_alive = alive_count(side1_obs_raw)
                    s2_det_alive, s2_fig_alive = alive_count(side2_obs_raw)

                    writer.writerow([
                        round_idx, step_cnt + 1,
                        round(side1_total_reward, 3), round(side2_total_reward, 3),
                        side1_round_reward, side2_round_reward,
                        winner,
                        s1_det_alive, s1_fig_alive,
                        s2_det_alive, s2_fig_alive,
                    ])
                    f.flush()

                    print(
                        f'[Round {round_idx}/{args.rounds}] '
                        f'steps={step_cnt + 1} winner={winner} '
                        f'round_reward=({side1_round_reward},{side2_round_reward}) '
                        f'alive=({s1_det_alive},{s1_fig_alive}) vs ({s2_det_alive},{s2_fig_alive})'
                    )

                    total_steps += step_cnt + 1
                    total_s1_step_reward += side1_total_reward
                    total_s2_step_reward += side2_total_reward
                    total_s1_det_alive += s1_det_alive
                    total_s1_fig_alive += s1_fig_alive
                    total_s2_det_alive += s2_det_alive
                    total_s2_fig_alive += s2_fig_alive
                    break

                step_cnt += 1

    rounds = args.rounds
    summary = {
        'map': map_path,
        'agent1': args.agent1,
        'agent2': args.agent2,
        'rounds': rounds,
        'seed': args.seed,
        'max_step': args.max_step,
        'random_pos': args.random_pos,
        'side1_win': side1_win,
        'side2_win': side2_win,
        'draw': draw,
        'side1_win_rate': round(side1_win / rounds, 4),
        'side2_win_rate': round(side2_win / rounds, 4),
        'avg_steps': round(total_steps / rounds, 3),
        'avg_side1_step_reward': round(total_s1_step_reward / rounds, 3),
        'avg_side2_step_reward': round(total_s2_step_reward / rounds, 3),
        'avg_side1_alive_detector': round(total_s1_det_alive / rounds, 3),
        'avg_side1_alive_fighter': round(total_s1_fig_alive / rounds, 3),
        'avg_side2_alive_detector': round(total_s2_det_alive / rounds, 3),
        'avg_side2_alive_fighter': round(total_s2_fig_alive / rounds, 3),
        'metrics_csv': args.metrics_csv,
        'elapsed_sec': round(time.time() - start, 3),
    }

    with open(args.summary_json, 'w') as sf:
        json.dump(summary, sf, indent=2)

    print('\nEvaluation finished.')
    print('Metrics:', args.metrics_csv)
    print('Summary:', args.summary_json)


if __name__ == '__main__':
    main()
