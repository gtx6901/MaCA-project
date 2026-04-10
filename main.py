#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: main.py
@time: 2018/7/25 0025 10:01
@desc: 
"""

import copy
import numpy as np
import torch.cuda
import random

from interface import Environment
import dqn
#from agent.selfrule.agent2 import Agent
from agent.fix_rule_no_att.agent import Agent
from collections import Counter
MAP_PATH = 'maps/1000_1000_fighter10v10.map'

RENDER = True
MAX_EPOCH = 1000
BATCH_SIZE = 320
LR = 0.007 #0.01                   # learning rate
EPSILON = 1              # greedy policy
GAMMA = 0.8      #0.9           # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1 # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM
LEARN_INTERVAL = 150

if __name__ == "__main__":
    # create blue agent
    blue_agent = Agent()
    # get agent obs type
    red_agent_obs_ind = 'simple'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    max_step = 1500
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind,max_step=max_step, render=RENDER)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    fighter_model = dqn.RLFighter(ACTION_NUM,LR,GAMMA,EPSILON,TARGET_REPLACE_ITER,BATCH_SIZE)
    # execution
    for x in range(MAX_EPOCH):
        total_reward = 0
        step_cnt = 0
        env.reset()
        while True:
            obs_list = []
            action_list = []
            red_fighter_action = []
            # get obs
            red_obs_dict, blue_obs_dict = env.get_obs()
            # get action
            # get blue action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)
            # get red action
            obs_got_ind = [False] * red_fighter_num
            for y in range(red_fighter_num):
                #探测
                radar_point = random.randint(1, 10)
                #干扰
                obs_dict = red_obs_dict
                disturb_point = 11
                # if len(obs_dict['fighter'][y]['radar']) == 1:
                #     disturb_point = obs_dict['fighter'][y]['radar'][0]['r_fp']
                # elif len(obs_dict['fighter'][y]['radar']) > 1:
                #     disturb_point_list = []
                #     for x in range(len(obs_dict['fighter'][y]['radar'])):
                #         disturb_point_list.append(obs_dict['fighter'][y]['radar'][x]['r_fp'])
                #     disturb_counter_list = Counter(disturb_point_list).most_common(1)
                #     if float(disturb_counter_list[0][1]) * 2 > float(len(obs_dict['fighter'][y]['radar'])):
                #         disturb_point = disturb_counter_list[0][0]


                true_action = np.array([0, radar_point, disturb_point, 0], dtype=np.int32)

                if red_obs_dict['fighter'][y]['alive']:
                    obs_got_ind[y] = True
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    tmp_action = fighter_model.choose_action(tmp_img_obs, tmp_info_obs)
                    #tmp_action = fighter_model.choose_action(tmp_info_obs)
                    obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                    action_list.append(tmp_action)
                    # action formation
                    true_action[0] = int(360 / COURSE_NUM * int(tmp_action[0] / ATTACK_IND_NUM))
                    true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM)
                else:
                    obs_list.append(0)
                    action_list.append(0)
                red_fighter_action.append(true_action)
            red_fighter_action = np.array(red_fighter_action)
            # step
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward + red_game_reward
            fighter_reward = red_fighter_reward + red_game_reward
            # save replay
            red_obs_dict, blue_obs_dict = env.get_obs()
            total_reward += fighter_reward.sum()
            for y in range(red_fighter_num):
                if obs_got_ind[y]:
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    #print(tmp_info_obs)
                    fighter_model.store_transition(obs_list[y], action_list[y], fighter_reward[y],
                                                   {'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})

            # if done, perform a learn
            if env.get_done():
                # detector_model.learn()
                fighter_model.learn()
                break
            # if not done learn when learn interval
            if (step_cnt > 0) and (step_cnt % LEARN_INTERVAL == 0):
                # detector_model.learn()
                fighter_model.learn()
            step_cnt += 1
        print(total_reward)
