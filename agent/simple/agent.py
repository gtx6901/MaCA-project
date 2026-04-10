#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: agent.py
@time: 2018/3/13 0013 10:51
@desc: rule based agent
"""

import os
import glob
from agent.base_agent import BaseAgent
from agent.simple import dqn
from fighter_action_utils import get_support_action
import interface
from world import config
import copy
import random
import numpy as np

DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1 # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM


class Agent(BaseAgent):
    def __init__(self):
        """
        Init this agent
        :param size_x: battlefield horizontal size
        :param size_y: battlefield vertical size
        :param detector_num: detector quantity of this side
        :param fighter_num: fighter quantity of this side
        """
        BaseAgent.__init__(self)
        self.obs_ind = 'simple'
        model_path = self._resolve_model_path()
        self.fighter_model = dqn.RLFighter(ACTION_NUM, model_path=model_path)

    @staticmethod
    def _resolve_model_path():
        model_dir = 'model/simple'
        fixed_model = os.path.join(model_dir, 'model.pkl')
        if os.path.exists(fixed_model):
            return fixed_model

        candidates = sorted(glob.glob(os.path.join(model_dir, 'model_*.pkl')))
        if candidates:
            latest = candidates[-1]
            print('Warning: model/simple/model.pkl not found, using latest checkpoint:', latest)
            return latest

        print('Error: no model checkpoint found under model/simple/')
        exit(1)

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    def __reset(self):
        pass

    def get_action(self, obs_dict, step_cnt):
        """
        get actions
        :param detector_obs_list:
        :param fighter_obs_list:
        :param joint_obs_dict:
        :param step_cnt:
        :return:
        """

        detector_action = []
        fighter_action = []
        for y in range(self.fighter_num):
            radar_point, disturb_point = get_support_action(step_cnt, y)
            true_action = np.array([0, radar_point, disturb_point, 0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                tmp_img_obs = obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = obs_dict['fighter'][y]['info']
                tmp_action = self.fighter_model.choose_action(tmp_img_obs, tmp_info_obs)
                # action formation
                true_action[0] = int(360 / COURSE_NUM * int(tmp_action[0] / ATTACK_IND_NUM))
                true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM)
            fighter_action.append(copy.deepcopy(true_action))
        fighter_action = np.array(fighter_action)

        return detector_action, fighter_action

