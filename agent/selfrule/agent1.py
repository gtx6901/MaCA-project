from agent.base_agent import BaseAgent
from collections import Counter
import random
import numpy as np
import copy
import math

class Agent(BaseAgent):


    def __init__(self):      #初始化接口
        BaseAgent.__init__(self)   #从 BaseAgent 继承
        self.obs_ind = 'selfconstruct4' #默认观测的文件名称
        self.side = 1 #红方
        self.direction = [1]*10
        self.last_change_direction_step = [1] * 10
        self.offset = np.random.random(10)
        self.last_attack_info = [[1]*10]*10
        self.last_visible_total_list = []

    def set_init(self,obs_dict):
        info_obs = obs_dict['fighter'][0]['info']
        self_pos_x = info_obs[3]
        self_pos_y = info_obs[4]
        self.direction = [1] * 10
        self.last_change_direction_step = [1] * 10
        self.last_visible_total_list = []
        if self_pos_x > 500:
            self.side = 2
    def set_map_info(self, size_x, size_y, detector_num,fighter_num) :
        self.size_x = 1000  #同构智能体地图
        self.size_y = 1000
        self.detector_num = detector_num
        self.fighter_num = 10
    # def get_action(self, obs_dict, step_cnt):  #主调用函数
    #     detector_action = []   #输入为观测字典obs_dict及时间步长step_cnt，输出为各单位的动作列表
    #     fighter_action = []
    #     if step_cnt==1:
    #         self.direction = [1] * 10
    #         info_obs=obs_dict['fighter'][0]['info']
    #         self_pos_x=info_obs[3]
    #         if self_pos_x>500:
    #             self.side=2
    #     return detector_action,fighter_action

    def get_move_actions(self,step_cnt,obs_dict,agent_id):
        '''处理可见的敌方信息'''
        visible_total_list = []
        visible_list = obs_dict['fighter'][agent_id]['visible']
        for z in range(self.fighter_num):
            if  obs_dict['fighter'][z]["alive"] and len(obs_dict['fighter'][z]['visible']) > 0 :
                for n in range(len(obs_dict['fighter'][z]['visible'])):
                    visible_total_list.append(obs_dict['fighter'][z]['visible'][n])

        info_obs = obs_dict['fighter'][agent_id]['info']
        self_pos_x = info_obs[3]
        self_pos_y = info_obs[4]
        min_dist = 1000
        min_dist_enemy_id = 0
        if  len(visible_total_list) >=1:
            for i in range(len(visible_total_list)):
                dist = math.sqrt(pow(visible_total_list[i]["pos_y"] - self_pos_y, 2) + pow(
                    visible_total_list[i]["pos_x"] - self_pos_x, 2))
                if dist < min_dist:
                    min_dist = dist
                    min_dist_enemy_id = i

        engage = (len(visible_total_list) >=1 and min_dist < 400)

        if step_cnt < 30:
            move_action = 150
            '''测试编队'''
            '''
            if  agent_id not in self.leader_list :
                temp_list = [obs_dict['fighter'][1]['info'],obs_dict['fighter'][4]['info'],obs_dict['fighter'][7]['info']]
                temp_dist = 1000
                nearest_leader = obs_dict['fighter'][1]['info']
                for i in temp_list:
                    print(i[3])
                    dist = math.sqrt((i[3]-self_pos_x)**2 + (i[4]-self_pos_y)**2)
                    if dist < temp_dist:
                        temp_dist = dist
                        nearest_leader = i

                move_action = 180 / 3.14 * math.atan2(nearest_leader[4] - self_pos_y,
                                                      (nearest_leader[3]-2) - self_pos_x)
            else:
                move_action = 0
            '''

        else:
            if (self.side == 1 and self.direction[agent_id] == 1) or (self.side == 2 and self.direction[agent_id] == -1):
                if engage:
                    move_action = 180 / 3.14 * math.atan2(visible_total_list[min_dist_enemy_id]["pos_y"] - self_pos_y, visible_total_list[min_dist_enemy_id]["pos_x"] - self_pos_x)
                else:
                    move_action = 0 + 15 * math.sin(step_cnt / 1 + self.offset[agent_id])
            else:
                if engage:
                    move_action = 180 / 3.14 * math.atan2(visible_total_list[min_dist_enemy_id]["pos_y"] - self_pos_y,
                                                          visible_total_list[min_dist_enemy_id]["pos_x"] - self_pos_x)
                else:
                    move_action = 180 + 15 * math.sin(step_cnt / 1 + self.offset[agent_id])

        #更改方向
        if step_cnt >= 25 and (step_cnt - self.last_change_direction_step[agent_id] > 25) and (self_pos_x > 950 or self_pos_x < 50):
            self.direction[agent_id] *= -1
            self.last_change_direction_step[agent_id] = step_cnt
        return move_action

    def get_radar_point(self, radar_point_num):
        radar_point = random.randint(1,radar_point_num)
        return radar_point

    def get_disturb_point(self, obs_dict,agent_id):
        disturb_point = 11
        if len(obs_dict['fighter'][agent_id]['radar']) == 1:
            disturb_point = obs_dict['fighter'][agent_id]['radar'][0]['r_fp']
        elif len(obs_dict['fighter'][agent_id]['radar']) > 1:
            disturb_point_list = []
            for x in range(len(obs_dict['fighter'][agent_id]['radar'])):
                disturb_point_list.append(obs_dict['fighter'][agent_id]['radar'][x]['r_fp'])
            disturb_counter_list = Counter(disturb_point_list).most_common(1)
            if float(disturb_counter_list[0][1]) * 2 > float(len(obs_dict['fighter'][agent_id]['radar'])):
                disturb_point = disturb_counter_list[0][0]
        return disturb_point

    def get_attack_actions(self,obs_dict,info_obs,step_cnt,agent_id,fighter_num=10):
        visible_list = obs_dict['fighter'][agent_id]['visible']
        visible_total_list = []
        attack_id = 0
        for z in range(fighter_num):
            if len(obs_dict['fighter'][z]['visible']) > 0:
                for n in range(len(obs_dict['fighter'][z]['visible'])):
                    visible_total_list.append(obs_dict['fighter'][z]['visible'][n]['id'])
        self_pos_x = info_obs[3]
        self_pos_y = info_obs[4]
        for x in range(len(visible_list)):
            enemy_id = visible_list[x]['id']
            enemy_pos_x = visible_list[x]['pos_x']
            enemy_pos_y = visible_list[x]['pos_y']
            distance = ((self_pos_x - enemy_pos_x) ** 2 + (enemy_pos_y - self_pos_y) ** 2) ** 0.5
            l_missile_left = info_obs[1]
            s_missile_left = info_obs[2]
            if (distance <= 50) and (s_missile_left > 0) :
                self.last_attack_info[agent_id][x] = step_cnt
                return enemy_id + 10
            elif (distance <= 120) and (l_missile_left > 0) :
                self.last_attack_info[agent_id][x] = step_cnt
                return enemy_id

        return attack_id

    def get_action( self, obs_dict, step_cnt):
        detector_action = []
        fighter_action = []
        if step_cnt == 1:
            self.set_init(obs_dict)
        for y in range(10):
            true_action = np.array([0,0,11,0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                tmp_info_obs = obs_dict['fighter'][y]['info']
                move_action = self.get_move_actions(step_cnt,obs_dict,y)
                radar_point = self.get_radar_point(10)
                disturb_point = self.get_disturb_point(obs_dict, y)
                attack_action = self.get_attack_actions(obs_dict,tmp_info_obs,step_cnt,y)
                true_action[0] = move_action
                true_action[1] = radar_point
                true_action[2] = disturb_point #disturb_point
                true_action[3] = attack_action
            fighter_action.append(copy.deepcopy (true_action))
        fighter_action = np.array(fighter_action)
        return detector_action,fighter_action