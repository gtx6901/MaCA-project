from agent.base_agent import BaseAgent
from collections import Counter
import random
import numpy as np
import copy

class Agent(BaseAgent):
    def __init__(self):      #初始化接口
        BaseAgent.__init__(self)   #从 BaseAgent 继承
        self.obs_ind = 'selfconstruct4' #默认观测的文件名称
        self.side = 1 #红方
    def set_map_info(self, size_x, size_y, detector_num,fighter_num) :
        self.size_x = 1000  #同构智能体地图
        self.size_y = 1000
        self.detector_num = detector_num
        self.fighter_num = 10
    def get_action(self, obs_dict, step_cnt):  #主调用函数
        detector_action = []   #输入为观测字典obs_dict及时间步长step_cnt，输出为各单位的动作列表
        fighter_action = []
        if step_cnt==1:
            info_obs=obs_dict['fighter'][0]['info']
            self_pos_x=info_obs[3]
            if self_pos_x>500:
                self.side=2
        return detector_action,fighter_action

    def get_move_actions(self):
        if self.side == 1:
            move_action = 0
        else:
            move_action = 180
        return move_action
    def get_radar_point(self, radar_point_num):
        radar_point = random.randint(1,radar_point_num)
        return radar_point
    def get_disturb_point(self, obs_dict,agent_id):
        disturb_point = 11
        if len(obs_dict[ 'fighter ' ][agent_id][ 'radar']) == 1:
            disturb_point = obs_dict[ 'fighter ' ][agent_id] [ ' radar'][0][ 'r_fp']
        elif len(obs_dict[ 'fighter ' ][ agent_id][ 'radar ']) >1:
            disturb_point_list = []
            for x in range(len(obs_dict[ 'fighter ' ][agent_id] [ ' radar '])):
                disturb_point_list.append(obs_dict[ 'fighter ' ][agent_id]['radar'][x][ 'r_fp'])
            disturb_counter_list = Counter(disturb_point_list).most_common(1)
            if float(disturb_counter_list[0][1])*2 > float(len(obs_dict[ ' fighter'][agent_id][ ' radar'])):
                disturb_point = disturb_counter_list[0][0]
        return disturb_point
    def get_attack_actions(self,obs_dict,info_obs,agent_id,fighter_num=10):
        visible_list = obs_dict[ 'fighter ' ][agent_id][ 'visible' ]
        visible_total_list =[]
        attack_id = 0
        for z in range(fighter_num) :
            if len(obs_dict[ 'fighter '][z][ 'visible '])>0:
                for n in range(len(obs_dict[ 'fighter '][z][ 'visible' ])):
                    visible_total_list.append(obs_dict[ 'fighter '][z][ 'visible'][n][ 'id'])
        self_pos_x = info_obs[3]
        self_pos_y = info_obs[4]
        for x in range(len(visible_list)):
            enemy_id = visible_list[x]['id']
            enemy_pos_x = visible_list[x][ ' pos_x ']
            enemy_pos_y = visible_list[x][ ' pos_y']
            distance = ((self_pos_x - enemy_pos_x)** 2 + (enemy_pos_y - self_pos_y)**2)**0.5
            l_missile_left = info_obs[1]
            s_missile_left = info_obs[2]
            if (distance <= 50) and (s_missile_left > 0):
                return enemy_id + 10
            elif (distance <= 120) and (l_missile_left > 0):
                return enemy_id
        return attack_id
    def get_action( self, obs_dict, step_cnt):
        detector_action = []
        fighter_action = []
        if step_cnt == 1:
            info_obs = obs_dict[ 'fighter ' ][0][ 'info ']
            self_pos_x = info_obs[3]
            if self_pos_x >500:
                self.side = 2
        for y in range(10):
            true_action = np.array([0,0,11,0], dtype=np.int32)
            if obs_dict[ 'fighter ' ][y][ 'alive ' ]:
                tmp_info_obs = obs_dict[ 'fighter'][y][ 'info ']
                move_action = self.get_move_actions()
                radar_point = self.get_radar_point(10)
                disturb_point = self.get_disturb_point(obs_dict, y)
                attack_action = self.get_attack_actions(obs_dict,tmp_info_obs,y)
                true_action[0] = move_action
                true_action[1] = radar_point
                true_action[2] = disturb_point
                true_action[3] = attack_action
            fighter_action.append(copy.deepcopy (true_action))
        fighter_action = np.array(fighter_action)
        return detector_action,fighter_action

