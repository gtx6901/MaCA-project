from agent.base_agent import BaseAgent
from collections import Counter
import random
import numpy as np
import copy
import time

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
    # def get_action(self, obs_dict, step_cnt):  #主调用函数
    #     detector_action = []   #输入为观测字典obs_dict及时间步长step_cnt，输出为各单位的动作列表
    #     fighter_action = []
    #     if step_cnt==1:
    #         info_obs=obs_dict['fighter'][0]['info']
    #         self_pos_x=info_obs[3]
    #         if self_pos_x>500:
    #             self.side=2
    #     return detector_action,fighter_action
    def rp_to_xy(self, dst, angle):
        x = dst*np.cos(angle*np.pi/180)
        y = dst*np.sin(angle*np.pi/180)
        return x,y

    def angle_limit(self, angle, zero=0):
        while angle < zero:
            angle += 360
        while angle >= zero+360:
            angle -= 360
        return angle

    agent_leader = [2, 2, 2, 2, 2, 7, 7, 7, 7, 7]
    agent_id_group = [3, 1, 0, 2 ,4, 3, 1, 0, 2, 4]
    def find_leader(self, agent_id):
        path = []
        while self.agent_leader[agent_id] != agent_id:
            path.append(agent_id)
            agent_id = self.agent_leader[agent_id]
        for i in path:
            self.agent_leader[i] = agent_id
        return agent_id

    def id_init(self, leader_id):
        count = 0
        count_id = 0
        new_leader = leader_id
        for i in range(self.fighter_num):
            if self.find_leader(i) == self.find_leader(leader_id) and self.agent_id_group[i] >= 0:
                count += 1
        for i in range(self.fighter_num):
            if self.find_leader(i) == self.find_leader(leader_id) and self.agent_id_group[i] >= 0:
                count_id += 1
                if count_id > count / 2:
                    new_leader = i
                    break
        self.agent_leader[leader_id] = new_leader
        self.agent_leader[new_leader] = new_leader
        leader_id = new_leader

        temp = 0
        for i in range(leader_id,self.fighter_num):
            if self.find_leader(i) == leader_id and self.agent_id_group[i] >= 0:
                self.agent_id_group[i] = temp
                temp += 2
        temp = 1
        for i in range(leader_id-1,-1,-1):
            if self.find_leader(i) == leader_id and self.agent_id_group[i] >= 0:
                self.agent_id_group[i] = temp
                temp += 2

    def id_swap(self, leader_id):
        for i in range(0,self.fighter_num,1):
            if self.find_leader(i) == leader_id and i != leader_id:
                if self.agent_id_group[i] % 2 == 0:
                    self.agent_id_group[i] -= 1
                else:
                    self.agent_id_group[i] += 1

    def change_leader(self, leader_id, change_state):
        new_leader= leader_id

        # 领导者死亡
        if change_state == 'death':
            count = [0, 0]
            for i in range(self.fighter_num):
                if self.agent_id_group[i] > 0:
                    if self.find_leader(i) == leader_id:
                        count[self.agent_id_group[i] % 2] += 1
            if count[1] > count[0]:
                for i in range(self.fighter_num):
                    if self.agent_id_group[i] > 0:
                        if self.find_leader(i) == leader_id:
                            if self.agent_id_group[i] == 1:
                                new_leader = i
                            elif self.agent_id_group[i] % 2 == 1:
                               self.agent_id_group[i] -= 2
            else:
                for i in range(self.fighter_num):
                    if self.agent_id_group[i] > 0:
                        if self.find_leader(i) == leader_id:
                            if self.agent_id_group[i] == 2:
                                new_leader = i
                            elif self.agent_id_group[i] % 2 == 0:
                                self.agent_id_group[i] -= 2
            self.agent_id_group[leader_id] = -1
            print("领导者", leader_id,"被击毁，新领导者为",new_leader)

        # 队伍人数不足
        elif change_state == 'merge':
            for i in range(self.fighter_num):
                if self.agent_id_group[i] == 0 and self.find_leader(i) != leader_id:
                    new_leader= self.find_leader(i)
            print("编队整合：", leader_id,"加入",new_leader)

        self.agent_leader[leader_id] = new_leader
        self.agent_leader[new_leader] = new_leader
        self.agent_id_group[new_leader] = 0


    def member_num_check(self, leader_id,fighter_list):
        member_count = 0
        left_flag = -1
        direct = 0
        for i in range(self.fighter_num):
            if self.agent_id_group[i] == 0 and self.find_leader(i) == leader_id:
                left_flag = 1
            if fighter_list[i]['alive'] == 0 and self.find_leader(i) == leader_id:
                if self.agent_id_group[i] > 0:
                    direct = 100 * left_flag
                    print("跟随者",i,"坠毁")
                self.agent_id_group[i] = -1
            if self.find_leader(i) == leader_id and self.agent_id_group[i] >= 0:
                member_count += 1
        print("领导者",leader_id,"的编队存活数为",member_count)
        if member_count < 4:
            self.change_leader(leader_id, 'merge')
            self.id_init(self.find_leader(leader_id))
        print(self.agent_id_group)
        return direct

    move_action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    def get_move_actions(self, obs_dict, step_cnt, agent_id, fighter_num=10):
        self_pos_x=obs_dict['fighter'][agent_id]['info'][3]
        self_pos_y=obs_dict['fighter'][agent_id]['info'][4]

        # 索敌
        visible_list = obs_dict['fighter'][agent_id]['visible']
        visible_total_list = []
        closest_enemy = [1500, 0, 0]
        for z in range(fighter_num):
            if len(obs_dict['fighter'][z]['visible']) > 0:
                for n in range(len(obs_dict['fighter'][z]['visible'])):
                    visible_total_list.append(obs_dict['fighter'][z]['visible'][n]['id'])
        for x in range(len(visible_list)):
            enemy_pos_x = visible_list[x]['pos_x']
            enemy_pos_y = visible_list[x]['pos_y']
            distance = ((self_pos_x - enemy_pos_x) ** 2 + (enemy_pos_y - self_pos_y) ** 2) ** 0.5
            if distance < closest_enemy[0]:
                closest_enemy[0] = distance
                closest_enemy[1] = enemy_pos_x
                closest_enemy[2] = enemy_pos_y

        # 追击模式
        if len(visible_list) > 0:
            enemy_delta_x = closest_enemy[1] - self_pos_x
            enemy_delta_y = closest_enemy[2] - self_pos_y
            if enemy_delta_x == 0:
                enemy_angle = 90
            else:
                enemy_angle: int = abs(np.arctan(enemy_delta_y / enemy_delta_x) * 180 / np.pi)
            if enemy_delta_x >= 0 and enemy_delta_y >= 0:
                self.move_action[agent_id] = enemy_angle
            elif enemy_delta_x <= 0 <= enemy_delta_y:
                self.move_action[agent_id] = 180 - enemy_angle
            elif enemy_delta_x >= 0 >= enemy_delta_y:
                self.move_action[agent_id] = -enemy_angle
            elif enemy_delta_x <= 0 and enemy_delta_y <= 0:
                self.move_action[agent_id] = 180 + enemy_angle
            agent_in_pos = 0

        # 巡航模式
            #领导者
        elif self.find_leader(agent_id) == agent_id:
            agent_in_pos = 1
            if step_cnt == 1:
                self.id_init(agent_id)
                if self.side == 1:
                    self.move_action[agent_id] = 0
                else:
                    self.move_action[agent_id] = 180
            else:
                if self_pos_x > 950 or self_pos_x < 50:
                    self.move_action[agent_id] = 180 - self.move_action[agent_id]
                    self.id_swap(agent_id)
                if self_pos_y > 950 or self_pos_y < 50:
                    self.move_action[agent_id] = - self.move_action[agent_id]
                    self.id_swap(agent_id)
            if step_cnt % 50 == 0:
                direct = self.member_num_check(agent_id,obs_dict['fighter'])
                random.seed(agent_id+time.time())
                self.move_action[agent_id] += direct + 30 * random.randint(-1,1)

            #跟随者
        else:
            if obs_dict['fighter'][self.find_leader(agent_id)]['alive'] == 0:
                self.change_leader(self.find_leader(agent_id),'death')
                self.id_init(self.find_leader(agent_id))
            leader_delta_x = obs_dict['fighter'][self.find_leader(agent_id)]['info'][3]-self_pos_x
            leader_delta_y = obs_dict['fighter'][self.find_leader(agent_id)]['info'][4]-self_pos_y
            target_x,target_y = self.rp_to_xy(50*int((self.agent_id_group[agent_id]+1)/2),self.move_action[self.find_leader(agent_id)]+110*(-1)**self.agent_id_group[agent_id])
            target_x = int(target_x) + leader_delta_x
            target_y = int(target_y) + leader_delta_y

            if ((self_pos_x-target_x)**2+(self_pos_y-target_y)**2)**0.5<2:
                agent_in_pos=1
            else:
                agent_in_pos=0

            if target_x == 0:
                leader_angle = 90
            else:
                leader_angle:int = abs(np.arctan(target_y/target_x)*180/np.pi)
            if target_x >= 0 and target_y >= 0:
                self.move_action[agent_id] = leader_angle
            elif target_x <= 0 <= target_y:
                self.move_action[agent_id] = 180 - leader_angle
            elif target_x >= 0 >= target_y:
                self.move_action[agent_id] = -leader_angle
            elif target_x <= 0 and target_y <= 0:
                self.move_action[agent_id] = 180 + leader_angle

        # 到位时，随领导者前进；未到位时，按自己方向前进
        if agent_in_pos == 1:
            return self.angle_limit(self.move_action[self.find_leader(agent_id)])
        else:
            return self.angle_limit(self.move_action[agent_id])

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

    last_attack = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    cool_down = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    def get_attack_actions(self,obs_dict,info_obs,agent_id,fighter_num=10):
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
            if self.last_attack[agent_id] == enemy_id and self.cool_down[agent_id] < 0:
                self.cool_down[agent_id] += 1
            else:
                self.last_attack[agent_id] = enemy_id
                self.cool_down[agent_id] = 0
                if (distance <= 50) and (s_missile_left > 0):
                    return enemy_id + 10
                elif (distance <= 120) and (l_missile_left > 0):
                    return enemy_id
        return attack_id

    def get_action( self, obs_dict, step_cnt):
        detector_action = []
        fighter_action = []
        if step_cnt == 1:
            info_obs = obs_dict['fighter'][0]['info']
            self_pos_x = info_obs[3]
            if self_pos_x >500:
                self.side = 2
            self.agent_leader = [2, 2, 2, 2, 2, 7, 7, 7, 7, 7]
            self.agent_id_group = [3, 1, 0, 2 ,4, 3, 1, 0, 2, 4]
        for y in range(10):
            true_action = np.array([0,0,11,0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                tmp_info_obs = obs_dict['fighter'][y]['info']
                move_action = self.get_move_actions(obs_dict,step_cnt,y)
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