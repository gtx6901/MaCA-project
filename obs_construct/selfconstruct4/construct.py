import numpy as np
import copy
from interface import Environment

class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num
        self.img_obs_reduce_ratio = 10

    def obs_construct(self, obs_raw_dict):
        obs_dict = {}
        detector_obs_list = []
        fighter_obs_list = []
        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']
        detector_img, fighter_img, joint_img = self.__get_img_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        detector_data, fighter_data, radar_data, striking_data, visible_data = self.__get_data_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        alive_status = self.__get_alive_status(detector_data_obs_list, fighter_data_obs_list)
        # o方
        # 预警机
        for x in range(self.detector_num):
            img_context = detector_img[x, :, :, :]
            img_context = np.concatenate((img_context, joint_img[0, :, :, :]), axis=2)
            data_context = detector_data[x, :]
            detector_obs_list.append({'info': copy.deepcopy(data_context), 'screen': copy.deepcopy(img_context),
                             'alive': alive_status[x][0]})
        # 战机
        for x in range(self.fighter_num):
            img_context = fighter_img[x, :, :, :]
            img_context = np.concatenate((img_context, joint_img[0, :, :, :]), axis=2)
            data_context = fighter_data[x, :]
            fighter_obs_list.append({'info': copy.deepcopy(data_context), 'screen': copy.deepcopy(img_context),
                             'alive': alive_status[x + self.detector_num][0], 'radar': radar_data[x], 'strike': striking_data[x], 'visible': visible_data[x]})
        obs_dict['detector'] = detector_obs_list
        obs_dict['fighter'] = fighter_obs_list
        obs_dict['radar'] = radar_data # obs_dict['radar'][0]['r_fp']
        # 全局状态
        """
        state_date = np.zeros((10, 19), dtype=np.int32)
        for x in range(10):
            if fighter_data_obs_list[x]['alive']:
                state_date[x][0] = fighter_data_obs_list[x]['id']
                state_date[x][1] = fighter_data_obs_list[x]['alive']
                state_date[x][2] = fighter_data_obs_list[x]['pos_x']/1000
                state_date[x][3] = fighter_data_obs_list[x]['pos_y']/1000
                state_date[x][4] = fighter_data_obs_list[x]['course']/360
                state_date[x][5] = fighter_data_obs_list[x]['l_missile_left']/10
                state_date[x][6] = fighter_data_obs_list[x]['s_missile_left']/10
                for z in range(len(visible_data[x])):
                    if z <= 2:
                        state_date[x][3 * z + 7] = visible_data[x][z]['id']
                        state_date[x][3 * z + 8] = visible_data[x][z]['pos_x']
                        state_date[x][3 * z + 9] = visible_data[x][z]['pos_y']
                for y in range(len(radar_data[x])):
                    if y <= 2:
                        state_date[x][16 + y] = radar_data[x][y]['id']
                """
        # 全局状态信息 n_state = 14*4 = 56
        state_date = np.zeros((4, 14), dtype=np.float32)
        for x in range(4):
            # 存活单位进行状态记录
            if fighter_data_obs_list[x]['alive']:
                # one_hot形式的id编码
                agents_id = np.zeros(4)
                agents_id[fighter_data_obs_list[x]['id']-1] = 1
                for i in range(4):
                    state_date[x][i] = agents_id[i]
                #state_date[x][0] = fighter_data_obs_list[x]['id']
                state_date[x][4] = fighter_data_obs_list[x]['alive']
                state_date[x][5] = fighter_data_obs_list[x]['pos_x']
                state_date[x][6] = fighter_data_obs_list[x]['pos_y']
                state_date[x][7] = fighter_data_obs_list[x]['course']
                state_date[x][8] = fighter_data_obs_list[x]['l_missile_left']
                state_date[x][9] = fighter_data_obs_list[x]['s_missile_left']
                for z in range(len(visible_data[x])):
                    if z < 2:
                        state_date[x][2 * z + 10] = visible_data[x][z]['pos_x']
                        state_date[x][2 * z + 11] = visible_data[x][z]['pos_y']

        obs_dict['state'] = state_date.flatten()

        return obs_dict

    def __get_alive_status(self,detector_data_obs_list,fighter_data_obs_list):
        alive_status = np.full((self.detector_num+self.fighter_num,1),True)
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                alive_status[x][0] = False
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                alive_status[x+self.detector_num][0] = False
        return alive_status  # 10*1

    def __get_img_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        img_obs_size_x = int(self.battlefield_size_y / self.img_obs_reduce_ratio)
        img_obs_size_y = int(self.battlefield_size_x / self.img_obs_reduce_ratio)
        # 个体img：所有己方单位位置
        detector_img = np.full((self.detector_num, img_obs_size_x, img_obs_size_y, 3), 0, dtype=np.int32)
        fighter_img = np.full((self.fighter_num, img_obs_size_x, img_obs_size_y, 3), 0, dtype=np.int32)
        # 企鹅据img：所有可见敌方单元位置和类型
        joint_img = np.full((1, img_obs_size_x, img_obs_size_y, 2), 0, dtype=np.int32)

        # set all self unit pos, detector: 1, fighter: 2, self: 255
        tmp_pos_obs = np.full((img_obs_size_x, img_obs_size_y), 0, dtype=np.int32)
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                continue
            self.__set_value_in_img(tmp_pos_obs, int(detector_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(detector_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 1)
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                continue
            self.__set_value_in_img(tmp_pos_obs, int(fighter_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(fighter_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 2)
        # Detector obs
        for x in range(self.detector_num):
            # if not alive, skip
            if not detector_data_obs_list[x]['alive']:
                continue
            # self detection target. target: id
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                self.__set_value_in_img(detector_img[x, :, :, 0],
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['id'])
            # self detection target. target: type (detector: 1, fighter 2)
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                self.__set_value_in_img(detector_img[x, :, :, 1],
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
            # friendly pos. self: 255, other: type (detector: 1, fighter 2)
            detector_img[x, :, :, 2] = copy.deepcopy(tmp_pos_obs)
            self.__set_value_in_img(detector_img[x, :, :, 2],
                                    int(detector_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(detector_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 255)

        # Fighter obs
        for x in range(self.fighter_num):
            # if not alive, skip
            if not fighter_data_obs_list[x]['alive']:
                continue
            # self detection target. target: id
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                self.__set_value_in_img(fighter_img[x, :, :, 0],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['id'])
            # self detection target. target: type (detector: 1, fighter 2)
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                self.__set_value_in_img(fighter_img[x, :, :, 1],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
            # friendly pos. self: 255, other: type (detector: 1, fighter 2)
            fighter_img[x, :, :, 2] = copy.deepcopy(tmp_pos_obs)
            self.__set_value_in_img(fighter_img[x, :, :, 2],
                                    int(fighter_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(fighter_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 255)

        # Global obs
        # Passive detection
        for x in range(len(joint_data_obs_dict['passive_detection_enemy_list'])):
            # Channel: detected enemy pos. value=enemy id
            self.__set_value_in_img(joint_img[0, :, :, 0], int(joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_x'] / self.img_obs_reduce_ratio),
                                    joint_data_obs_dict['passive_detection_enemy_list'][x]['id'])
            # Channe2: detected enemy pos. value=enemy type
            self.__set_value_in_img(joint_img[0, :, :, 1], int(joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_x'] / self.img_obs_reduce_ratio),
                                    joint_data_obs_dict['passive_detection_enemy_list'][x]['type'] + 1)
        # detector
        for x in range(self.detector_num):
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                # Channel: detected enemy pos. value=enemy id
                self.__set_value_in_img(joint_img[0, :, :, 0],
                                        int(detector_data_obs_list[x]['r_visible_list'][y]['pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y]['pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['id'])
                # Channe2: detected enemy pos. value=enemy type
                self.__set_value_in_img(joint_img[0, :, :, 1],
                                        int(detector_data_obs_list[x]['r_visible_list'][y]['pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y]['pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
        # fighter
        for x in range(self.fighter_num):
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                # Channel: detected enemy pos. value=enemy id
                self.__set_value_in_img(joint_img[0, :, :, 0],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['id'])
                # Channe2: detected enemy pos. value=enemy type
                self.__set_value_in_img(joint_img[0, :, :, 1],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
        return detector_img, fighter_img, joint_img

    def __set_value_in_img(self, img, pos_x, pos_y, value):
        """
        draw 3*3 rectangle in img
        :param img:
        :param pos_x:
        :param pos_y:
        :param value:
        :return:
        """
        img_obs_size_x = int(self.battlefield_size_y / self.img_obs_reduce_ratio)
        img_obs_size_y = int(self.battlefield_size_x / self.img_obs_reduce_ratio)
        # 左上角
        if pos_x == 0 and pos_y == 0:
            img[pos_x: pos_x + 2, pos_y: pos_y + 2] = value
        # 左下角
        elif pos_x == 0 and pos_y == (img_obs_size_y - 1):
            img[pos_x: pos_x + 2, pos_y - 1: pos_y + 1] = value
        # 右上角
        elif pos_x == (img_obs_size_x - 1) and pos_y == 0:
            img[pos_x - 1: pos_x + 1, pos_y: pos_y + 2] = value
        # 右下角
        elif pos_x == (img_obs_size_x - 1) and pos_y == (img_obs_size_y - 1):
            img[pos_x - 1: pos_x + 1, pos_y - 1: pos_y + 1] = value
        # 左边
        elif pos_x == 0:
            img[pos_x: pos_x + 2, pos_y - 1: pos_y + 2] = value
        # 右边
        elif pos_x == img_obs_size_x - 1:
            img[pos_x - 1: pos_x + 1, pos_y - 1: pos_y + 2] = value
        # 上边
        elif pos_y == 0:
            img[pos_x - 1: pos_x + 2, pos_y: pos_y + 2] = value
        # 下边
        elif pos_y == img_obs_size_y - 1:
            img[pos_x - 1: pos_x + 2, pos_y - 1: pos_y + 1] = value
        # 其他位置
        else:
            img[pos_x - 1: pos_x + 2, pos_y - 1: pos_y + 2] = value

    def __get_data_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        detector_data = np.full((self.detector_num, 1), -1, dtype=np.int32)
        #fighter_data = np.zeros((self.fighter_num, 46), dtype=np.int32)
        fighter_data = np.zeros((self.fighter_num, 10), dtype=np.float32)
        enemy_data = np.zeros((self.fighter_num, 3), dtype=np.int32)
        radar_data = []
        striking_data = []
        visible_data = []
        # Detector info
        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                detector_data[x, 0] = detector_data_obs_list[x]['course']
        # Fighter info
        # 每个agent的局部观测   n_state = 6 + 2*2 = 10
        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                fighter_data[x, 0] = fighter_data_obs_list[x]['course']
                fighter_data[x, 1] = fighter_data_obs_list[x]['l_missile_left']
                fighter_data[x, 2] = fighter_data_obs_list[x]['s_missile_left']
                fighter_data[x, 3] = fighter_data_obs_list[x]['pos_x']
                fighter_data[x, 4] = fighter_data_obs_list[x]['pos_y']
                fighter_data[x, 5] = fighter_data_obs_list[x]['alive']
                for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                    if y < 2:
                        #fighter_data[x, 3 * y + 6] = fighter_data_obs_list[x]['r_visible_list'][y]['id']
                        fighter_data[x, 2 * y + 6] = fighter_data_obs_list[x]['r_visible_list'][y]['pos_x']
                        fighter_data[x, 2 * y + 7] = fighter_data_obs_list[x]['r_visible_list'][y]['pos_y']
                """
                for z in range(len(fighter_data_obs_list[x]['j_recv_list'])):
                    if z <= 2:
                        fighter_data[x, z + 15] = fighter_data_obs_list[x]['j_recv_list'][z]['id']
                """
            radar_data.append(fighter_data_obs_list[x]['j_recv_list'])
            striking_data.append(fighter_data_obs_list[x]['striking_dict_list'])
            visible_data.append(fighter_data_obs_list[x]['r_visible_list'])
        # Enemy  info
        '''
        for x in range(self.fighter_num):
            for y in range(len(joint_data_obs_dict['passive_detection_enemy_list'])):
                enemy_data[x, 0] = joint_data_obs_dict['passive_detection_enemy_list'][y]['pos_x']
                enemy_data[x, 1] = joint_data_obs_dict['passive_detection_enemy_list'][y]['pos_y']
                enemy_data[x, 2] = joint_data_obs_dict['passive_detection_enemy_list'][y]['course']
        
        '''
        return detector_data, fighter_data, radar_data, striking_data, visible_data

def get_state(obs_red_dict, obs_blue_dict):
    fighter_red_obs_list = obs_red_dict['fighter_obs_list']
    fighter_blue_obs_list = obs_blue_dict['fighter_obs_list']
    # joint_data_obs_dict = obs_raw_dict['joint_obs_dict']
    state_date = np.zeros((20, 6), dtype=np.int32)
    for x in range(10):
        state_date[x][0] = fighter_red_obs_list[x]['alive']
        state_date[x][1] = fighter_red_obs_list[x]['pos_x']
        state_date[x][2] = fighter_red_obs_list[x]['pos_y']
        state_date[x][3] = fighter_red_obs_list[x]['course']
        state_date[x][4] = fighter_red_obs_list[x]['l_missile_left']
        state_date[x][5] = fighter_red_obs_list[x]['s_missile_left']
    for y in range(10):
        state_date[y+10][0] = fighter_blue_obs_list[x]['alive']
        state_date[y+10][1] = fighter_blue_obs_list[x]['pos_x']
        state_date[y+10][2] = fighter_blue_obs_list[x]['pos_y']
        state_date[y+10][3] = fighter_blue_obs_list[x]['course']
        state_date[y+10][4] = fighter_blue_obs_list[x]['l_missile_left']
        state_date[y+10][5] = fighter_blue_obs_list[x]['s_missile_left']

    return state_date.flatten()
