import numpy as np
import math


class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num
        self.img_obs_reduce_ratio = 10
        self.img_obs_size_x = int(self.battlefield_size_y / self.img_obs_reduce_ratio)
        self.img_obs_size_y = int(self.battlefield_size_x / self.img_obs_reduce_ratio)

        # Reuse buffers to reduce per-step allocations.
        self._detector_img = np.zeros((self.detector_num, self.img_obs_size_x, self.img_obs_size_y, 3), dtype=np.int32)
        self._fighter_img = np.zeros((self.fighter_num, self.img_obs_size_x, self.img_obs_size_y, 3), dtype=np.int32)
        self._joint_img = np.zeros((1, self.img_obs_size_x, self.img_obs_size_y, 2), dtype=np.int32)
        self._tmp_pos_obs = np.zeros((self.img_obs_size_x, self.img_obs_size_y), dtype=np.int32)

    def obs_construct(self, obs_raw_dict):
        obs_dict = {}
        detector_obs_list = []
        fighter_obs_list = []
        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']
        detector_img, fighter_img, joint_img = self.__get_img_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        detector_data, fighter_data, radar_data = self.__get_data_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        alive_status = self.__get_alive_status(detector_data_obs_list, fighter_data_obs_list)
        joint_plane = joint_img[0]

        for x in range(self.detector_num):
            img_context = np.empty((self.img_obs_size_x, self.img_obs_size_y, 5), dtype=np.int32)
            img_context[:, :, :3] = detector_img[x]
            img_context[:, :, 3:] = joint_plane
            data_context = detector_data[x, :]
            detector_obs_list.append({'info': data_context.copy(), 'screen': img_context, 'alive': bool(alive_status[x, 0])})

        for x in range(self.fighter_num):
            img_context = np.empty((self.img_obs_size_x, self.img_obs_size_y, 5), dtype=np.int32)
            img_context[:, :, :3] = fighter_img[x]
            img_context[:, :, 3:] = joint_plane
            data_context = fighter_data[x, :]
            fighter_obs_list.append({'info': data_context.copy(), 'screen': img_context, 'alive': bool(alive_status[x + self.detector_num, 0])})

        obs_dict['detector'] = detector_obs_list
        obs_dict['fighter'] = fighter_obs_list
        obs_dict['radar'] = radar_data

        return obs_dict

    def __get_alive_status(self, detector_data_obs_list, fighter_data_obs_list):
        alive_status = np.full((self.detector_num + self.fighter_num, 1), True)
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                alive_status[x, 0] = False
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                alive_status[x + self.detector_num, 0] = False
        return alive_status

    def __get_img_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        ratio = self.img_obs_reduce_ratio
        detector_img = self._detector_img
        fighter_img = self._fighter_img
        joint_img = self._joint_img
        tmp_pos_obs = self._tmp_pos_obs

        detector_img.fill(0)
        fighter_img.fill(0)
        joint_img.fill(0)
        tmp_pos_obs.fill(0)

        # set all friendly positions: detector=1, fighter=2, self=255
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                continue
            self.__set_value_in_img(
                tmp_pos_obs,
                int(detector_data_obs_list[x]['pos_y'] / ratio),
                int(detector_data_obs_list[x]['pos_x'] / ratio),
                1,
            )
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                continue
            self.__set_value_in_img(
                tmp_pos_obs,
                int(fighter_data_obs_list[x]['pos_y'] / ratio),
                int(fighter_data_obs_list[x]['pos_x'] / ratio),
                2,
            )

        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                continue
            visible_list = detector_data_obs_list[x]['r_visible_list']
            det_id = detector_img[x, :, :, 0]
            det_type = detector_img[x, :, :, 1]
            for target in visible_list:
                pos_x = int(target['pos_y'] / ratio)
                pos_y = int(target['pos_x'] / ratio)
                self.__set_value_in_img(det_id, pos_x, pos_y, target['id'])
                self.__set_value_in_img(det_type, pos_x, pos_y, target['type'] + 1)

            det_friend = detector_img[x, :, :, 2]
            det_friend[:] = tmp_pos_obs
            self.__set_value_in_img(
                det_friend,
                int(detector_data_obs_list[x]['pos_y'] / ratio),
                int(detector_data_obs_list[x]['pos_x'] / ratio),
                255,
            )

        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                continue
            visible_list = fighter_data_obs_list[x]['r_visible_list']
            fig_id = fighter_img[x, :, :, 0]
            fig_type = fighter_img[x, :, :, 1]
            for target in visible_list:
                pos_x = int(target['pos_y'] / ratio)
                pos_y = int(target['pos_x'] / ratio)
                self.__set_value_in_img(fig_id, pos_x, pos_y, target['id'])
                self.__set_value_in_img(fig_type, pos_x, pos_y, target['type'] + 1)

            fig_friend = fighter_img[x, :, :, 2]
            fig_friend[:] = tmp_pos_obs
            self.__set_value_in_img(
                fig_friend,
                int(fighter_data_obs_list[x]['pos_y'] / ratio),
                int(fighter_data_obs_list[x]['pos_x'] / ratio),
                255,
            )

        joint_id = joint_img[0, :, :, 0]
        joint_type = joint_img[0, :, :, 1]
        passive_detection_enemy_list = joint_data_obs_dict['passive_detection_enemy_list']

        for target in passive_detection_enemy_list:
            pos_x = int(target['pos_y'] / ratio)
            pos_y = int(target['pos_x'] / ratio)
            self.__set_value_in_img(joint_id, pos_x, pos_y, target['id'])
            self.__set_value_in_img(joint_type, pos_x, pos_y, target['type'] + 1)

        for x in range(self.detector_num):
            visible_list = detector_data_obs_list[x]['r_visible_list']
            for target in visible_list:
                pos_x = int(target['pos_y'] / ratio)
                pos_y = int(target['pos_x'] / ratio)
                self.__set_value_in_img(joint_id, pos_x, pos_y, target['id'])
                self.__set_value_in_img(joint_type, pos_x, pos_y, target['type'] + 1)

        for x in range(self.fighter_num):
            visible_list = fighter_data_obs_list[x]['r_visible_list']
            for target in visible_list:
                pos_x = int(target['pos_y'] / ratio)
                pos_y = int(target['pos_x'] / ratio)
                self.__set_value_in_img(joint_id, pos_x, pos_y, target['id'])
                self.__set_value_in_img(joint_type, pos_x, pos_y, target['type'] + 1)

        return detector_img, fighter_img, joint_img

    def __set_value_in_img(self, img, pos_x, pos_y, value):
        x0 = max(pos_x - 1, 0)
        x1 = min(pos_x + 2, self.img_obs_size_x)
        y0 = max(pos_y - 1, 0)
        y1 = min(pos_y + 2, self.img_obs_size_y)
        img[x0:x1, y0:y1] = value

    def __get_data_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        detector_data = np.full((self.detector_num, 1), -1, dtype=np.int32)
        fighter_data = np.full((self.fighter_num, 6), -1, dtype=np.int32)
        radar_data = []
        visible_data = []

        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                detector_data[x, 0] = detector_data_obs_list[x]['course']

        for x in range(self.fighter_num):
            fighter = fighter_data_obs_list[x]
            if fighter['alive']:
                fighter_data[x, 0] = fighter['course']
                fighter_data[x, 1] = fighter['l_missile_left']
                fighter_data[x, 2] = fighter['s_missile_left']
                visible_list = fighter['r_visible_list']
                if len(visible_list) >= 1:
                    closest_id = 0
                    min_dist_sq = 10000 * 10000
                    own_pos_x = fighter['pos_x']
                    own_pos_y = fighter['pos_y']
                    for y, target in enumerate(visible_list):
                        dx = own_pos_x - target['pos_x']
                        dy = own_pos_y - target['pos_y']
                        dist_sq = dx * dx + dy * dy
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            closest_id = y
                    fighter_data[x, 3] = math.sqrt(min_dist_sq)
                    closest_target = visible_list[closest_id]
                    fighter_data[x, 4] = closest_target['id']
                    fighter_data[x, 5] = 180 / 3.1415 * math.atan2(
                        closest_target['pos_y'] - own_pos_y,
                        closest_target['pos_x'] - own_pos_x,
                    )
                else:
                    fighter_data[x, 3] = 0
                    fighter_data[x, 4] = 0
                    fighter_data[x, 5] = 0

        return detector_data, fighter_data, radar_data
