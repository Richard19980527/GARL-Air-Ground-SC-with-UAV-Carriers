from util import *


class SubRollout:
    def __init__(self, rollout_element_name_list):
        self.rollout_element_name_list = rollout_element_name_list
        self.sub_episode_buffer_list = []

    def reset(self):
        self.sub_episode_buffer_list = []
        self.add_new_sub_episode_buffer()

    def add_new_sub_episode_buffer(self):
        sub_episode_buffer = {}
        for rollout_element_name in self.rollout_element_name_list:
            if rollout_element_name not in sub_episode_buffer:
                sub_episode_buffer[rollout_element_name] = []
        self.sub_episode_buffer_list.append(copy.deepcopy(sub_episode_buffer))

    def delete_last_empty_sub_episode_buffer(self):
        if len(self.sub_episode_buffer_list[-1]['step_id_s']) == 0:
            self.sub_episode_buffer_list = self.sub_episode_buffer_list[:-1]

    def append_rollout_element(self, rollout_element_name, rollout_element_value):
        self.sub_episode_buffer_list[-1][rollout_element_name].append(copy.deepcopy(rollout_element_value))


class SubRolloutManager:
    def __init__(self, env):
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.env = env
        self.sub_rollout_dict = {}

    def reset_sub_rollouts(self):
        for sub_rollout_type in self.sub_rollout_dict:
            for sub_rollout_id in self.sub_rollout_dict[sub_rollout_type]:
                self.sub_rollout_dict[sub_rollout_type][sub_rollout_id].reset()

    def add_sub_rollout(self, rollout_element_name_list, sub_rollout_type, sub_rollout_id):
        if sub_rollout_type not in self.sub_rollout_dict:
            self.sub_rollout_dict[sub_rollout_type] = {}
        self.sub_rollout_dict[sub_rollout_type][sub_rollout_id] = SubRollout(rollout_element_name_list)

    def delete_last_empty_sub_episode_buffer(self):
        for sub_rollout_type in self.sub_rollout_dict:
            for sub_rollout_id in self.sub_rollout_dict[sub_rollout_type]:
                self.sub_rollout_dict[sub_rollout_type][sub_rollout_id].delete_last_empty_sub_episode_buffer()

    def gen_rewards(self):
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            # for UGV
            ugv_id = UGV_UAVs_Group_id
            ugv_sub_rollout_id = str(ugv_id)
            if ugv_sub_rollout_id in self.sub_rollout_dict['UGV']:
                for sub_episode_buffer in self.sub_rollout_dict['UGV'][ugv_sub_rollout_id].sub_episode_buffer_list:
                    for step_id_idx in range(len(sub_episode_buffer['step_id_s']) - 1):
                        last_step = sub_episode_buffer['step_id_s'][step_id_idx]
                        next_step = sub_episode_buffer['step_id_s'][step_id_idx + 1]
                        # fairness
                        fairness = 0.0
                        next_final_poi_visit_time = self.env.episode_log_info_dict['final_poi_visit_time'][next_step]
                        square_of_sum = np.square(np.sum(next_final_poi_visit_time))
                        sum_of_square = np.sum(np.square(next_final_poi_visit_time))
                        if sum_of_square > 1e-5:
                            fairness = square_of_sum / sum_of_square / next_final_poi_visit_time.shape[0]
                        # data_collection
                        data_collection = 0.0
                        for uav_id in range(self.env_conf['uav_num_each_group']):
                            uav = self.env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                            data_collection += np.sum(uav.episode_log_info_dict['final_data_collection'][next_step] -
                                                      uav.episode_log_info_dict['final_data_collection'][last_step])
                        # hit
                        hit = 0.0
                        for uav_id in range(self.env_conf['uav_num_each_group']):
                            uav = self.env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                            hit += uav.episode_log_info_dict['final_hit'][next_step] - \
                                   uav.episode_log_info_dict['final_hit'][last_step]
                        # energy_consumption
                        energy_consumption = 0.0
                        for uav_id in range(self.env_conf['uav_num_each_group']):
                            uav = self.env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                            energy_consumption += uav.episode_log_info_dict['final_energy_consumption'][next_step] - \
                                                  uav.episode_log_info_dict['final_energy_consumption'][last_step]
                        # UGV_UAV_cooperation_ratio (cor)
                        total_collect_data_time = 0.0
                        total_fly_time = 0.0
                        for uav_id in range(self.env_conf['uav_num_each_group']):
                            uav = self.env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                            total_collect_data_time += uav.episode_log_info_dict['final_collect_data_time'][next_step] - \
                                   uav.episode_log_info_dict['final_collect_data_time'][last_step]
                            total_fly_time += uav.episode_log_info_dict['final_fly_time'][next_step] - \
                                                       uav.episode_log_info_dict['final_fly_time'][last_step]
                        if total_fly_time > 0:
                            cor = total_collect_data_time / total_fly_time
                        else:
                            cor = 0

                        # reward
                        reward_collect = (self.method_conf['ugv_positive_factor'] * fairness * data_collection) / (
                                energy_consumption + min_value)
                        reward_collect_clipped = np.clip(reward_collect, 0, self.method_conf['ugv_max_positive_reward'])
                        penalty = -self.method_conf['ugv_penalty_factor'] * hit / self.env_conf['uav_num_each_group']
                        reward = reward_collect_clipped + penalty
                        sub_episode_buffer['reward_s'].append(reward)
            # for UAV
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                uav = self.env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id]
                if uav_sub_rollout_id in self.sub_rollout_dict['UAV']:
                    for sub_episode_buffer in self.sub_rollout_dict['UAV'][uav_sub_rollout_id].sub_episode_buffer_list:
                        for step_id_idx in range(len(sub_episode_buffer['step_id_s']) - 1):
                            last_step = sub_episode_buffer['step_id_s'][step_id_idx]
                            next_step = sub_episode_buffer['step_id_s'][step_id_idx + 1]
                            # fairness
                            fairness = 0.0
                            next_final_poi_visit_time = self.env.episode_log_info_dict['final_poi_visit_time'][
                                next_step]
                            square_of_sum = np.square(np.sum(next_final_poi_visit_time))
                            sum_of_square = np.sum(np.square(next_final_poi_visit_time))
                            if sum_of_square > 1e-5:
                                fairness = square_of_sum / sum_of_square / next_final_poi_visit_time.shape[0]
                            # data_collection
                            data_collection = np.sum(
                                uav.episode_log_info_dict['final_data_collection'][next_step] -
                                uav.episode_log_info_dict['final_data_collection'][last_step])
                            # hit
                            hit = uav.episode_log_info_dict['final_hit'][next_step] - \
                                  uav.episode_log_info_dict['final_hit'][last_step]
                            # energy_consumption
                            energy_consumption = uav.episode_log_info_dict['final_energy_consumption'][next_step] - \
                                                 uav.episode_log_info_dict['final_energy_consumption'][last_step]
                            # reward
                            reward_collect = (self.method_conf['uav_positive_factor'] * fairness * data_collection) / (
                                    energy_consumption + min_value)
                            reward_collect_clipped = np.clip(reward_collect, 0, self.method_conf['uav_max_positive_reward'])
                            penalty = -self.method_conf['uav_penalty_factor'] * hit
                            reward = reward_collect_clipped + penalty
                            sub_episode_buffer['reward_s'].append(reward)

    def gen_returns(self):
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            # for UGV
            ugv_id = UGV_UAVs_Group_id
            ugv_sub_rollout_id = str(ugv_id)
            if ugv_sub_rollout_id in self.sub_rollout_dict['UGV']:
                for sub_episode_buffer in self.sub_rollout_dict['UGV'][ugv_sub_rollout_id].sub_episode_buffer_list:
                    if self.method_conf['use_gae']:
                        gae = 0
                        for sub_episode_step in reversed(range(len(sub_episode_buffer['reward_s']))):
                            delta = sub_episode_buffer['reward_s'][sub_episode_step] + self.method_conf['gamma'] * \
                                    sub_episode_buffer['value_s'][sub_episode_step + 1] - sub_episode_buffer['value_s'][
                                        sub_episode_step]
                            gae = delta + self.method_conf['gamma'] * self.method_conf['tau'] * gae
                            sub_episode_buffer['return_s'].insert(0,
                                                                  gae + sub_episode_buffer['value_s'][sub_episode_step])
                    else:
                        next_return = sub_episode_buffer['value_s'][-1]
                        for sub_episode_step in reversed(range(len(sub_episode_buffer['reward_s']))):
                            sub_episode_buffer['return_s'].insert(0, next_return * self.method_conf['gamma'] +
                                                                  sub_episode_buffer['reward_s'][sub_episode_step])
                            next_return = sub_episode_buffer['return_s'][-1]
                    sub_episode_buffer['value_s'] = sub_episode_buffer['value_s'][:-1]
            # for UAV
            for uav_id in range(self.env_conf['uav_num_each_group']):
                uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                if uav_sub_rollout_id in self.sub_rollout_dict['UAV']:
                    for sub_episode_buffer in self.sub_rollout_dict['UAV'][uav_sub_rollout_id].sub_episode_buffer_list:
                        if self.method_conf['use_gae']:
                            gae = 0
                            for sub_episode_step in reversed(range(len(sub_episode_buffer['reward_s']))):
                                delta = sub_episode_buffer['reward_s'][sub_episode_step] + self.method_conf['gamma'] * \
                                        sub_episode_buffer['value_s'][sub_episode_step + 1] - \
                                        sub_episode_buffer['value_s'][
                                            sub_episode_step]
                                gae = delta + self.method_conf['gamma'] * self.method_conf['tau'] * gae
                                sub_episode_buffer['return_s'].insert(0,
                                                                      gae + sub_episode_buffer['value_s'][
                                                                          sub_episode_step])
                        else:
                            next_return = sub_episode_buffer['value_s'][-1]
                            for sub_episode_step in reversed(range(len(sub_episode_buffer['reward_s']))):
                                sub_episode_buffer['return_s'].insert(0, next_return * self.method_conf['gamma'] +
                                                                      sub_episode_buffer['reward_s'][sub_episode_step])
                                next_return = sub_episode_buffer['return_s'][-1]
                        sub_episode_buffer['value_s'] = sub_episode_buffer['value_s'][:-1]
