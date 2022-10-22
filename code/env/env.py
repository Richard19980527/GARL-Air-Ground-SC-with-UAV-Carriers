from util import *


class UGV:
    def __init__(self, ugv_id):
        self.method_conf = get_global_dict_value('method_conf')
        self.ugv_id = ugv_id
        if self.method_conf['ugv_trace_type'] == 'roads_net':
            self.final_road_pos = None
            self.final_passed_road_node_id_list = None
            self.passed_road_node_id_list = None
        self.cur_stop_id = None
        self.cur_loc_poi_value_array = None
        self.obs_X_B_u = None
        self.episode_log_info_dict = {}

    def add_log_info(self):
        if self.method_conf['ugv_trace_type'] == 'roads_net':
            self.episode_log_info_dict['final_road_pos'].append(copy.deepcopy(self.final_road_pos))
            self.episode_log_info_dict['final_passed_road_node_id_list'].append(
                copy.deepcopy(self.final_passed_road_node_id_list))
        self.episode_log_info_dict['cur_stop_id_list'].append(copy.deepcopy(self.cur_stop_id))


class UAV:
    def __init__(self, uav_id):
        self.uav_id = uav_id
        self.final_pos = None
        self.final_energy = None
        self.final_energy_consumption = None
        self.final_hit = None
        self.final_out_of_ugv = None
        self.final_data_collection = None
        self.final_poi_visit_time = None
        self.final_collect_data_time = None
        self.final_fly_time = None
        self.episode_log_info_dict = {}

    def add_log_info(self):
        self.episode_log_info_dict['final_pos'].append(copy.deepcopy(self.final_pos))
        self.episode_log_info_dict['final_energy'].append(self.final_energy)
        self.episode_log_info_dict['final_energy_consumption'].append(self.final_energy_consumption)
        self.episode_log_info_dict['final_hit'].append(self.final_hit)
        self.episode_log_info_dict['final_out_of_ugv'].append(self.final_out_of_ugv)
        self.episode_log_info_dict['final_data_collection'].append(copy.deepcopy(self.final_data_collection))
        self.episode_log_info_dict['final_poi_visit_time'].append(copy.deepcopy(self.final_poi_visit_time))
        self.episode_log_info_dict['final_collect_data_time'].append(self.final_collect_data_time)
        self.episode_log_info_dict['final_fly_time'].append(self.final_fly_time)


class UGV_UAVs_Group:
    def __init__(self, ugv_uavs_group_id):
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.ugv_uavs_group_id = ugv_uavs_group_id
        self.ugv = UGV(self.ugv_uavs_group_id)
        self.uav_list = [UAV(uav_id) for uav_id in range(self.env_conf['uav_num_each_group'])]
        # status:
        # Init: 0
        # Deliver: 1
        # Wait: 2
        # Call: 3
        self.last_status = None
        self.last_status_length = None
        self.next_status = None

        self.wait_step_num = self.env_conf['wait_step_num']
        self.call_step_num = self.env_conf['call_step_num']
        self.episode_log_info_dict = {}

    def add_log_info(self):
        self.episode_log_info_dict['status'].append(self.last_status)


class Env:
    def __init__(self):
        # load confs
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')
        # gen ugv_uavs_groups
        self.UGV_UAVs_Group_list = [
            UGV_UAVs_Group(ugv_uavs_group_id) for ugv_uavs_group_id in range(self.env_conf['UGV_UAVs_Group_num'])]

        # load pre_set dicts
        self.poi2coordxy_dict = \
            np.load(os.path.join(self.dataset_conf['dataset_path'], 'poi2coordxy_dict.npy'), allow_pickle=True)[()]
        self.poi_num = len(self.poi2coordxy_dict.keys())
        self.poi2coordxy_value_dict = \
            np.load(os.path.join(self.dataset_conf['dataset_path'], 'poi2coordxy_value_dict_' + str(
                self.env_conf['poi_value_min']) + '_' + str(
                self.env_conf['poi_value_max']) + '.npy'), allow_pickle=True)[()]
        self.poi_coordxy_array = np.zeros([self.poi_num, 2], dtype=np.float32)
        for poi_id in range(self.poi_num):
            self.poi_coordxy_array[poi_id][0] = self.poi2coordxy_value_dict[poi_id]['coordxy'][0]
            self.poi_coordxy_array[poi_id][1] = self.poi2coordxy_value_dict[poi_id]['coordxy'][1]
        self.poi_init_value_array = np.zeros(self.poi_num, dtype=np.float32)
        for poi_id in range(self.poi_num):
            self.poi_init_value_array[poi_id] = self.poi2coordxy_value_dict[poi_id]['value']
        self.poi2cell_dict = np.load(os.path.join(self.dataset_conf['dataset_path'],
                                                  'poi2cell_dict_' + str(
                                                      self.env_conf['uav_cellset_grid_size']) + '_' + str(
                                                      self.env_conf['uav_sensing_range']) + '.npy'), allow_pickle=True)[
            ()]
        self.uav_cellset = \
            np.load(os.path.join(self.dataset_conf['dataset_path'],
                                 'uav_cellset_' + str(self.env_conf['uav_cellset_grid_size']) + '.npy'),
                    allow_pickle=True)[()]
        self.uav_cell2poi_dict = \
            np.load(os.path.join(self.dataset_conf['dataset_path'],
                                 'uav_cell2poi_dict_' + str(self.env_conf['uav_cellset_grid_size']) + '_' + str(
                                     self.env_conf['uav_sensing_range']) + '.npy'), allow_pickle=True)[()]
        self.roads_cellset = \
            np.load(os.path.join(self.dataset_conf['dataset_path'],
                                 'roads_cellset_' + str(self.env_conf['uav_cellset_grid_size']) + '.npy'),
                    allow_pickle=True)[()]
        self.roads_net_dict = \
            np.load(os.path.join(self.dataset_conf['dataset_path'], 'roads_net_dict.npy'), allow_pickle=True)[()]
        self.stops_net_dict = \
            np.load(os.path.join(self.dataset_conf['dataset_path'],
                                 'stops_net_dict_' + str(self.env_conf['stop_gap']) + '.npy'), allow_pickle=True)[()]
        self.stop2cell_dict = np.load(os.path.join(self.dataset_conf['dataset_path'],
                                                   'stop2cell_dict_' + str(
                                                       self.env_conf['uav_cellset_grid_size']) + '_' + str(
                                                       self.env_conf['stop_poi_max_dis']) + '.npy'), allow_pickle=True)[
            ()]
        self.stop_num = len(self.stops_net_dict)
        self.stops_pois_AdjMatrix = np.load(os.path.join(self.dataset_conf['dataset_path'],
                                                         'stops_pois_AdjMatrix_' + str(
                                                             self.env_conf['stop_gap']) + '_' + str(
                                                             self.env_conf['stop_poi_max_dis']) + '.npy'),
                                            allow_pickle=True)
        self.stops_net_SP_dict = np.load(os.path.join(self.dataset_conf['dataset_path'],
                                                      'stops_net_SP_dict_' + str(self.env_conf['stop_gap']) + '.npy'),
                                         allow_pickle=True)[()]
        self.episode_log_info_dict = {}
        # helpers
        self.ec_upper_bound = self.env_conf['UGV_UAVs_Group_num'] * self.env_conf['uav_num_each_group'] * self.env_conf[
            'max_uav_move_dis_each_step'] * self.env_conf['max_step_num'] * self.env_conf[
                                  'uav_move_energy_consume_ratio']

        self.stops_net_SP_Matrix = np.ones([self.stop_num, self.stop_num], dtype=np.float32)
        self.stops_net_SP_Matrix *= 1e3
        for key in self.stops_net_SP_dict:
            start_stop_id = int(key.split('_')[0])
            goal_stop_id = int(key.split('_')[1])
            self.stops_net_SP_Matrix[start_stop_id, goal_stop_id] = self.stops_net_SP_dict[key]['min_dis']

        # gen Laplacian Matrix
        self.stops_net_AdjMatrix = np.zeros([self.stop_num, self.stop_num], dtype=np.float32)
        for stop_id in self.stops_net_dict:
            for next_stop_id in self.stops_net_dict[stop_id]['next_node_list']:
                self.stops_net_AdjMatrix[stop_id, next_stop_id] = 1
        self.stops_net_AdjMatrix_tilde = self.stops_net_AdjMatrix + np.identity(self.stop_num, dtype=np.float32)
        self.stops_net_DegreeMatrix_tilde = np.diag(np.sum(self.stops_net_AdjMatrix_tilde, axis=1))
        self.stops_net_LaplacianMatrix = fractional_matrix_power(self.stops_net_DegreeMatrix_tilde,
                                                                 -0.5) @ self.stops_net_AdjMatrix_tilde @ fractional_matrix_power(
            self.stops_net_DegreeMatrix_tilde, -0.5)

        if 'dsp_q' in self.method_conf:
            # gen pre_S_Matrix
            self.pre_S_Matrix = np.zeros([self.stop_num, self.stop_num], dtype=np.float32)
            for i in range(self.stop_num):
                for j in range(self.stop_num):
                    if self.stops_net_SP_Matrix[i, j] <= self.method_conf['dsp_q']:
                        self.pre_S_Matrix[i, j] = 1 / (1 + self.stops_net_SP_Matrix[i, j])
        # gen ugv_move_mask_Matrix
        self.ugv_move_mask_Matrix = np.zeros([self.stop_num, self.stop_num + 2], dtype=np.float32)
        for i in range(self.stop_num):
            for j in range(self.stop_num):
                if self.env_conf['stop_gap'] * self.stops_net_SP_Matrix[i, j] <= self.env_conf[
                    'max_ugv_move_dis_each_step']:
                    self.ugv_move_mask_Matrix[i, j] = 1
        self.ugv_move_mask_Matrix[:, -2:] = 1

        # gen UGV_init_obs_X_B_u
        self.UGV_init_obs_X_B_u = np.zeros([self.stop_num, 3], dtype=np.float32)
        self.UGV_init_loc_poi_value_array = self.env_conf['poi_value_max'] * np.ones(self.poi_num, dtype=np.float32)
        self.stop_max_value = self.env_conf['poi_value_max'] * np.max(np.sum(self.stops_pois_AdjMatrix, axis=1))
        for stop_id in self.stops_net_dict:
            self.UGV_init_obs_X_B_u[stop_id, 0] = self.stops_net_dict[stop_id]['coordxy'][0]
            self.UGV_init_obs_X_B_u[stop_id, 1] = self.stops_net_dict[stop_id]['coordxy'][1]
        self.UGV_init_obs_X_B_u[:, 2] = np.sum(self.UGV_init_loc_poi_value_array * self.stops_pois_AdjMatrix, axis=1)

        # gen road_node_id_pair2stop_id_dict
        self.road_node_id_pair2stop_id_dict = {}
        for stop_id in self.stops_net_dict:
            road_pos = self.stops_net_dict[stop_id]['road_pos']
            if road_pos['progress'] < 1e-5:
                start_road_node = self.roads_net_dict[road_pos['start_road_node_id']]
                for next_road_node_id in start_road_node['next_node_list']:
                    key = str(road_pos['start_road_node_id']) + '_' + str(next_road_node_id)
                    if key not in self.road_node_id_pair2stop_id_dict:
                        self.road_node_id_pair2stop_id_dict[key] = []
                    self.road_node_id_pair2stop_id_dict[key].append(stop_id)
            elif 1 - road_pos['progress'] < 1e-5:
                start_road_node = self.roads_net_dict[road_pos['end_road_node_id']]
                for next_road_node_id in start_road_node['next_node_list']:
                    key = str(road_pos['start_road_node_id']) + '_' + str(next_road_node_id)
                    if key not in self.road_node_id_pair2stop_id_dict:
                        self.road_node_id_pair2stop_id_dict[key] = []
                    self.road_node_id_pair2stop_id_dict[key].append(stop_id)
            else:
                key = str(road_pos['start_road_node_id']) + '_' + str(road_pos['end_road_node_id'])
                if key not in self.road_node_id_pair2stop_id_dict:
                    self.road_node_id_pair2stop_id_dict[key] = []
                self.road_node_id_pair2stop_id_dict[key].append(stop_id)
        # gen UGV_init_stop_id
        self.UGV_init_stop_id = self.road_pos2stop_id(self.dataset_conf['UGV_init_road_pos'])



    def add_log_info(self):
        self.episode_log_info_dict['final_poi_visit_time'].append(copy.deepcopy(self.final_poi_visit_time))
        self.episode_log_info_dict['poi_cur_value_array'].append(copy.deepcopy(self.poi_cur_value_array))

    def reset(self):
        self.cur_step = 0
        self.poi_last_value_array = copy.deepcopy(self.poi_init_value_array)
        self.poi_cur_value_array = copy.deepcopy(self.poi_init_value_array)
        self.final_poi_visit_time = np.zeros(self.poi_num, dtype=np.float32)
        self.final_total_hit = 0
        self.final_total_out_of_ugv = 0
        self.final_energy_consumption = 0
        self.final_total_collect_data_time = 0
        self.final_total_fly_time = 0
        self.final_total_eff_relax_time = 0
        self.final_total_relax_time = 0
        for UGV_UAVs_Group in self.UGV_UAVs_Group_list:
            UGV_UAVs_Group.episode_log_info_dict = {}
            UGV_UAVs_Group.last_status = 0
            UGV_UAVs_Group.last_status_length = 0
            UGV_UAVs_Group.episode_log_info_dict['status'] = []

            ugv = UGV_UAVs_Group.ugv
            ugv.episode_log_info_dict = {}
            if self.method_conf['ugv_trace_type'] == 'roads_net':
                ugv.episode_log_info_dict['final_passed_road_node_id_list'] = []
                ugv.episode_log_info_dict['final_road_pos'] = []
                ugv.final_passed_road_node_id_list = []
                ugv.passed_road_node_id_list = []
                ugv.final_road_pos = self.dataset_conf['UGV_init_road_pos']
            ugv.episode_log_info_dict['cur_stop_id_list'] = []
            ugv.cur_stop_id = self.UGV_init_stop_id
            ugv.cur_loc_poi_value_array = copy.deepcopy(self.UGV_init_loc_poi_value_array)
            ugv.obs_X_B_u = copy.deepcopy(self.UGV_init_obs_X_B_u)

            ugv.add_log_info()
            for uav in UGV_UAVs_Group.uav_list:
                uav.episode_log_info_dict = {}
                uav.episode_log_info_dict['final_pos'] = []
                uav.episode_log_info_dict['final_energy'] = []
                uav.episode_log_info_dict['final_energy_consumption'] = []
                uav.episode_log_info_dict['final_hit'] = []
                uav.episode_log_info_dict['final_out_of_ugv'] = []
                uav.episode_log_info_dict['final_data_collection'] = []
                uav.episode_log_info_dict['final_poi_visit_time'] = []
                uav.episode_log_info_dict['final_collect_data_time'] = []
                uav.episode_log_info_dict['final_fly_time'] = []
                if self.method_conf['ugv_trace_type'] == 'roads_net':
                    uav.final_pos = self.road_pos2pos(UGV_UAVs_Group.ugv.final_road_pos)
                elif self.method_conf['ugv_trace_type'] == 'stops_net':
                    uav.final_pos = self.stops_net_dict[UGV_UAVs_Group.ugv.cur_stop_id]['coordxy']
                uav.final_energy = self.env_conf['uav_init_energy']
                uav.final_energy_consumption = 0
                uav.final_hit = 0
                uav.final_out_of_ugv = 0
                uav.final_data_collection = np.zeros(self.poi_num, dtype=np.float32)
                uav.final_poi_visit_time = np.zeros(self.poi_num, dtype=np.float32)
                uav.final_collect_data_time = 0
                uav.final_fly_time = 0
                uav.add_log_info()
        self.episode_log_info_dict = {}
        self.episode_log_info_dict['final_poi_visit_time'] = []
        self.episode_log_info_dict['poi_cur_value_array'] = []
        self.add_log_info()

    def road_pos2pos(self, road_pos):
        start_road_node_coordxy = self.roads_net_dict[road_pos['start_road_node_id']]['coordxy']
        end_road_node_coordxy = self.roads_net_dict[road_pos['end_road_node_id']]['coordxy']
        pos_coordx = start_road_node_coordxy[0] * (1 - road_pos['progress']) + end_road_node_coordxy[0] * road_pos[
            'progress']
        pos_coordy = start_road_node_coordxy[1] * (1 - road_pos['progress']) + end_road_node_coordxy[1] * road_pos[
            'progress']
        pos = (pos_coordx, pos_coordy)
        return pos

    def road_pos2stop_id(self, road_pos):
        stop_id_list = []
        if road_pos['progress'] < 1e-5:
            start_road_node = self.roads_net_dict[road_pos['start_road_node_id']]
            for next_road_node_id in start_road_node['next_node_list']:
                tmp_start_road_node_id = road_pos['start_road_node_id']
                tmp_next_road_node_id = next_road_node_id
                while True:
                    key = str(tmp_start_road_node_id) + '_' + str(tmp_next_road_node_id)
                    equa_key = str(tmp_next_road_node_id) + '_' + str(tmp_start_road_node_id)
                    if key in self.road_node_id_pair2stop_id_dict or equa_key in self.road_node_id_pair2stop_id_dict:
                        if key in self.road_node_id_pair2stop_id_dict:
                            stop_id_list.extend(self.road_node_id_pair2stop_id_dict[key])
                        else:
                            stop_id_list.extend(self.road_node_id_pair2stop_id_dict[equa_key])
                        break
                    for sub_next_road_node_id in self.roads_net_dict[tmp_next_road_node_id]['next_node_list']:
                        if sub_next_road_node_id != tmp_start_road_node_id:
                            tmp_start_road_node_id = tmp_next_road_node_id
                            tmp_next_road_node_id = sub_next_road_node_id
                            break
        elif 1 - road_pos['progress'] < 1e-5:
            start_road_node = self.roads_net_dict[road_pos['end_road_node_id']]
            for next_road_node_id in start_road_node['next_node_list']:
                tmp_start_road_node_id = road_pos['end_road_node_id']
                tmp_next_road_node_id = next_road_node_id
                while True:
                    key = str(tmp_start_road_node_id) + '_' + str(tmp_next_road_node_id)
                    equa_key = str(tmp_next_road_node_id) + '_' + str(tmp_start_road_node_id)
                    if key in self.road_node_id_pair2stop_id_dict or equa_key in self.road_node_id_pair2stop_id_dict:
                        if key in self.road_node_id_pair2stop_id_dict:
                            stop_id_list.extend(self.road_node_id_pair2stop_id_dict[key])
                        else:
                            stop_id_list.extend(self.road_node_id_pair2stop_id_dict[equa_key])
                        break
                    for sub_next_road_node_id in self.roads_net_dict[tmp_next_road_node_id]['next_node_list']:
                        if sub_next_road_node_id != tmp_start_road_node_id:
                            tmp_start_road_node_id = tmp_next_road_node_id
                            tmp_next_road_node_id = sub_next_road_node_id
                            break
        else:
            key = str(road_pos['start_road_node_id']) + '_' + str(road_pos['end_road_node_id'])
            equa_key = str(road_pos['end_road_node_id']) + '_' + str(road_pos['start_road_node_id'])
            if key in self.road_node_id_pair2stop_id_dict or equa_key in self.road_node_id_pair2stop_id_dict:
                if key in self.road_node_id_pair2stop_id_dict:
                    stop_id_list.extend(self.road_node_id_pair2stop_id_dict[key])
                else:
                    stop_id_list.extend(self.road_node_id_pair2stop_id_dict[equa_key])

            start_road_node = self.roads_net_dict[road_pos['start_road_node_id']]
            for next_road_node_id in start_road_node['next_node_list']:
                if next_road_node_id == road_pos['end_road_node_id']:
                    continue
                tmp_start_road_node_id = road_pos['start_road_node_id']
                tmp_next_road_node_id = next_road_node_id
                while True:
                    key = str(tmp_start_road_node_id) + '_' + str(tmp_next_road_node_id)
                    equa_key = str(tmp_next_road_node_id) + '_' + str(tmp_start_road_node_id)
                    if key in self.road_node_id_pair2stop_id_dict or equa_key in self.road_node_id_pair2stop_id_dict:
                        if key in self.road_node_id_pair2stop_id_dict:
                            stop_id_list.extend(self.road_node_id_pair2stop_id_dict[key])
                        else:
                            stop_id_list.extend(self.road_node_id_pair2stop_id_dict[equa_key])
                        break
                    for sub_next_road_node_id in self.roads_net_dict[tmp_next_road_node_id]['next_node_list']:
                        if sub_next_road_node_id != tmp_start_road_node_id:
                            tmp_start_road_node_id = tmp_next_road_node_id
                            tmp_next_road_node_id = sub_next_road_node_id
                            break
            start_road_node = self.roads_net_dict[road_pos['end_road_node_id']]
            for next_road_node_id in start_road_node['next_node_list']:
                if next_road_node_id == road_pos['start_road_node_id']:
                    continue
                tmp_start_road_node_id = road_pos['end_road_node_id']
                tmp_next_road_node_id = next_road_node_id
                while True:
                    key = str(tmp_start_road_node_id) + '_' + str(tmp_next_road_node_id)
                    equa_key = str(tmp_next_road_node_id) + '_' + str(tmp_start_road_node_id)
                    if key in self.road_node_id_pair2stop_id_dict or equa_key in self.road_node_id_pair2stop_id_dict:
                        if key in self.road_node_id_pair2stop_id_dict:
                            stop_id_list.extend(self.road_node_id_pair2stop_id_dict[key])
                        else:
                            stop_id_list.extend(self.road_node_id_pair2stop_id_dict[equa_key])
                        break
                    for sub_next_road_node_id in self.roads_net_dict[tmp_next_road_node_id]['next_node_list']:
                        if sub_next_road_node_id != tmp_start_road_node_id:
                            tmp_start_road_node_id = tmp_next_road_node_id
                            tmp_next_road_node_id = sub_next_road_node_id
                            break
        min_dis = 1e5
        target_stop_id = -1
        for stop_id in stop_id_list:
            cur_dis = dis_p2p(self.road_pos2pos(road_pos), self.stops_net_dict[stop_id]['coordxy'])
            if cur_dis < min_dis:
                target_stop_id = stop_id
                min_dis = cur_dis
        return target_stop_id

    def gen_next_status(self, last_status, last_status_length, signal, wait_step_num, call_step_num,
                        ugv_final_pos):
        next_status = -1
        # init status
        if last_status == 0:
            # start wait
            if signal > 0:
                next_status = 2
            # start deliver
            else:
                next_status = 1
        # delivering
        elif last_status == 1:
            # start wait
            if signal > 0:
                next_status = 2
            # keep deliver
            else:
                next_status = 1
        # waiting
        elif last_status == 2:
            if last_status_length >= wait_step_num:
                # start call
                next_status = 3
            else:
                # keep wait
                next_status = 2
        # calling
        elif last_status == 3:
            if last_status_length >= call_step_num:
                # start wait
                if signal > 0:
                    next_status = 2
                # start deliver
                else:
                    next_status = 1
            else:
                # keep call
                next_status = 3
        if self.pos2cell_id(ugv_final_pos) not in self.uav_cellset:
            next_status = 1
        return next_status

    def ugv_move_ca_cm(self, ugv, action4ugv_move):
        cur_vector2target_pos = (action4ugv_move[0], action4ugv_move[1])
        ugv_cur_road_pos = copy.deepcopy(ugv.final_road_pos)
        ugv_cur_pos = self.road_pos2pos(ugv_cur_road_pos)
        move_dis = 0
        while True:
            target_road_node_id_list = []
            if abs(ugv_cur_road_pos['progress']) < 1e-5:
                target_road_node_id_list = copy.deepcopy(
                    self.roads_net_dict[ugv_cur_road_pos['start_road_node_id']]['next_node_list'])
            elif abs(1 - ugv_cur_road_pos['progress']) < 1e-5:
                target_road_node_id_list = copy.deepcopy(
                    self.roads_net_dict[ugv_cur_road_pos['end_road_node_id']]['next_node_list'])
            else:
                target_road_node_id_list.append(ugv_cur_road_pos['start_road_node_id'])
                target_road_node_id_list.append(ugv_cur_road_pos['end_road_node_id'])

            # choose a road_node to go
            choosen_road_node_id = -1
            max_projected_length = 0
            for target_road_node_id in target_road_node_id_list:
                vector2target_road_node = (self.roads_net_dict[target_road_node_id]['coordxy'][0] - ugv_cur_pos[0],
                                           self.roads_net_dict[target_road_node_id]['coordxy'][1] - ugv_cur_pos[1])
                projected_length = vector_dot_product(cur_vector2target_pos, vector2target_road_node)
                if projected_length > max_projected_length:
                    max_projected_length = projected_length
                    choosen_road_node_id = target_road_node_id
            if choosen_road_node_id == -1:
                break
            else:
                vector2choosen_road_node = (self.roads_net_dict[choosen_road_node_id]['coordxy'][0] - ugv_cur_pos[0],
                                            self.roads_net_dict[choosen_road_node_id]['coordxy'][1] - ugv_cur_pos[1])
                dis2choosen_road_node = dis_p2p(ugv_cur_pos, self.roads_net_dict[choosen_road_node_id]['coordxy'])
                left_move_dis2choosen_road_node = min(
                    vector_length_counter(vector1_project2_vector2(cur_vector2target_pos, vector2choosen_road_node)),
                    self.env_conf['max_ugv_move_dis_each_step'] - move_dis)
                # cannot reach choosen road_node
                if left_move_dis2choosen_road_node < dis2choosen_road_node:
                    if abs(ugv_cur_road_pos['progress']) < 1e-5:
                        if len(ugv.passed_road_node_id_list) == 0 or ugv.passed_road_node_id_list[-1] != \
                                ugv_cur_road_pos['start_road_node_id']:
                            ugv.passed_road_node_id_list.append(ugv_cur_road_pos['start_road_node_id'])
                        ugv_cur_road_pos['end_road_node_id'] = choosen_road_node_id
                        choosen_road_length = dis_p2p(
                            self.roads_net_dict[ugv_cur_road_pos['start_road_node_id']]['coordxy'],
                            self.roads_net_dict[ugv_cur_road_pos['end_road_node_id']]['coordxy'])
                        ugv_cur_road_pos['progress'] = left_move_dis2choosen_road_node / choosen_road_length
                    elif abs(1 - ugv_cur_road_pos['progress']) < 1e-5:
                        ugv_cur_road_pos['start_road_node_id'] = ugv_cur_road_pos['end_road_node_id']
                        ugv_cur_road_pos['end_road_node_id'] = choosen_road_node_id
                        if len(ugv.passed_road_node_id_list) == 0 or ugv.passed_road_node_id_list[-1] != \
                                ugv_cur_road_pos['start_road_node_id']:
                            ugv.passed_road_node_id_list.append(ugv_cur_road_pos['start_road_node_id'])
                        choosen_road_length = dis_p2p(
                            self.roads_net_dict[ugv_cur_road_pos['start_road_node_id']]['coordxy'],
                            self.roads_net_dict[ugv_cur_road_pos['end_road_node_id']]['coordxy'])
                        ugv_cur_road_pos['progress'] = left_move_dis2choosen_road_node / choosen_road_length
                    else:
                        choosen_road_length = dis_p2p(
                            self.roads_net_dict[ugv_cur_road_pos['start_road_node_id']]['coordxy'],
                            self.roads_net_dict[ugv_cur_road_pos['end_road_node_id']]['coordxy'])
                        if choosen_road_node_id == ugv_cur_road_pos['start_road_node_id']:
                            ugv_cur_road_pos['progress'] -= left_move_dis2choosen_road_node / choosen_road_length
                        else:
                            ugv_cur_road_pos['progress'] += left_move_dis2choosen_road_node / choosen_road_length
                    break
                # can reach choosen road_node
                else:
                    ugv_cur_road_pos['start_road_node_id'] = choosen_road_node_id
                    ugv_cur_road_pos['end_road_node_id'] = self.roads_net_dict[choosen_road_node_id]['next_node_list'][
                        0]
                    ugv_cur_road_pos['progress'] = 0
                    ugv.passed_road_node_id_list.append(choosen_road_node_id)
                    ugv_cur_pos = self.road_pos2pos(ugv_cur_road_pos)
                    move_dis += dis2choosen_road_node
                    u = 1 - dis2choosen_road_node / vector_length_counter(
                        vector1_project2_vector2(cur_vector2target_pos, vector2choosen_road_node))
                    cur_vector2target_pos = (cur_vector2target_pos[0] * u, cur_vector2target_pos[1] * u)
        ugv.final_road_pos = copy.deepcopy(ugv_cur_road_pos)

        # update ugv_cur_stop_id
        target_stop_id = self.road_pos2stop_id(ugv.final_road_pos)
        if ugv.cur_stop_id != target_stop_id:
            stops_net_SP_key = str(ugv.cur_stop_id) + '_' + str(target_stop_id)
            shortest_path = self.stops_net_SP_dict[stops_net_SP_key]['shortest_path']
            visited_pois = np.nonzero(np.sum(self.stops_pois_AdjMatrix[shortest_path], axis=0))[0]
            ugv.cur_loc_poi_value_array[visited_pois] = copy.deepcopy(self.poi_cur_value_array[visited_pois])
            ugv.obs_X_B_u[:, 2] = np.sum(ugv.cur_loc_poi_value_array * self.stops_pois_AdjMatrix, axis=1)
        ugv.cur_stop_id = target_stop_id

    def ugv_move_ca_dm(self, ugv, action4ugv_move):
        cur_vector2target_pos = (action4ugv_move[0], action4ugv_move[1])
        ugv_cur_road_pos = self.stops_net_dict[ugv.cur_stop_id]['road_pos']
        ugv_cur_pos = self.road_pos2pos(ugv_cur_road_pos)
        move_dis = 0
        while True:
            target_road_node_id_list = []
            if abs(ugv_cur_road_pos['progress']) < 1e-5:
                target_road_node_id_list = copy.deepcopy(
                    self.roads_net_dict[ugv_cur_road_pos['start_road_node_id']]['next_node_list'])
            elif abs(1 - ugv_cur_road_pos['progress']) < 1e-5:
                target_road_node_id_list = copy.deepcopy(
                    self.roads_net_dict[ugv_cur_road_pos['end_road_node_id']]['next_node_list'])
            else:
                target_road_node_id_list.append(ugv_cur_road_pos['start_road_node_id'])
                target_road_node_id_list.append(ugv_cur_road_pos['end_road_node_id'])

            # choose a road_node to go
            choosen_road_node_id = -1
            max_projected_length = 0
            for target_road_node_id in target_road_node_id_list:
                vector2target_road_node = (self.roads_net_dict[target_road_node_id]['coordxy'][0] - ugv_cur_pos[0],
                                           self.roads_net_dict[target_road_node_id]['coordxy'][1] - ugv_cur_pos[1])
                projected_length = vector_dot_product(cur_vector2target_pos, vector2target_road_node)
                if projected_length > max_projected_length:
                    max_projected_length = projected_length
                    choosen_road_node_id = target_road_node_id
            if choosen_road_node_id == -1:
                break
            else:
                vector2choosen_road_node = (self.roads_net_dict[choosen_road_node_id]['coordxy'][0] - ugv_cur_pos[0],
                                            self.roads_net_dict[choosen_road_node_id]['coordxy'][1] - ugv_cur_pos[1])
                dis2choosen_road_node = dis_p2p(ugv_cur_pos, self.roads_net_dict[choosen_road_node_id]['coordxy'])
                left_move_dis2choosen_road_node = min(
                    vector_length_counter(vector1_project2_vector2(cur_vector2target_pos, vector2choosen_road_node)),
                    self.env_conf['max_ugv_move_dis_each_step'] - move_dis)
                # cannot reach choosen road_node
                if left_move_dis2choosen_road_node < dis2choosen_road_node:
                    if abs(ugv_cur_road_pos['progress']) < 1e-5:
                        ugv_cur_road_pos['end_road_node_id'] = choosen_road_node_id
                        choosen_road_length = dis_p2p(
                            self.roads_net_dict[ugv_cur_road_pos['start_road_node_id']]['coordxy'],
                            self.roads_net_dict[ugv_cur_road_pos['end_road_node_id']]['coordxy'])
                        ugv_cur_road_pos['progress'] = left_move_dis2choosen_road_node / choosen_road_length
                    elif abs(1 - ugv_cur_road_pos['progress']) < 1e-5:
                        ugv_cur_road_pos['start_road_node_id'] = ugv_cur_road_pos['end_road_node_id']
                        ugv_cur_road_pos['end_road_node_id'] = choosen_road_node_id
                        choosen_road_length = dis_p2p(
                            self.roads_net_dict[ugv_cur_road_pos['start_road_node_id']]['coordxy'],
                            self.roads_net_dict[ugv_cur_road_pos['end_road_node_id']]['coordxy'])
                        ugv_cur_road_pos['progress'] = left_move_dis2choosen_road_node / choosen_road_length
                    else:
                        choosen_road_length = dis_p2p(
                            self.roads_net_dict[ugv_cur_road_pos['start_road_node_id']]['coordxy'],
                            self.roads_net_dict[ugv_cur_road_pos['end_road_node_id']]['coordxy'])
                        if choosen_road_node_id == ugv_cur_road_pos['start_road_node_id']:
                            ugv_cur_road_pos['progress'] -= left_move_dis2choosen_road_node / choosen_road_length
                        else:
                            ugv_cur_road_pos['progress'] += left_move_dis2choosen_road_node / choosen_road_length
                    break
                # can reach choosen road_node
                else:
                    ugv_cur_road_pos['start_road_node_id'] = choosen_road_node_id
                    ugv_cur_road_pos['end_road_node_id'] = self.roads_net_dict[choosen_road_node_id]['next_node_list'][
                        0]
                    ugv_cur_road_pos['progress'] = 0
                    ugv_cur_pos = self.road_pos2pos(ugv_cur_road_pos)
                    move_dis += dis2choosen_road_node
                    u = 1 - dis2choosen_road_node / vector_length_counter(
                        vector1_project2_vector2(cur_vector2target_pos, vector2choosen_road_node))
                    cur_vector2target_pos = (cur_vector2target_pos[0] * u, cur_vector2target_pos[1] * u)

        # update ugv_cur_stop_id
        target_stop_id = self.road_pos2stop_id(ugv_cur_road_pos)
        if ugv.cur_stop_id != target_stop_id:
            stops_net_SP_key = str(ugv.cur_stop_id) + '_' + str(target_stop_id)
            shortest_path = self.stops_net_SP_dict[stops_net_SP_key]['shortest_path']
            visited_pois = np.nonzero(np.sum(self.stops_pois_AdjMatrix[shortest_path], axis=0))[0]
            ugv.cur_loc_poi_value_array[visited_pois] = copy.deepcopy(self.poi_cur_value_array[visited_pois])
            ugv.obs_X_B_u[:, 2] = np.sum(ugv.cur_loc_poi_value_array * self.stops_pois_AdjMatrix, axis=1)
        ugv.cur_stop_id = target_stop_id

    def ugv_move_da_dm(self, ugv, action4ugv_move):
        target_stop_id = int(action4ugv_move)
        if ugv.cur_stop_id != target_stop_id:
            stops_net_SP_key = str(ugv.cur_stop_id) + '_' + str(target_stop_id)
            shortest_path = self.stops_net_SP_dict[stops_net_SP_key]['shortest_path']
            visited_pois = np.nonzero(np.sum(self.stops_pois_AdjMatrix[shortest_path], axis=0))[0]
            ugv.cur_loc_poi_value_array[visited_pois] = copy.deepcopy(self.poi_cur_value_array[visited_pois])
            ugv.obs_X_B_u[:, 2] = np.sum(ugv.cur_loc_poi_value_array * self.stops_pois_AdjMatrix, axis=1)
        ugv.cur_stop_id = target_stop_id


    def update_UGV_obs_X_B_u(self, ugv):
        visited_pois = np.nonzero(self.stops_pois_AdjMatrix[ugv.cur_stop_id])[0]
        ugv.cur_loc_poi_value_array[visited_pois] = copy.deepcopy(self.poi_cur_value_array[visited_pois])
        ugv.obs_X_B_u[:, 2] = np.sum(ugv.cur_loc_poi_value_array * self.stops_pois_AdjMatrix, axis=1)


    def check_whether_hit(self, start_pos, move_vector):
        whether_hit_flag = False
        move_dis = vector_length_counter(move_vector)
        check_point_num = int(move_dis / self.env_conf['uav_move_check_hit_gap']) + 1
        for check_point_id in range(check_point_num):
            check_point_move_dis = move_dis - check_point_id * self.env_conf['uav_move_check_hit_gap']
            check_point_pos = (start_pos[0] + move_vector[0] * check_point_move_dis / move_dis,
                               start_pos[1] + move_vector[1] * check_point_move_dis / move_dis)
            check_point_cell_id = self.pos2cell_id(check_point_pos)
            if check_point_cell_id not in self.uav_cellset:
                whether_hit_flag = True
                break
        return whether_hit_flag

    def check_whether_out_of_ugv(self, start_pos, move_vector, ugv_pos):
        out_of_ugv_flag = False
        uav_ugv_dis = dis_p2p((start_pos[0] + move_vector[0], start_pos[1] + move_vector[1]), ugv_pos)
        if uav_ugv_dis > self.env_conf['uav_ugv_max_dis']:
            out_of_ugv_flag = True
        return out_of_ugv_flag

    def pos2cell_id(self, pos):
        cell_x = int(pos[0] / self.dataset_conf['coordx_max'] * self.env_conf['uav_cellset_grid_size'])
        cell_y = int(pos[1] / self.dataset_conf['coordy_max'] * self.env_conf['uav_cellset_grid_size'])
        cell_id = str(cell_x) + '_' + str(cell_y)
        return cell_id

    def uav_move_and_collect(self, uav, action4UAV, ugv_pos):
        uav.final_fly_time += 1
        self.final_total_fly_time += 1
        # uav is alive (has energy)
        if abs(uav.final_energy) > 1e-5:
            uav_move_dis_capability = min(uav.final_energy / self.env_conf['uav_move_energy_consume_ratio'],
                                          self.env_conf['max_uav_move_dis_each_step'])
            origin_move_vector = (action4UAV[0], action4UAV[1])
            origin_move_dis = vector_length_counter(origin_move_vector)
            move_dis = min(uav_move_dis_capability, origin_move_dis)
            move_vector = (action4UAV[0] / origin_move_dis * move_dis, action4UAV[1] / origin_move_dis * move_dis)
            hit_flag = self.check_whether_hit(uav.final_pos, move_vector)
            out_of_ugv_flag = self.check_whether_out_of_ugv(uav.final_pos, move_vector, ugv_pos)
            if hit_flag:
                uav.final_hit += 1
                self.final_total_hit += 1
            elif out_of_ugv_flag:
                uav.final_out_of_ugv += 1
                self.final_total_out_of_ugv += 1
            else:
                # move and consume energy
                uav_final_pos_tmp = (uav.final_pos[0] + move_vector[0], uav.final_pos[1] + move_vector[1])
                uav.final_pos = uav_final_pos_tmp
                uav.final_energy -= move_dis * self.env_conf['uav_move_energy_consume_ratio']
                uav.final_energy_consumption += move_dis * self.env_conf['uav_move_energy_consume_ratio']
                self.final_energy_consumption += move_dis * self.env_conf['uav_move_energy_consume_ratio']
                if abs(uav.final_energy) < 1e-5:
                    uav.final_energy = 0

                wheather_collect_data_flag = False
                uav_cell_id = self.pos2cell_id(uav.final_pos)
                if uav_cell_id in self.uav_cell2poi_dict:
                    available_poi_list = self.uav_cell2poi_dict[uav_cell_id]
                    uav.final_poi_visit_time[available_poi_list] += 1
                    self.final_poi_visit_time[available_poi_list] += 1
                    # collect
                    for poi_id in available_poi_list:
                        if self.poi_cur_value_array[poi_id] > 0:
                            wheather_collect_data_flag = True
                        if self.poi_cur_value_array[poi_id] < self.env_conf['uav_collect_speed_per_poi']:
                            uav.final_data_collection[poi_id] += self.poi_cur_value_array[poi_id]
                            self.poi_cur_value_array[poi_id] = 0
                        else:
                            uav.final_data_collection[poi_id] += self.env_conf['uav_collect_speed_per_poi']
                            self.poi_cur_value_array[poi_id] -= self.env_conf['uav_collect_speed_per_poi']
                if wheather_collect_data_flag:
                    uav.final_collect_data_time += 1
                    self.final_total_collect_data_time += 1

    def uav_back_to_ugv(self, uav, ugv):
        if self.method_conf['ugv_trace_type'] == 'roads_net':
            uav.final_pos = self.road_pos2pos(ugv.final_road_pos)
        elif self.method_conf['ugv_trace_type'] == 'stops_net':
            uav.final_pos = self.stops_net_dict[ugv.cur_stop_id]['coordxy']
        uav.final_energy = self.env_conf['uav_init_energy']

    def uav_delivered_by_ugv(self, uav, ugv):
        if self.method_conf['ugv_trace_type'] == 'roads_net':
            uav.final_pos = self.road_pos2pos(ugv.final_road_pos)
        elif self.method_conf['ugv_trace_type'] == 'stops_net':
            uav.final_pos = self.stops_net_dict[ugv.cur_stop_id]['coordxy']

    def step(self, actions4UGV_UAVs_Group_list):
        self.poi_last_value_array = copy.deepcopy(self.poi_cur_value_array)
        for UGV_UAVs_Group_id, actions4UGV_UAVs_Group in enumerate(actions4UGV_UAVs_Group_list):
            action4UGV = actions4UGV_UAVs_Group['UGV']
            UGV_UAVs_Group = self.UGV_UAVs_Group_list[UGV_UAVs_Group_id]
            ugv = UGV_UAVs_Group.ugv
            if self.method_conf['ugv_trace_type'] == 'roads_net':
                ugv.passed_road_node_id_list = []
            # gen next_status according to signal
            if isinstance(action4UGV, type(None)):
                signal = None
            else:
                signal = action4UGV[-1]
            if self.method_conf['ugv_trace_type'] == 'roads_net':
                ugv_final_pos = self.road_pos2pos(ugv.final_road_pos)
            elif self.method_conf['ugv_trace_type'] == 'stops_net':
                ugv_final_pos = self.stops_net_dict[ugv.cur_stop_id]['coordxy']
            UGV_UAVs_Group.next_status = self.gen_next_status(UGV_UAVs_Group.last_status,
                                                              UGV_UAVs_Group.last_status_length, signal,
                                                              UGV_UAVs_Group.wait_step_num,
                                                              UGV_UAVs_Group.call_step_num,
                                                              ugv_final_pos)

            # update UGV and UAVs cur info
            if UGV_UAVs_Group.next_status == 1:
                # do action for UGV
                if self.method_conf['ugv_action_type'] == 'continue' and self.method_conf[
                    'ugv_move_type'] == 'continue':
                    self.ugv_move_ca_cm(ugv, action4UGV[:-1])
                elif self.method_conf['ugv_action_type'] == 'continue' and self.method_conf[
                    'ugv_move_type'] == 'discrete':
                    self.ugv_move_ca_dm(ugv, action4UGV[:-1])
                elif self.method_conf['ugv_action_type'] == 'discrete' and self.method_conf[
                    'ugv_move_type'] == 'discrete':
                    self.ugv_move_da_dm(ugv, action4UGV[0])
                # UAVs deliverd by UGV
                for uav in UGV_UAVs_Group.uav_list:
                    self.uav_delivered_by_ugv(uav, ugv)
            elif UGV_UAVs_Group.next_status == 2:
                # do actions for UAVs
                actions4UAVs = actions4UGV_UAVs_Group['UAVs']
                for uav_id, action4UAV in enumerate(actions4UAVs):
                    uav = UGV_UAVs_Group.uav_list[uav_id]
                    self.uav_move_and_collect(uav, action4UAV, ugv_final_pos)
            elif UGV_UAVs_Group.next_status == 3:
                # update obs_X_B_u of UGV
                self.update_UGV_obs_X_B_u(ugv)
                self.final_total_relax_time += len(UGV_UAVs_Group.uav_list)
                # UAVs back to UGV
                for uav in UGV_UAVs_Group.uav_list:
                    self.uav_back_to_ugv(uav, ugv)
                    if uav.final_collect_data_time - uav.episode_log_info_dict['final_collect_data_time'][-(self.env_conf['wait_step_num'] + 1)] > 0:
                        self.final_total_eff_relax_time += 1

            # update UGV_UAVs_Group cur info
            if UGV_UAVs_Group.next_status != UGV_UAVs_Group.last_status:
                UGV_UAVs_Group.last_status = UGV_UAVs_Group.next_status
                UGV_UAVs_Group.last_status_length = 1
                UGV_UAVs_Group.next_status = None
            else:
                UGV_UAVs_Group.last_status_length += 1
                UGV_UAVs_Group.next_status = None

            if self.method_conf['ugv_trace_type'] == 'roads_net':
                ugv.final_passed_road_node_id_list.append(copy.deepcopy(ugv.passed_road_node_id_list))
            # add UGV and UAVs log info
            ugv.add_log_info()
            for uav in UGV_UAVs_Group.uav_list:
                uav.add_log_info()
            # add UGV_UAVs_Group log info
            UGV_UAVs_Group.add_log_info()
        self.add_log_info()
        self.cur_step += 1
