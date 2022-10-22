from util import *
from env.env import *


class ObsGenerator:
    def __init__(self, env):
        # load confs
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.env = env

        self.uav_glb_obs4loc_obs_obsts = self.gen_glb_obs_obsts(self.method_conf['uav_glb_obs4loc_obs_grid_size'])
        self.uav_glb_obs4loc_obs_roads = self.gen_glb_obs_roads(self.method_conf['uav_glb_obs4loc_obs_grid_size'])
        self.uav_glb_obs4loc_obs_pois, self.poi2uav_glb_obs4loc_obs_grid_dict = self.gen_glb_obs_pois(
            self.method_conf['uav_glb_obs4loc_obs_grid_size'])

    def gen_glb_obs_obsts(self, glb_obs_grid_size):
        glb_obs_obsts = np.zeros([glb_obs_grid_size, glb_obs_grid_size], dtype=np.float32)
        for i in range(glb_obs_grid_size):
            cell_x = int((i + 0.5) / glb_obs_grid_size * self.env_conf['uav_cellset_grid_size'])
            for j in range(glb_obs_grid_size):
                cell_y = int((j + 0.5) / glb_obs_grid_size * self.env_conf['uav_cellset_grid_size'])
                cell_key = str(cell_x) + '_' + str(cell_y)
                if cell_key in self.env.uav_cellset:
                    glb_obs_obsts[i, j] = 1
        return glb_obs_obsts

    def gen_glb_obs_roads(self, glb_obs_grid_size):
        glb_obs_roads = np.zeros([glb_obs_grid_size, glb_obs_grid_size], dtype=np.float32)
        for roads_cell_id in self.env.roads_cellset:
            roads_cell_x = float(roads_cell_id.split('_')[0])
            roads_cell_y = float(roads_cell_id.split('_')[1])
            glb_obs_grid_x = int(
                (roads_cell_x + 0.5) / self.env_conf['roads_cellset_grid_size'] * glb_obs_grid_size)
            glb_obs_grid_y = int(
                (roads_cell_y + 0.5) / self.env_conf['roads_cellset_grid_size'] * glb_obs_grid_size)
            glb_obs_roads[glb_obs_grid_x, glb_obs_grid_y] = 1
        return glb_obs_roads

    def gen_glb_obs_pois(self, glb_obs_grid_size):
        glb_obs_pois = np.zeros([glb_obs_grid_size, glb_obs_grid_size], dtype=np.float32)
        poi2glb_obs_grid_dict_pre = {}
        for poi_id in self.env.poi2cell_dict:
            poi2glb_obs_grid_dict_pre[poi_id] = set()
            for cell_id in self.env.poi2cell_dict[poi_id]:
                cell_x = float(cell_id.split('_')[0])
                cell_y = float(cell_id.split('_')[1])
                glb_obs_grid_x = int((cell_x + 0.5) / self.env_conf['uav_cellset_grid_size'] * glb_obs_grid_size)
                glb_obs_grid_y = int((cell_y + 0.5) / self.env_conf['uav_cellset_grid_size'] * glb_obs_grid_size)
                glb_obs_grid_id = str(glb_obs_grid_x) + '_' + str(glb_obs_grid_y)
                poi2glb_obs_grid_dict_pre[poi_id].add(glb_obs_grid_id)
        poi2glb_obs_grid_dict = {}
        for poi_id in poi2glb_obs_grid_dict_pre:
            poi2glb_obs_grid_dict[poi_id] = [[], []]
            for glb_obs_grid_id in poi2glb_obs_grid_dict_pre[poi_id]:
                glb_obs_grid_x = int(glb_obs_grid_id.split('_')[0])
                glb_obs_grid_y = int(glb_obs_grid_id.split('_')[1])
                poi2glb_obs_grid_dict[poi_id][0].append(glb_obs_grid_x)
                poi2glb_obs_grid_dict[poi_id][1].append(glb_obs_grid_y)
        for poi_id in poi2glb_obs_grid_dict:
            glb_obs_pois[poi2glb_obs_grid_dict[poi_id][0], poi2glb_obs_grid_dict[poi_id][1]] += \
                self.env.poi2coordxy_value_dict[poi_id]['value'] / self.env_conf['poi_value_max']
        return glb_obs_pois, poi2glb_obs_grid_dict

    def reset(self):
        self.cur_uav_glb_obs4loc_obs_pois = copy.deepcopy(self.uav_glb_obs4loc_obs_pois)

    def step(self):
        poi_delta_value_array = self.env.poi_last_value_array - self.env.poi_cur_value_array
        for poi_id in np.nonzero(poi_delta_value_array)[0]:
            self.cur_uav_glb_obs4loc_obs_pois[
                self.poi2uav_glb_obs4loc_obs_grid_dict[poi_id][0], self.poi2uav_glb_obs4loc_obs_grid_dict[poi_id][1]] -= \
                poi_delta_value_array[poi_id] / self.env_conf['poi_value_max']

    def pos2grid_xy(self, pos, glb_obs_grid_size):
        grid_x = int(pos[0] / self.dataset_conf['coordx_max'] * glb_obs_grid_size)
        grid_y = int(pos[1] / self.dataset_conf['coordy_max'] * glb_obs_grid_size)
        grid_xy = (grid_x, grid_y)
        return grid_xy

    def glb_obs2loc_obs(self, glb_obs, center_grid_xy, loc_obs_shape):
        obs_channel_num = glb_obs.shape[0]
        glb_obs_shape = glb_obs.shape[1]
        loc_obs = np.zeros([obs_channel_num, loc_obs_shape, loc_obs_shape], dtype=np.float32)

        half_loc_obs_shape = int(loc_obs_shape / 2)

        glb_obs_x_min = center_grid_xy[0] - half_loc_obs_shape
        glb_obs_y_min = center_grid_xy[1] - half_loc_obs_shape
        glb_obs_x_max = glb_obs_x_min + loc_obs_shape
        glb_obs_y_max = glb_obs_y_min + loc_obs_shape

        glb_obs_x_min = np.clip(glb_obs_x_min, 0, glb_obs_shape - 1)
        glb_obs_y_min = np.clip(glb_obs_y_min, 0, glb_obs_shape - 1)
        glb_obs_x_max = np.clip(glb_obs_x_max, 1, glb_obs_shape)
        glb_obs_y_max = np.clip(glb_obs_y_max, 1, glb_obs_shape)

        loc_obs_x_min = half_loc_obs_shape - center_grid_xy[0]
        loc_obs_y_min = half_loc_obs_shape - center_grid_xy[1]
        loc_obs_x_min = np.clip(loc_obs_x_min, 0, loc_obs_shape - 1)
        loc_obs_y_min = np.clip(loc_obs_y_min, 0, loc_obs_shape - 1)

        loc_obs_x_max = loc_obs_x_min + glb_obs_x_max - glb_obs_x_min
        loc_obs_y_max = loc_obs_y_min + glb_obs_y_max - glb_obs_y_min
        loc_obs_x_max = np.clip(loc_obs_x_max, 1, loc_obs_shape)
        loc_obs_y_max = np.clip(loc_obs_y_max, 1, loc_obs_shape)

        loc_obs[:, loc_obs_x_min:loc_obs_x_max, loc_obs_y_min:loc_obs_y_max] = glb_obs[:, glb_obs_x_min:glb_obs_x_max,
                                                                               glb_obs_y_min:glb_obs_y_max]
        return loc_obs

    def gen_ugv_obs_gnn(self, ugv_id):
        ugv = self.env.UGV_UAVs_Group_list[ugv_id].ugv
        obs_X_B_u = copy.deepcopy(ugv.obs_X_B_u)
        obs_X_B_u[:, 0] /= self.dataset_conf['coordx_max']
        obs_X_B_u[:, 1] /= self.dataset_conf['coordy_max']
        obs_X_B_u[:, 2] /= self.env.stop_max_value
        obs_u_stopid = ugv.cur_stop_id
        obs_u_stopid_vector = np.zeros(self.env.stop_num, dtype=np.float32)
        obs_neighbor_stopids_vector = np.zeros(self.env.stop_num, dtype=np.float32)
        obs_u_stopid_vector[obs_u_stopid] = 1
        obs_neighbor_uids = []
        obs_neighbor_stopids = []
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            if UGV_UAVs_Group_id != ugv_id:
                ugv_prime = self.env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].ugv
                obs_neighbor_uids.append(UGV_UAVs_Group_id)
                obs_neighbor_stopids.append(ugv_prime.cur_stop_id)
                obs_neighbor_stopids_vector[ugv_prime.cur_stop_id] += 1
        obs_neighbor_uids = np.array(obs_neighbor_uids, dtype=np.int)
        obs_neighbor_stopids = np.array(obs_neighbor_stopids, dtype=np.int)
        obs_S_u = self.env.pre_S_Matrix[:, obs_u_stopid] - np.mean(self.env.pre_S_Matrix[:, obs_neighbor_stopids],
                                                                   axis=1)
        obs_LMatrix = self.env.stops_net_LaplacianMatrix
        obs_action_mask = self.env.ugv_move_mask_Matrix[ugv.cur_stop_id]
        obs_u_x = np.array(self.env.stops_net_dict[ugv.cur_stop_id]['coordxy'], dtype=np.float32)
        obs_u_x[0] /= self.dataset_conf['coordx_max']
        obs_u_x[1] /= self.dataset_conf['coordy_max']
        return obs_X_B_u, obs_S_u, obs_neighbor_uids, obs_u_stopid_vector, obs_neighbor_stopids_vector, obs_LMatrix, obs_action_mask, obs_u_x

    def gen_uav_obs(self, ugv_id, uav_id):
        # gen uav_loc_obs
        uav_glb_obs4loc_obs = np.zeros(
            [self.method_conf['uav_loc_obs_channel_num'], self.method_conf['uav_glb_obs4loc_obs_grid_size'],
             self.method_conf['uav_glb_obs4loc_obs_grid_size']], dtype=np.float32)
        uav_glb_obs4loc_obs[0] = copy.deepcopy(self.uav_glb_obs4loc_obs_obsts)
        uav_glb_obs4loc_obs[1] = copy.deepcopy(self.uav_glb_obs4loc_obs_roads)
        uav_glb_obs4loc_obs[2] = copy.deepcopy(self.cur_uav_glb_obs4loc_obs_pois)
        for UGV_UAVs_Group_id in range(self.env_conf['UGV_UAVs_Group_num']):
            uav_pos = self.env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id].final_pos
            uav_energy = self.env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].uav_list[uav_id].final_energy
            uav_grid_xy = self.pos2grid_xy(uav_pos, self.method_conf['uav_glb_obs4loc_obs_grid_size'])
            if UGV_UAVs_Group_id != ugv_id:
                uav_glb_obs4loc_obs[3][uav_grid_xy[0], uav_grid_xy[1]] = uav_energy / self.env_conf['uav_init_energy']
            else:
                uav_glb_obs4loc_obs[4][uav_grid_xy[0], uav_grid_xy[1]] = uav_energy / self.env_conf['uav_init_energy']
        center_grid_xy = self.pos2grid_xy(
            self.env.UGV_UAVs_Group_list[ugv_id].uav_list[uav_id].final_pos,
            self.method_conf['uav_glb_obs4loc_obs_grid_size'])
        uav_loc_obs = self.glb_obs2loc_obs(uav_glb_obs4loc_obs, center_grid_xy, self.method_conf['uav_loc_obs_shape'])

        return uav_loc_obs
