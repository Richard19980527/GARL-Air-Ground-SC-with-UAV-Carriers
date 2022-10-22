from util import *

ENV_CONF = {
    # field conf
    'margin_width': 10,  # meter
    # ROAD conf
    'road_width': 20,  # meter
    'roads_cellset_grid_size': 1000,
    # STOP conf
    'stop_gap': 100,  # meter
    'stop_poi_max_dis': 250,  # meter
    # UGV_UAVs_Group conf
    'UGV_UAVs_Group_num': 4,
    'uav_num_each_group': 2,
    'uav_ugv_max_dis': 250,  # meter
    # UGV conf
    'max_ugv_move_dis_each_step': 400,  # meter
    # UAV conf
    'uav_cellset_grid_size': 1000,
    'uav_sensing_range': 60,  # meter
    'uav_init_energy': 10.,
    'uav_collect_speed_per_poi': 5,
    'max_uav_move_dis_each_step': 100.,  # meter
    'uav_move_energy_consume_ratio': 0.01,
    'uav_move_check_hit_gap': 2,  # meter
    # POI conf
    'poi_value_max': 3 * 4,
    'poi_value_min': 2 * 4,
    # EPISODE conf
    'max_step_num': 100,
    'wait_step_num': 10,
    'call_step_num': 1,
}
