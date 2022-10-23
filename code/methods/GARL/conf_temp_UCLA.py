from util import *

METHOD_CONF = {
    'method_name': __file__.split('/')[-2],
    'gpu_id': 0,
    'seed': 2,
    'env_num': 1,
    # ---------algo------------
    'lr': 2.5e-4,
    'eps': 1e-5,
    'use_gae': True,
    'gamma': 0.7,
    'tau': 0.95,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.1,
    'max_grad_norm': 0.5,
    'use_clipped_value_loss': True,
    'clip_param': 0.1,
    # ---------buffer------------
    'mini_batch_size': 400,
    'buffer_replay_time': 2,
    # ----------train test-----------
    'train_iter': int(1e4),
    'continue_train_start_iter_id': 0,
    'test_num': 50,
    # ----------lr decay-----------
    'decay_rate': 0.9995,
    'decay_start_iter_id': 3000,
    # --------UGV setting-------------
    'ugv_obs_type': 'stops_net_obs',
    'ugv_obs_NN': 'GNN',
    'ugv_action_type': 'discrete',
    'ugv_move_type': 'discrete',
    'ugv_trace_type': 'stops_net',
    # --------obs size-------------
    'uav_glb_obs4loc_obs_grid_size': 400,
    'uav_loc_obs_shape': 20,
    'uav_loc_obs_channel_num': 5,
    # --------action-------------
    'uav_action_dim': 2,
    # --------network-------------
    'hidden_size': 256,
    # --------reward-------------
    'ugv_max_positive_reward': 10.0,
    'ugv_positive_factor': 0.75,
    'ugv_penalty_factor': 0.1,
    'uav_max_positive_reward': 10.0,
    'uav_positive_factor': 3.0,
    'uav_penalty_factor': 0.1,
    # --------GCN-------------
    'dsp_q': 5,
    'stop_size': 3,
    'stop_hidden_size': 8,
    'GNN_layer_num': 3,
    # --------Comm-------------
    'Comm_layer_num': 3,
    'g_u_s_tilde_factor': 0.1,
}
