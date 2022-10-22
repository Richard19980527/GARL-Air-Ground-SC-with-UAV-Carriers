from .network import *
from log.sublog import *
from env.env import *
from .sub_rollout_manager import *
from .obs_generator import *


def subp(process_id,
         log_path,
         shared_ifdone,
         dataset_conf,
         env_conf,
         method_conf,
         log_conf,
         ):
    with torch.no_grad():
        global_dict_init()
        set_global_dict_value('dataset_conf', dataset_conf)
        set_global_dict_value('env_conf', env_conf)
        set_global_dict_value('method_conf', method_conf)
        set_global_dict_value('log_conf', log_conf)

        env = Env()
        loc_ugv_network = UGV_Network(env.stop_num)
        loc_ugv_network.eval()
        loc_uav_network = UAV_Network()
        loc_uav_network.eval()

        # create action temp
        actions4UGV_UAVs_Group_list_temp = []
        for UGV_UAVs_Group_id in range(env_conf['UGV_UAVs_Group_num']):
            actions4UGV_UAVs_Group = {}
            actions4UGV_UAVs_Group['UGV'] = None
            actions4UGV_UAVs_Group['UAVs'] = []
            actions4UGV_UAVs_Group_list_temp.append(copy.deepcopy(actions4UGV_UAVs_Group))

        sub_rollout_manager = SubRolloutManager(env=env)
        obs_generator = ObsGenerator(env=env)

        # add sub_rollouts for ugvs and uavs
        rollout_element_name_list4ugv = ['obs_X_B_u_s', 'obs_S_u_s', 'obs_u_stopid_vector_s',
                                         'obs_neighbor_stopids_vector_s', 'obs_LMatrix_s',
                                         'obs_action_mask_s', 'obs_u_x_s', 'msg_H_neighbor_ls_s', 'msg_G_neighbor_ls_s',
                                         'value_s', 'action_s', 'action_log_prob_s',
                                         'reward_s', 'return_s', 'step_id_s']
        rollout_element_name_list4uav = ['loc_obs_s', 'value_s', 'action_s', 'action_log_prob_s', 'reward_s',
                                         'return_s',
                                         'step_id_s']
        for ugv_id in range(env_conf['UGV_UAVs_Group_num']):
            ugv_sub_rollout_id = str(ugv_id)
            sub_rollout_manager.add_sub_rollout(rollout_element_name_list4ugv, 'UGV', ugv_sub_rollout_id)
            for uav_id in range(env_conf['uav_num_each_group']):
                uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                sub_rollout_manager.add_sub_rollout(rollout_element_name_list4uav, 'UAV', uav_sub_rollout_id)

        sublog = SubLog(process_id, log_path=log_path)
        sub_iter_counter = 0
        while sub_iter_counter < method_conf['train_iter']:
            while True:
                if not shared_ifdone.value:
                    fix_random_seed(get_global_dict_value('method_conf')['seed'] + process_id)
                    # reset sub_rollouts
                    sub_rollout_manager.reset_sub_rollouts()
                    # sync global model to local model
                    loc_ugv_network.load_state_dict(
                        torch.load(os.path.join(log_path, 'cur_ugv_model.pth'), map_location=torch.device('cpu')))
                    loc_uav_network.load_state_dict(
                        torch.load(os.path.join(log_path, 'cur_uav_model.pth'), map_location=torch.device('cpu')))
                    ################################## interact with env ####################################
                    # reset env
                    env.reset()
                    obs_generator.reset()
                    for step_id in range(env_conf['max_step_num']):
                        #################### prepare step_UGV_input_dict start! ####################
                        step_UGV_input_dict = {}
                        for UGV_UAVs_Group_id in range(env_conf['UGV_UAVs_Group_num']):
                            # for UGV
                            ugv_id = UGV_UAVs_Group_id
                            # gen obs for ugv
                            obs_X_B_u, obs_S_u, obs_neighbor_uids, obs_u_stopid_vector, obs_neighbor_stopids_vector, obs_LMatrix, obs_action_mask, obs_u_x = obs_generator.gen_ugv_obs_gnn(
                                ugv_id)
                            # gen h_tilde for ugv
                            h_tilde_s = loc_ugv_network.extract_feature_s_from_loc_obs_s(
                                torch.tensor(obs_X_B_u, dtype=torch.float32).unsqueeze(0),
                                torch.tensor(obs_S_u, dtype=torch.float32).unsqueeze(0),
                                torch.tensor(obs_u_stopid_vector, dtype=torch.float32).unsqueeze(0),
                                torch.tensor(obs_neighbor_stopids_vector, dtype=torch.float32).unsqueeze(0),
                                torch.tensor(obs_LMatrix, dtype=torch.float32).unsqueeze(0),
                            )
                            step_UGV_input_dict[ugv_id] = {}
                            step_UGV_input_dict[ugv_id]['obs_X_B_u'] = obs_X_B_u
                            step_UGV_input_dict[ugv_id]['obs_S_u'] = obs_S_u
                            step_UGV_input_dict[ugv_id]['obs_neighbor_uids'] = obs_neighbor_uids
                            step_UGV_input_dict[ugv_id]['obs_u_stopid_vector'] = obs_u_stopid_vector
                            step_UGV_input_dict[ugv_id]['obs_neighbor_stopids_vector'] = obs_neighbor_stopids_vector
                            step_UGV_input_dict[ugv_id]['obs_LMatrix'] = obs_LMatrix
                            step_UGV_input_dict[ugv_id]['obs_action_mask'] = obs_action_mask
                            step_UGV_input_dict[ugv_id]['obs_u_x'] = obs_u_x
                            step_UGV_input_dict[ugv_id]['h_tilde'] = np.array(h_tilde_s[0])
                            step_UGV_input_dict[ugv_id]['h_u_list'] = [np.array(h_tilde_s[0])]
                            step_UGV_input_dict[ugv_id]['g_u_list'] = [obs_u_x]
                            step_UGV_input_dict[ugv_id]['msg_H_neighbor_list'] = []
                            step_UGV_input_dict[ugv_id]['msg_G_neighbor_list'] = []
                        for comm_id in range(method_conf['Comm_layer_num']):
                            for UGV_UAVs_Group_id in range(env_conf['UGV_UAVs_Group_num']):
                                # for UGV
                                ugv_id = UGV_UAVs_Group_id
                                h_u = step_UGV_input_dict[ugv_id]['h_u_list'][comm_id]
                                g_u = step_UGV_input_dict[ugv_id]['g_u_list'][comm_id]
                                msg_H_neighbor_list = []
                                msg_G_neighbor_list = []
                                for neighbor_uid in list(step_UGV_input_dict[ugv_id]['obs_neighbor_uids']):
                                    msg_H_neighbor_list.append(step_UGV_input_dict[neighbor_uid]['h_u_list'][comm_id])
                                    msg_G_neighbor_list.append(step_UGV_input_dict[neighbor_uid]['g_u_list'][comm_id])
                                msg_H_neighbor = np.stack(msg_H_neighbor_list, axis=0)
                                msg_G_neighbor = np.stack(msg_G_neighbor_list, axis=0)
                                step_UGV_input_dict[ugv_id]['msg_H_neighbor_list'].append(msg_H_neighbor)
                                step_UGV_input_dict[ugv_id]['msg_G_neighbor_list'].append(msg_G_neighbor)
                                h_u_s_prime, g_u_s_prime = loc_ugv_network.comm_variant.comm_with_neighbors(comm_id, torch.tensor(h_u, dtype=torch.float32).unsqueeze(0), torch.tensor(g_u, dtype=torch.float32).unsqueeze(0), torch.tensor(msg_H_neighbor, dtype=torch.float32).unsqueeze(0), torch.tensor(msg_G_neighbor, dtype=torch.float32).unsqueeze(0))
                                step_UGV_input_dict[ugv_id]['h_u_list'].append(np.array(h_u_s_prime[0]))
                                step_UGV_input_dict[ugv_id]['g_u_list'].append(np.array(g_u_s_prime[0]))
                        #################### prepare step_UGV_input_dict finish! ####################
                        actions4UGV_UAVs_Group_list = copy.deepcopy(actions4UGV_UAVs_Group_list_temp)
                        for UGV_UAVs_Group_id in range(env_conf['UGV_UAVs_Group_num']):
                            # for UGV
                            ugv_id = UGV_UAVs_Group_id
                            use_loc_ugv_network_flag = True
                            last_status = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].last_status
                            last_status_length = env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].last_status_length
                            if last_status == 2 or last_status == 3 and last_status_length < env_conf['call_step_num']:
                                use_loc_ugv_network_flag = False
                            if use_loc_ugv_network_flag:
                                # gen action for ugv
                                H_u = np.stack(step_UGV_input_dict[ugv_id]['h_u_list'], axis=0)
                                G_u = np.stack(step_UGV_input_dict[ugv_id]['g_u_list'], axis=0)
                                value_s, action_s, action_log_prob_s = loc_ugv_network.get_action_s(
                                    torch.tensor(step_UGV_input_dict[ugv_id]['obs_X_B_u'], dtype=torch.float32).unsqueeze(0),
                                    torch.tensor(H_u, dtype=torch.float32).unsqueeze(0),
                                    torch.tensor(G_u, dtype=torch.float32).unsqueeze(0),
                                    torch.tensor(step_UGV_input_dict[ugv_id]['obs_action_mask'], dtype=torch.float32).unsqueeze(0),
                                )
                                actions4UGV_UAVs_Group_list[UGV_UAVs_Group_id]['UGV'] = np.array(action_s[0], dtype=np.float32)
                                msg_H_neighbor_ls = np.stack(step_UGV_input_dict[ugv_id]['msg_H_neighbor_list'], axis=0)
                                msg_G_neighbor_ls = np.stack(step_UGV_input_dict[ugv_id]['msg_G_neighbor_list'], axis=0)

                                ugv_sub_rollout_id = str(ugv_id)
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'obs_X_B_u_s', step_UGV_input_dict[ugv_id]['obs_X_B_u'])
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'obs_S_u_s', step_UGV_input_dict[ugv_id]['obs_S_u'])
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'obs_u_stopid_vector_s', step_UGV_input_dict[ugv_id]['obs_u_stopid_vector'])
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'obs_neighbor_stopids_vector_s', step_UGV_input_dict[ugv_id]['obs_neighbor_stopids_vector'])
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'obs_LMatrix_s', step_UGV_input_dict[ugv_id]['obs_LMatrix'])
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'obs_u_x_s', step_UGV_input_dict[ugv_id]['obs_u_x'])
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'msg_H_neighbor_ls_s', msg_H_neighbor_ls)
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'msg_G_neighbor_ls_s', msg_G_neighbor_ls)
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'obs_action_mask_s', step_UGV_input_dict[ugv_id]['obs_action_mask'])

                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'value_s', np.array(value_s[0], dtype=np.float32))
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'action_s', np.array(action_s[0], dtype=np.float32))
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'action_log_prob_s', np.array(action_log_prob_s[0], dtype=np.float32))
                                sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                                    'step_id_s', step_id)
                            # for UAVs
                            if isinstance(actions4UGV_UAVs_Group_list[UGV_UAVs_Group_id]['UGV'], type(None)):
                                signal = None
                            else:
                                signal = actions4UGV_UAVs_Group_list[UGV_UAVs_Group_id]['UGV'][-1]
                            if method_conf['ugv_trace_type'] == 'roads_net':
                                ugv_final_pos = env.road_pos2pos(env.UGV_UAVs_Group_list[
                                                                      UGV_UAVs_Group_id].ugv.final_road_pos)
                            elif method_conf['ugv_trace_type'] == 'stops_net':
                                ugv_final_pos = env.stops_net_dict[env.UGV_UAVs_Group_list[
                                    UGV_UAVs_Group_id].ugv.cur_stop_id]['coordxy']
                            next_status = env.gen_next_status(last_status, last_status_length, signal,
                                                              env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].wait_step_num,
                                                              env.UGV_UAVs_Group_list[UGV_UAVs_Group_id].call_step_num,
                                                              ugv_final_pos)
                            need_action4uav_flag = False
                            if next_status == 2:
                                need_action4uav_flag = True
                            if need_action4uav_flag:
                                for uav_id in range(env_conf['uav_num_each_group']):
                                    # gen obs for uav
                                    uav_loc_obs = obs_generator.gen_uav_obs(ugv_id, uav_id)
                                    # gen action for uav
                                    value_s, action_s, action_log_prob_s = loc_uav_network.get_action_s(
                                        torch.tensor(uav_loc_obs, dtype=torch.float32).unsqueeze(0))
                                    actions4UGV_UAVs_Group_list[UGV_UAVs_Group_id]['UAVs'].append(
                                        np.array(action_s[0], dtype=np.float32) * env_conf[
                                            'max_uav_move_dis_each_step'] / (2 ** 0.5))
                                    uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                                    sub_rollout_manager.sub_rollout_dict['UAV'][
                                        uav_sub_rollout_id].append_rollout_element('loc_obs_s', uav_loc_obs)
                                    sub_rollout_manager.sub_rollout_dict['UAV'][
                                        uav_sub_rollout_id].append_rollout_element('value_s', np.array(value_s[0],
                                                                                                       dtype=np.float32))
                                    sub_rollout_manager.sub_rollout_dict['UAV'][
                                        uav_sub_rollout_id].append_rollout_element('action_s', np.array(action_s[0],
                                                                                                        dtype=np.float32))
                                    sub_rollout_manager.sub_rollout_dict['UAV'][
                                        uav_sub_rollout_id].append_rollout_element('action_log_prob_s',
                                                                                   np.array(action_log_prob_s[0],
                                                                                            dtype=np.float32))
                                    sub_rollout_manager.sub_rollout_dict['UAV'][
                                        uav_sub_rollout_id].append_rollout_element('step_id_s', step_id)
                            else:
                                add_new_episode_buffer_flag = False
                                uav_id = 0
                                uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                                if step_id - 1 in sub_rollout_manager.sub_rollout_dict['UAV'][
                                    uav_sub_rollout_id].sub_episode_buffer_list[-1]['step_id_s']:
                                    add_new_episode_buffer_flag = True
                                if add_new_episode_buffer_flag:
                                    for uav_id in range(env_conf['uav_num_each_group']):
                                        # gen obs for uav
                                        uav_loc_obs = obs_generator.gen_uav_obs(ugv_id, uav_id)
                                        # gen value for uav
                                        value_s = loc_uav_network.get_value_s(
                                            torch.tensor(uav_loc_obs, dtype=torch.float32).unsqueeze(0))
                                        uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                                        sub_rollout_manager.sub_rollout_dict['UAV'][
                                            uav_sub_rollout_id].append_rollout_element('value_s', np.array(value_s[0],
                                                                                                           dtype=np.float32))
                                        sub_rollout_manager.sub_rollout_dict['UAV'][
                                            uav_sub_rollout_id].append_rollout_element('step_id_s', step_id)
                                        sub_rollout_manager.sub_rollout_dict['UAV'][
                                            uav_sub_rollout_id].add_new_sub_episode_buffer()
                        # env step
                        env.step(actions4UGV_UAVs_Group_list)
                        obs_generator.step()
                    # finish gen SubRollout.sub_episode_buffer_list
                    #################### prepare step_UGV_input_dict start! ####################
                    step_UGV_input_dict = {}
                    for UGV_UAVs_Group_id in range(env_conf['UGV_UAVs_Group_num']):
                        # for UGV
                        ugv_id = UGV_UAVs_Group_id
                        # gen obs for ugv
                        obs_X_B_u, obs_S_u, obs_neighbor_uids, obs_u_stopid_vector, obs_neighbor_stopids_vector, obs_LMatrix, obs_action_mask, obs_u_x = obs_generator.gen_ugv_obs_gnn(
                            ugv_id)
                        # gen h_tilde for ugv
                        h_tilde_s = loc_ugv_network.extract_feature_s_from_loc_obs_s(
                            torch.tensor(obs_X_B_u, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(obs_S_u, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(obs_u_stopid_vector, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(obs_neighbor_stopids_vector, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(obs_LMatrix, dtype=torch.float32).unsqueeze(0),
                        )
                        step_UGV_input_dict[ugv_id] = {}
                        step_UGV_input_dict[ugv_id]['obs_X_B_u'] = obs_X_B_u
                        step_UGV_input_dict[ugv_id]['obs_S_u'] = obs_S_u
                        step_UGV_input_dict[ugv_id]['obs_neighbor_uids'] = obs_neighbor_uids
                        step_UGV_input_dict[ugv_id]['obs_u_stopid_vector'] = obs_u_stopid_vector
                        step_UGV_input_dict[ugv_id]['obs_neighbor_stopids_vector'] = obs_neighbor_stopids_vector
                        step_UGV_input_dict[ugv_id]['obs_LMatrix'] = obs_LMatrix
                        step_UGV_input_dict[ugv_id]['obs_action_mask'] = obs_action_mask
                        step_UGV_input_dict[ugv_id]['obs_u_x'] = obs_u_x
                        step_UGV_input_dict[ugv_id]['h_tilde'] = np.array(h_tilde_s[0])
                        step_UGV_input_dict[ugv_id]['h_u_list'] = [np.array(h_tilde_s[0])]
                        step_UGV_input_dict[ugv_id]['g_u_list'] = [obs_u_x]
                        step_UGV_input_dict[ugv_id]['msg_H_neighbor_list'] = []
                        step_UGV_input_dict[ugv_id]['msg_G_neighbor_list'] = []
                    for comm_id in range(method_conf['Comm_layer_num']):
                        for UGV_UAVs_Group_id in range(env_conf['UGV_UAVs_Group_num']):
                            # for UGV
                            ugv_id = UGV_UAVs_Group_id
                            h_u = step_UGV_input_dict[ugv_id]['h_u_list'][comm_id]
                            g_u = step_UGV_input_dict[ugv_id]['g_u_list'][comm_id]
                            msg_H_neighbor_list = []
                            msg_G_neighbor_list = []
                            for neighbor_uid in list(step_UGV_input_dict[ugv_id]['obs_neighbor_uids']):
                                msg_H_neighbor_list.append(step_UGV_input_dict[neighbor_uid]['h_u_list'][comm_id])
                                msg_G_neighbor_list.append(step_UGV_input_dict[neighbor_uid]['g_u_list'][comm_id])
                            msg_H_neighbor = np.stack(msg_H_neighbor_list, axis=0)
                            msg_G_neighbor = np.stack(msg_G_neighbor_list, axis=0)
                            step_UGV_input_dict[ugv_id]['msg_H_neighbor_list'].append(msg_H_neighbor)
                            step_UGV_input_dict[ugv_id]['msg_G_neighbor_list'].append(msg_G_neighbor)
                            h_u_s_prime, g_u_s_prime = loc_ugv_network.comm_variant.comm_with_neighbors(comm_id,
                                                                                           torch.tensor(h_u,
                                                                                                        dtype=torch.float32).unsqueeze(
                                                                                               0), torch.tensor(g_u,
                                                                                                                dtype=torch.float32).unsqueeze(
                                    0), torch.tensor(msg_H_neighbor, dtype=torch.float32).unsqueeze(0),
                                                                                           torch.tensor(
                                                                                               msg_G_neighbor,
                                                                                               dtype=torch.float32).unsqueeze(
                                                                                               0))
                            step_UGV_input_dict[ugv_id]['h_u_list'].append(np.array(h_u_s_prime[0]))
                            step_UGV_input_dict[ugv_id]['g_u_list'].append(np.array(g_u_s_prime[0]))
                    #################### prepare step_UGV_input_dict finish! ####################
                    for UGV_UAVs_Group_id in range(env_conf['UGV_UAVs_Group_num']):
                        # for UGV
                        ugv_id = UGV_UAVs_Group_id
                        # gen value for ugv
                        H_u = np.stack(step_UGV_input_dict[ugv_id]['h_u_list'], axis=0)
                        G_u = np.stack(step_UGV_input_dict[ugv_id]['g_u_list'], axis=0)
                        value_s, _, _ = loc_ugv_network.get_action_s(
                            torch.tensor(step_UGV_input_dict[ugv_id]['obs_X_B_u'],
                                         dtype=torch.float32).unsqueeze(0),
                            torch.tensor(H_u, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(G_u, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(step_UGV_input_dict[ugv_id]['obs_action_mask'],
                                         dtype=torch.float32).unsqueeze(0),
                        )
                        ugv_sub_rollout_id = str(ugv_id)
                        sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                            'value_s', np.array(value_s[0], dtype=np.float32))
                        sub_rollout_manager.sub_rollout_dict['UGV'][ugv_sub_rollout_id].append_rollout_element(
                            'step_id_s', env_conf['max_step_num'])
                        # for UAV
                        add_last_step_flag = False
                        uav_id = 0
                        uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                        if env_conf['max_step_num'] - 1 in \
                                sub_rollout_manager.sub_rollout_dict['UAV'][uav_sub_rollout_id].sub_episode_buffer_list[
                                    -1][
                                    'step_id_s']:
                            add_last_step_flag = True
                        if add_last_step_flag:
                            for uav_id in range(env_conf['uav_num_each_group']):
                                # gen obs for uav
                                uav_loc_obs = obs_generator.gen_uav_obs(ugv_id, uav_id)
                                # gen value for uav
                                value_s = loc_uav_network.get_value_s(
                                    torch.tensor(uav_loc_obs, dtype=torch.float32).unsqueeze(0))
                                uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                                sub_rollout_manager.sub_rollout_dict['UAV'][uav_sub_rollout_id].append_rollout_element(
                                    'value_s', np.array(value_s[0], dtype=np.float32))
                                sub_rollout_manager.sub_rollout_dict['UAV'][uav_sub_rollout_id].append_rollout_element(
                                    'step_id_s', env_conf['max_step_num'])
                    sub_rollout_manager.delete_last_empty_sub_episode_buffer()
                    # gen reward
                    sub_rollout_manager.gen_rewards()
                    # gen returns
                    sub_rollout_manager.gen_returns()
                    ################################## sublog work ####################################
                    sublog.record_sub_rollout_dict(sub_rollout_manager)
                    sublog.gen_metrics_result(sub_iter_counter, env)
                    sublog.record_metrics_result()
                    shared_ifdone.value = True
                    sub_iter_counter += 1
                    break
