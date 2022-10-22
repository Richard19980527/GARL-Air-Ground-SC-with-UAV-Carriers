from log.mainlog import *
from .PPO import *
from .rollout_manager import *
from .subp import *


def main():
    dataset_conf = get_global_dict_value('dataset_conf')
    env_conf = get_global_dict_value('env_conf')
    method_conf = get_global_dict_value('method_conf')
    log_conf = get_global_dict_value('log_conf')

    mp.set_start_method("spawn", force=True)
    mainlog = MainLog()

    tmp_env = Env()
    stop_num = tmp_env.stop_num
    ugv_network = UGV_Network(stop_num).to('cuda:' + str(method_conf['gpu_id']))
    ugv_network.eval()
    ugv_agent = PPO(ac=ugv_network)

    uav_network = UAV_Network().to('cuda:' + str(method_conf['gpu_id']))
    uav_network.eval()
    uav_agent = PPO(ac=uav_network)

    mainlog.save_model(model_name='cur_ugv_model', model=ugv_network)
    mainlog.save_model(model_name='cur_uav_model', model=uav_network)

    # add sub_rollouts for ugvs and uavs
    rollout_element_name_list4ugv_rollout = ['obs_X_B_u_s', 'obs_S_u_s', 'obs_u_stopid_vector_s',
                                             'obs_neighbor_stopids_vector_s', 'obs_LMatrix_s',
                                             'obs_action_mask_s', 'obs_u_x_s', 'msg_H_neighbor_ls_s',
                                             'msg_G_neighbor_ls_s', 'value_s', 'action_s', 'action_log_prob_s',
                                             'return_s']
    rollout_element_name_list4uav_rollout = ['loc_obs_s', 'value_s', 'action_s', 'action_log_prob_s', 'return_s']
    ugv_rollout = RolloutManager(rollout_element_name_list4ugv_rollout)
    uav_rollout = RolloutManager(rollout_element_name_list4uav_rollout)

    shared_ifdone_list = [mp.Value('b', False) for _ in range(method_conf['env_num'])]

    simulator_processes = []
    for env_id in range(method_conf['env_num']):
        p = mp.Process(target=subp,
                       args=(env_id,
                             mainlog.log_path,
                             shared_ifdone_list[env_id],
                             dataset_conf,
                             env_conf,
                             method_conf,
                             log_conf,
                             )
                       )
        simulator_processes.append(p)
        p.start()

    max_avg_eff = 0
    for iter_id in range(method_conf['train_iter']):
        ################################## gen samples ####################################
        while True:
            global_ifdone = 0
            for shared_ifdone in shared_ifdone_list:
                if shared_ifdone.value:
                    global_ifdone += 1
                else:
                    break
            if global_ifdone == method_conf['env_num']:
                ################################## save good model ####################################
                mainlog.load_envs_info()
                avg_eff = np.mean(mainlog.envs_info['eff'][-50:])
                if avg_eff > max_avg_eff:
                    max_avg_eff = avg_eff
                    mainlog.save_model(model_name='avg_good_ugv_model', model=ugv_network)
                    mainlog.save_model(model_name='avg_good_uav_model', model=uav_network)
                ################################## load sub_rollouts into rollout ##################################
                ugv_rollout.reset()
                uav_rollout.reset()
                for env_id in range(method_conf['env_num']):
                    sub_rollout_dict = mainlog.load_sub_rollout_dict(env_id)
                    for ugv_id in range(env_conf['UGV_UAVs_Group_num']):
                        ugv_sub_rollout_id = str(ugv_id)
                        ugv_rollout.append_episodes_data(
                            sub_rollout_dict['UGV'][ugv_sub_rollout_id].sub_episode_buffer_list)
                        for uav_id in range(env_conf['uav_num_each_group']):
                            uav_sub_rollout_id = str(ugv_id) + '_' + str(uav_id)
                            uav_rollout.append_episodes_data(
                                sub_rollout_dict['UAV'][uav_sub_rollout_id].sub_episode_buffer_list)
                # delete sub_rollouts
                if iter_id == method_conf['train_iter'] - 1:
                    for env_id in range(method_conf['env_num']):
                        mainlog.delete_sub_rollout_dict(env_id)
                ################################## update params ####################################
                ugv_network.train()
                ugv_agent.update(ugv_rollout, iter_id)
                ugv_network.eval()
                uav_network.train()
                uav_agent.update(uav_rollout, iter_id)
                uav_network.eval()
                ################################## mainlog work ####################################
                mainlog.save_model(model_name='cur_ugv_model', model=ugv_network)
                mainlog.save_model(model_name='cur_uav_model', model=uav_network)
                # ------------------------------------------------------------
                eff = np.mean(mainlog.envs_info['eff'][iter_id])
                fairness = np.mean(mainlog.envs_info['fairness'][iter_id])
                dcr = np.mean(mainlog.envs_info['dcr'][iter_id])
                ecr = np.mean(mainlog.envs_info['ecr'][iter_id])
                cor = np.mean(mainlog.envs_info['cor'][iter_id])

                report_str = '#' * 100 \
                             + '\n' \
                             + 'iter: ' + str(iter_id) \
                             + ' max_avg_eff: ' + str(np.round(max_avg_eff, 5)) \
                             + ' eff: ' + str(np.round(eff, 5)) \
                             + ' fairness: ' + str(np.round(fairness, 5)) \
                             + ' dcr: ' + str(np.round(dcr, 5)) \
                             + ' cor: ' + str(np.round(cor, 5)) \
                             + ' ecr: ' + str(np.round(ecr, 5)) \
                             + '\n'
                print(report_str)
                for shared_ifdone in shared_ifdone_list:
                    shared_ifdone.value = False
                break

    for p in simulator_processes:
        p.join()
