from util import *


class RolloutManager:
    def __init__(self, rollout_element_name_list):
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.rollout_element_name_list = rollout_element_name_list
        self.reset()

    def reset(self):
        self.buffer_dict = {}
        for rollout_element_name in self.rollout_element_name_list:
            self.buffer_dict[rollout_element_name] = []

    def append_episodes_data(self, sub_episode_buffer_list):
        for sub_episode_buffer in sub_episode_buffer_list:
            for rollout_element_name in self.rollout_element_name_list:
                self.buffer_dict[rollout_element_name].extend(sub_episode_buffer[rollout_element_name])

    def minibatch_generator(self, advantage_s, role=None):
        N = len(self.buffer_dict['action_s'])
        sampler = BatchSampler(SubsetRandomSampler(range(N)), self.method_conf['mini_batch_size'], drop_last=False)
        batch_dict = {}
        for rollout_element_name in self.rollout_element_name_list:
            if rollout_element_name == 'action_log_prob_s':
                new_rollout_element_name = 'old_' + rollout_element_name
            else:
                new_rollout_element_name = rollout_element_name
            batch_dict[new_rollout_element_name] = torch.tensor(
                np.array(self.buffer_dict[rollout_element_name], dtype=np.float32),
                dtype=torch.float32)
        batch_dict['adv_targ_s'] = torch.tensor(advantage_s, dtype=torch.float32)

        for indices in sampler:
            if role == 'UGV_Network':
                obs_X_B_u_mini_batch = batch_dict['obs_X_B_u_s'][indices]
                obs_S_u_mini_batch = batch_dict['obs_S_u_s'][indices]
                obs_u_stopid_vector_mini_batch = batch_dict['obs_u_stopid_vector_s'][indices]
                obs_neighbor_stopids_vector_mini_batch = batch_dict['obs_neighbor_stopids_vector_s'][indices]
                obs_LMatrix_mini_batch = batch_dict['obs_LMatrix_s'][indices]
                obs_action_mask_mini_batch = batch_dict['obs_action_mask_s'][indices]
                obs_u_x_mini_batch = batch_dict['obs_u_x_s'][indices]
                msg_H_neighbor_ls_mini_batch = batch_dict['msg_H_neighbor_ls_s'][indices]
                msg_G_neighbor_ls_mini_batch = batch_dict['msg_G_neighbor_ls_s'][indices]

                action_mini_batch = batch_dict['action_s'][indices]
                value_mini_batch = batch_dict['value_s'][indices]
                return_mini_batch = batch_dict['return_s'][indices]
                old_action_log_prob_mini_batch = batch_dict['old_action_log_prob_s'][indices]
                adv_targ_mini_batch = batch_dict['adv_targ_s'][indices]
                yield obs_X_B_u_mini_batch, obs_S_u_mini_batch, obs_u_stopid_vector_mini_batch, \
                      obs_neighbor_stopids_vector_mini_batch, obs_LMatrix_mini_batch, obs_action_mask_mini_batch, \
                      obs_u_x_mini_batch, msg_H_neighbor_ls_mini_batch, msg_G_neighbor_ls_mini_batch, \
                      action_mini_batch, value_mini_batch, return_mini_batch, \
                      old_action_log_prob_mini_batch, adv_targ_mini_batch
            elif role == 'UAV_Network':
                loc_obs_mini_batch = batch_dict['loc_obs_s'][indices]
                action_mini_batch = batch_dict['action_s'][indices]
                value_mini_batch = batch_dict['value_s'][indices]
                return_mini_batch = batch_dict['return_s'][indices]
                old_action_log_prob_mini_batch = batch_dict['old_action_log_prob_s'][indices]
                adv_targ_mini_batch = batch_dict['adv_targ_s'][indices]
                yield loc_obs_mini_batch, action_mini_batch, value_mini_batch, return_mini_batch, \
                      old_action_log_prob_mini_batch, adv_targ_mini_batch
