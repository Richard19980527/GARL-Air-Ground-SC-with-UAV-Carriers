from util import *
from methods.net_tools import *


class PPO:
    def __init__(self, ac):
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.ac = ac
        self.lr = self.method_conf['lr']
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr, eps=self.method_conf['eps'], weight_decay=1e-6)

    def update(self, rollout, iter_id):
        if len(rollout.buffer_dict['return_s']) == 0:
            return 0, 0, 0, 0
        advantage_s = np.array(rollout.buffer_dict['return_s']) - np.array(rollout.buffer_dict['value_s'])
        advantage_s = (advantage_s - advantage_s.mean()) / (advantage_s.std() + 1e-5)

        role = self.ac.__class__.__name__
        for _ in range(self.method_conf['buffer_replay_time']):
            data_generator = rollout.minibatch_generator(advantage_s, role)
            for sample_mini_batch in data_generator:
                if role == 'UGV_Network':
                    obs_X_B_u_mini_batch, obs_S_u_mini_batch, obs_u_stopid_vector_mini_batch, \
                    obs_neighbor_stopids_vector_mini_batch, obs_LMatrix_mini_batch, obs_action_mask_mini_batch, \
                    obs_u_x_mini_batch, msg_H_neighbor_ls_mini_batch, msg_G_neighbor_ls_mini_batch, \
                    action_mini_batch, value_mini_batch, return_mini_batch, \
                    old_action_s_log_prob_mini_batch, adv_targ_mini_batch = sample_mini_batch

                    obs_X_B_u_mini_batch = obs_X_B_u_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    obs_S_u_mini_batch = obs_S_u_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    obs_u_stopid_vector_mini_batch = obs_u_stopid_vector_mini_batch.to(
                        'cuda:' + str(self.method_conf['gpu_id']))
                    obs_neighbor_stopids_vector_mini_batch = obs_neighbor_stopids_vector_mini_batch.to(
                        'cuda:' + str(self.method_conf['gpu_id']))
                    obs_LMatrix_mini_batch = obs_LMatrix_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    obs_action_mask_mini_batch = obs_action_mask_mini_batch.to(
                        'cuda:' + str(self.method_conf['gpu_id']))
                    obs_u_x_mini_batch = obs_u_x_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    msg_H_neighbor_ls_mini_batch = msg_H_neighbor_ls_mini_batch.to(
                        'cuda:' + str(self.method_conf['gpu_id']))
                    msg_G_neighbor_ls_mini_batch = msg_G_neighbor_ls_mini_batch.to(
                        'cuda:' + str(self.method_conf['gpu_id']))

                    action_mini_batch = action_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    value_mini_batch = value_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    return_mini_batch = return_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    old_action_s_log_prob_mini_batch = old_action_s_log_prob_mini_batch.to(
                        'cuda:' + str(self.method_conf['gpu_id']))
                    adv_targ_mini_batch = adv_targ_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))

                    evl_value_s, dist_entropy_s, action_s_log_prob = self.ac.evaluate_action_s(obs_X_B_u_mini_batch,
                                                                                               obs_S_u_mini_batch,
                                                                                               obs_u_stopid_vector_mini_batch,
                                                                                               obs_neighbor_stopids_vector_mini_batch,
                                                                                               obs_LMatrix_mini_batch,
                                                                                               obs_u_x_mini_batch,
                                                                                               msg_H_neighbor_ls_mini_batch,
                                                                                               msg_G_neighbor_ls_mini_batch,
                                                                                               obs_action_mask_mini_batch,
                                                                                               action_mini_batch)
                elif role == 'UAV_Network':
                    loc_obs_mini_batch, action_mini_batch, value_mini_batch, return_mini_batch, \
                    old_action_s_log_prob_mini_batch, adv_targ_mini_batch = sample_mini_batch

                    loc_obs_mini_batch = loc_obs_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    action_mini_batch = action_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    value_mini_batch = value_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    return_mini_batch = return_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))
                    old_action_s_log_prob_mini_batch = old_action_s_log_prob_mini_batch.to(
                        'cuda:' + str(self.method_conf['gpu_id']))
                    adv_targ_mini_batch = adv_targ_mini_batch.to('cuda:' + str(self.method_conf['gpu_id']))

                    evl_value_s, dist_entropy_s, action_s_log_prob = self.ac.evaluate_action_s(
                        loc_obs_mini_batch, action_mini_batch)

                ratio = torch.exp(action_s_log_prob - old_action_s_log_prob_mini_batch)

                surr1 = ratio * adv_targ_mini_batch
                surr2 = torch.clamp(ratio, 1.0 - self.method_conf['clip_param'],
                                    1.0 + self.method_conf['clip_param']) * adv_targ_mini_batch
                action_loss = -torch.min(surr1, surr2).mean()

                if self.method_conf['use_clipped_value_loss']:
                    value_pred_clipped = value_mini_batch + \
                                         (evl_value_s - value_mini_batch).clamp(-self.method_conf['clip_param'],
                                                                                self.method_conf['clip_param'])
                    value_losses = (evl_value_s - return_mini_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_mini_batch).pow(2)
                    value_loss = .5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(return_mini_batch, evl_value_s)

                self.optimizer.zero_grad()
                loss = value_loss * self.method_conf['value_loss_coef'] + action_loss - dist_entropy_s * \
                       self.method_conf['entropy_coef']

                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.method_conf['max_grad_norm'])
                self.optimizer.step()
        self.lr = adjust_learning_rate(optimizer=self.optimizer, lr=self.lr, iter_id=iter_id)
