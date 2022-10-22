from methods.distributions import *
from methods.net_tools import *


class MC_GCN_Layer(nn.Module):
    def __init__(self):
        super(MC_GCN_Layer, self).__init__()
        self.method_conf = get_global_dict_value('method_conf')
        self.stop_hidden_size = self.method_conf['stop_hidden_size']
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.linear_F = init_(nn.Linear(self.stop_hidden_size, self.stop_hidden_size))
        self.linear_1 = init_(nn.Linear(self.stop_hidden_size, self.stop_hidden_size))
        self.my_Sigmoid = nn.Sigmoid()
        self.my_softmax = nn.Softmax(dim=-1)
        self.my_relu = nn.ReLU()

    def forward(self, H_B_u_s, obs_S_u_s, obs_u_stopid_vector_s, obs_neighbor_stopids_vector_s, obs_LMatrix_s):
        BatchSize, stop_num, stop_hidden_size = H_B_u_s.shape
        obs_H_B_u_s_prime_F = self.linear_F(H_B_u_s.view(BatchSize * stop_num, stop_hidden_size)).view(BatchSize,
                                                                                                       stop_num,
                                                                                                       stop_hidden_size)
        MC_integrated_h_B_u_s = torch.sum(
            torch.mul(H_B_u_s, obs_u_stopid_vector_s.unsqueeze(dim=-1).expand(BatchSize, stop_num, stop_hidden_size)),
            dim=1) - torch.mul(torch.sum(torch.mul(H_B_u_s,
                                                   obs_neighbor_stopids_vector_s.unsqueeze(dim=-1).expand(BatchSize,
                                                                                                          stop_num,
                                                                                                          stop_hidden_size)),
                                         dim=1),
                               (1 / torch.sum(obs_neighbor_stopids_vector_s, dim=1)).unsqueeze(dim=-1).expand(BatchSize,
                                                                                                              stop_hidden_size))
        F_u_s = torch.bmm(obs_H_B_u_s_prime_F, MC_integrated_h_B_u_s.unsqueeze(dim=-1)).squeeze(dim=-1)
        C_u_s = self.my_softmax(torch.mul(self.my_Sigmoid(obs_S_u_s), self.my_Sigmoid(F_u_s)))
        obs_H_B_u_s_prime_1 = self.linear_1(H_B_u_s.view(BatchSize * stop_num, stop_hidden_size)).view(BatchSize,
                                                                                                       stop_num,
                                                                                                       stop_hidden_size)
        H_B_u_s_prime = self.my_relu(
            torch.bmm(torch.mul(C_u_s.unsqueeze(dim=-1).expand(BatchSize, stop_num, stop_num), obs_LMatrix_s),
                      obs_H_B_u_s_prime_1))
        return H_B_u_s_prime


class MC_GCN(nn.Module):
    def __init__(self):
        super(MC_GCN, self).__init__()
        self.method_conf = get_global_dict_value('method_conf')
        self.stop_size = self.method_conf['stop_size']
        self.stop_hidden_size = self.method_conf['stop_hidden_size']
        self.layer_num = self.method_conf['GNN_layer_num']
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.linear_init = init_(nn.Linear(self.stop_size, self.stop_hidden_size))
        self.gcn_layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.gcn_layers.append(MC_GCN_Layer())
        self.linear_H = init_(nn.Linear(self.stop_hidden_size * (self.layer_num + 1), 1))
        self.my_relu = nn.ReLU()

    def forward(self, obs_X_B_u_s, obs_S_u_s, obs_u_stopid_vector_s, obs_neighbor_stopids_vector_s, obs_LMatrix_s):
        BatchSize, stop_num, stop_obs_size = obs_X_B_u_s.shape
        H_B_u_s_list = []
        H_B_u_s = self.linear_init(obs_X_B_u_s.view(BatchSize * stop_num, stop_obs_size)).view(BatchSize, stop_num,
                                                                                               self.stop_hidden_size)
        H_B_u_s_list.append(H_B_u_s)
        for layer_id in range(self.layer_num):
            H_B_u_s = self.gcn_layers[layer_id](H_B_u_s, obs_S_u_s, obs_u_stopid_vector_s,
                                                obs_neighbor_stopids_vector_s, obs_LMatrix_s)
            H_B_u_s_list.append(H_B_u_s)
        integrated_H_B_u_s = torch.cat(H_B_u_s_list, dim=2)
        h_tilde_s = self.my_relu(
            self.linear_H(integrated_H_B_u_s.view(BatchSize * stop_num, -1)).view(BatchSize, stop_num))
        return h_tilde_s


class E_Comm_Layer(nn.Module):
    def __init__(self, ugv_hidden_size):
        super(E_Comm_Layer, self).__init__()
        self.method_conf = get_global_dict_value('method_conf')
        self.ugv_hidden_size = ugv_hidden_size
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.linear_e = init_(nn.Linear(self.ugv_hidden_size, self.ugv_hidden_size))
        self.linear_h = init_(nn.Linear(self.ugv_hidden_size * 2, self.ugv_hidden_size))
        self.linear_m = init_(nn.Linear(self.ugv_hidden_size, 1))
        self.my_softmax = nn.Softmax(dim=-1)
        self.my_relu = nn.ReLU()
        self.my_Sigmoid = nn.Sigmoid()
        self.my_Tanh = nn.Tanh()

    def forward(self, h_u_s, g_u_s, msg_H_neighbor_s, msg_G_neighbor_s):
        BatchSize, neighbor_num, ugv_hidden_size = msg_H_neighbor_s.shape
        _, _, ugv_geo_hidden_size = msg_G_neighbor_s.shape
        # invariant part
        R_u_s = g_u_s.unsqueeze(dim=1).expand(BatchSize, neighbor_num, ugv_geo_hidden_size) - msg_G_neighbor_s + 1e-5
        R_u_s_caret = F.normalize(R_u_s, dim=2)
        Alpha_u_s = self.my_softmax(R_u_s_caret[:, :, 0] / R_u_s[:, :, 0])
        M_neighbor_s = self.my_relu(
            self.linear_e(msg_H_neighbor_s.contiguous().view(BatchSize * neighbor_num, -1)).view(BatchSize, neighbor_num, -1))
        integrated_M_neighbor_s = torch.sum(
            torch.mul(Alpha_u_s.unsqueeze(dim=2).expand(BatchSize, neighbor_num, ugv_hidden_size), M_neighbor_s), dim=1)
        h_u_s_prime = self.my_relu(self.linear_h(torch.cat([h_u_s, integrated_M_neighbor_s], dim=1)))
        # equivariant part
        g_u_s_tilde = self.my_Tanh(torch.sum(torch.mul(torch.mul(self.my_Sigmoid(
            self.linear_m(M_neighbor_s.view(BatchSize * neighbor_num, -1)).view(BatchSize, neighbor_num)),
            Alpha_u_s).unsqueeze(dim=2).expand(BatchSize,
                                               neighbor_num,
                                               ugv_geo_hidden_size),
                                                       R_u_s_caret), dim=1))
        g_u_s_prime = g_u_s + g_u_s_tilde * self.method_conf['g_u_s_tilde_factor']
        return h_u_s_prime, g_u_s_prime


class E_Comm(nn.Module):
    def __init__(self, stop_num):
        super(E_Comm, self).__init__()
        self.method_conf = get_global_dict_value('method_conf')
        self.stop_num = stop_num
        self.ugv_hidden_size = self.stop_num
        self.layer_num = self.method_conf['Comm_layer_num']
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.comm_layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.comm_layers.append(E_Comm_Layer(self.ugv_hidden_size))
        self.linear_inte_H = init_(nn.Linear(self.ugv_hidden_size * (self.layer_num + 1), self.ugv_hidden_size))
        self.linear_inte_G = init_(nn.Linear(2 * (self.layer_num + 1), 2))
        self.linear_z = init_(nn.Linear(2, 2))
        self.linear_u = init_(nn.Linear(self.ugv_hidden_size + self.stop_num, self.ugv_hidden_size))
        self.my_relu = nn.ReLU()
        self.my_softmax = nn.Softmax(dim=-1)

    def comm_with_neighbors(self, comm_id, h_u_s, g_u_s, msg_H_neighbor_s, msg_G_neighbor_s):
        h_u_s_prime, g_u_s_prime = self.comm_layers[comm_id](h_u_s, g_u_s, msg_H_neighbor_s, msg_G_neighbor_s)
        return h_u_s_prime, g_u_s_prime

    def readout(self, obs_X_B_u_s, H_u_s, G_u_s):
        BatchSize, item_num, ugv_hidden_size = H_u_s.shape
        integrated_H_u_s = self.my_relu(self.linear_inte_H(H_u_s.view(BatchSize, -1)))
        integrated_G_u_s = self.my_relu(self.linear_inte_G(G_u_s.view(BatchSize, -1)))
        z_u_s = self.my_softmax(torch.bmm(
            self.linear_z(obs_X_B_u_s[:, :, :2].view(BatchSize * self.stop_num, 2)).view(BatchSize, self.stop_num, 2),
            integrated_G_u_s.unsqueeze(dim=2)).squeeze(dim=2))
        readout_u_s = self.my_relu(self.linear_u(torch.cat([integrated_H_u_s, z_u_s], dim=1)))
        return readout_u_s

    def forward(self, obs_X_B_u_s, h_tilde_s, obs_u_x_s, msg_H_neighbor_ls_s, msg_G_neighbor_ls_s):
        H_u_s_list = []
        G_u_s_list = []
        h_u_s = h_tilde_s
        g_u_s = obs_u_x_s
        H_u_s_list.append(h_u_s)
        G_u_s_list.append(g_u_s)
        for comm_id in range(self.layer_num):
            h_u_s, g_u_s = self.comm_with_neighbors(comm_id, h_u_s, g_u_s, msg_H_neighbor_ls_s[:, comm_id],
                                                    msg_G_neighbor_ls_s[:, comm_id])
            H_u_s_list.append(h_u_s)
            G_u_s_list.append(g_u_s)
        H_u_s = torch.stack(H_u_s_list, dim=1)
        G_u_s = torch.stack(G_u_s_list, dim=1)
        readout_u_s = self.readout(obs_X_B_u_s, H_u_s, G_u_s)
        return readout_u_s


class UGV_Network(nn.Module):
    def __init__(self, stop_num):
        super(UGV_Network, self).__init__()
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.stop_num = stop_num
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.gcn_variant = MC_GCN()
        self.comm_variant = E_Comm(self.stop_num)
        self.distribution = MultiCategorical_Generator(self.stop_num, [self.stop_num, 2])
        self.critic_linear = init_(nn.Linear(self.stop_num, 1))

    def extract_feature_s_from_loc_obs_s(self, obs_X_B_u_s, obs_S_u_s, obs_u_stopid_vector_s,
                                         obs_neighbor_stopids_vector_s, obs_LMatrix_s):
        h_tilde_s = self.gcn_variant(obs_X_B_u_s, obs_S_u_s, obs_u_stopid_vector_s, obs_neighbor_stopids_vector_s,
                                     obs_LMatrix_s)
        return h_tilde_s

    def get_action_s(self, obs_X_B_u_s, H_u_s, G_u_s, obs_action_mask_s):
        actor_feature_s = self.comm_variant.readout(obs_X_B_u_s, H_u_s, G_u_s)
        value_s = self.critic_linear(actor_feature_s)
        distribution_instance_s = self.distribution(actor_feature_s, obs_action_mask_s)
        action_s = distribution_instance_s.sample()
        action_log_prob_s = distribution_instance_s.log_prob(action_s)
        return value_s, action_s, action_log_prob_s

    def get_value_s(self, obs_X_B_u_s, H_u_s, G_u_s):
        actor_feature_s = self.comm_variant.readout(obs_X_B_u_s, H_u_s, G_u_s)
        value_s = self.critic_linear(actor_feature_s)
        return value_s

    def evaluate_action_s(self, obs_X_B_u_s, obs_S_u_s, obs_u_stopid_vector_s, obs_neighbor_stopids_vector_s,
                          obs_LMatrix_s, obs_u_x_s, msg_H_neighbor_ls_s, msg_G_neighbor_ls_s, obs_action_mask_s,
                          action_s):
        h_tilde_s = self.gcn_variant(obs_X_B_u_s, obs_S_u_s, obs_u_stopid_vector_s, obs_neighbor_stopids_vector_s,
                                     obs_LMatrix_s)
        actor_feature_s = self.comm_variant(obs_X_B_u_s, h_tilde_s, obs_u_x_s, msg_H_neighbor_ls_s, msg_G_neighbor_ls_s)
        value_s = self.critic_linear(actor_feature_s)

        distribution_instance_s = self.distribution(actor_feature_s, obs_action_mask_s)
        action_log_prob_s = distribution_instance_s.log_prob(action_s)
        distribution_entropy_s = distribution_instance_s.entropy().mean()
        return value_s, distribution_entropy_s, action_log_prob_s


class UAV_Network(nn.Module):
    def __init__(self):
        super(UAV_Network, self).__init__()
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.hidden_size = self.method_conf['hidden_size']
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        cnn_init_ = lambda m: init(m,
                                   nn.init.orthogonal_,
                                   lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain('relu'))
        self.CNN4loc_obs = nn.Sequential(
            cnn_init_(nn.Conv2d(self.method_conf['uav_loc_obs_channel_num'], 32, 8, stride=4, padding=4)),
            nn.ReLU(),
            cnn_init_(nn.Conv2d(32, 32, 5, stride=1, padding=1)),
            nn.ReLU(),
            cnn_init_(nn.Conv2d(32, 32, 4, stride=1, padding=1)),
            nn.ReLU(),
            Flatten(),
            cnn_init_(nn.Linear(32 * 3 * 3, self.hidden_size)),
            nn.ReLU()
        )
        self.distribution = DiagGaussian(self.hidden_size, self.method_conf['uav_action_dim'])
        self.critic_linear = init_(nn.Linear(self.hidden_size, 1))

    def get_action_s(self, loc_obs_s):
        loc_actor_feature_s = self.CNN4loc_obs(loc_obs_s)
        actor_feature_s = loc_actor_feature_s
        value_s = self.critic_linear(actor_feature_s)
        distribution_instance_s = self.distribution(actor_feature_s)
        action_s = distribution_instance_s.sample()
        action_log_prob_s = distribution_instance_s.log_probs(action_s)
        return value_s, action_s, action_log_prob_s

    def get_value_s(self, loc_obs_s):
        loc_actor_feature_s = self.CNN4loc_obs(loc_obs_s)
        actor_feature_s = loc_actor_feature_s
        value_s = self.critic_linear(actor_feature_s)
        return value_s

    def evaluate_action_s(self, loc_obs_s, action_s):
        loc_actor_feature_s = self.CNN4loc_obs(loc_obs_s)
        actor_feature_s = loc_actor_feature_s
        value_s = self.critic_linear(actor_feature_s)

        distribution_instance_s = self.distribution(actor_feature_s)
        action_log_prob_s = distribution_instance_s.log_probs(action_s)
        distribution_entropy_s = distribution_instance_s.entropy().mean()
        return value_s, distribution_entropy_s, action_log_prob_s
