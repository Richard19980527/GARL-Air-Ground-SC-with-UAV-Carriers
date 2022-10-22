from util import *


class SubLog:
    def __init__(self, process_id=None, log_path=None):
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.episode_metrics_result = {}
        self.episode_metrics_result['eff'] = []
        self.episode_metrics_result['fairness'] = []
        self.episode_metrics_result['dcr'] = []
        self.episode_metrics_result['hit'] = []
        self.episode_metrics_result['ec'] = []
        self.episode_metrics_result['ecr'] = []
        self.episode_metrics_result['cor'] = []

        self.root_log_path = log_path
        self.log_path = os.path.join(log_path, 'process_' + str(process_id))

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def record_sub_rollout_dict(self, sub_rollout_manager):
        sub_rollout_dict_path = os.path.join(self.log_path, 'sub_rollout_dict.npy')
        np.save(sub_rollout_dict_path, sub_rollout_manager.sub_rollout_dict)

    def gen_metrics_result(self, iter_id, env):
        # fairness
        fairness = 0.0
        final_poi_visit_time = np.clip(env.episode_log_info_dict['final_poi_visit_time'][-1], 0, 2)
        square_of_sum = np.square(np.sum(final_poi_visit_time))
        sum_of_square = np.sum(np.square(final_poi_visit_time))
        if sum_of_square > 1e-5:
            fairness = square_of_sum / sum_of_square / final_poi_visit_time.shape[0]
        self.episode_metrics_result['fairness'].append(fairness)

        # data_collection_ratio (dcr)
        dcr = np.sum(env.poi_init_value_array - env.poi_cur_value_array) / np.sum(env.poi_init_value_array)
        self.episode_metrics_result['dcr'].append(dcr)

        # hit
        hit = env.final_total_hit
        self.episode_metrics_result['hit'].append(hit)

        # energy_consumption (ec)
        ec = env.final_energy_consumption
        self.episode_metrics_result['ec'].append(ec)

        # energy_consumption_ratio (ecr)
        ecr = ec / env.ec_upper_bound
        self.episode_metrics_result['ecr'].append(ecr)

        # UGV_UAV_cooperation_ratio (cor)
        if env.final_total_relax_time > 0:
            cor = env.final_total_eff_relax_time / env.final_total_relax_time
        else:
            cor = 0
        self.episode_metrics_result['cor'].append(cor)

        # eff
        eff = 0.0
        if ecr > min_value:
            eff = fairness * dcr * cor / ecr
        self.episode_metrics_result['eff'].append(eff)

    def record_metrics_result(self):
        np.save(self.log_path + '/episode_metrics_result.npy', self.episode_metrics_result)
