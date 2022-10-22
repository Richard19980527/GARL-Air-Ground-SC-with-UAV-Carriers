from util import *


class MainLog:
    loss_dict = {}
    envs_info = {}

    def __init__(self, mode='train'):
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')

        self.mode = mode
        self.root_path = '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-3], 'log')
        self.dataset_name = self.dataset_conf['dataset_name']
        self.method_name = self.method_conf['method_name']
        if mode == 'train':
            self.log_path = os.path.join(self.root_path, self.dataset_name, self.method_name)
            if os.path.exists(self.log_path):
                shutil.rmtree(self.log_path)
            os.makedirs(self.log_path)
        elif mode == 'test':
            self.log_root_path = os.path.join(self.root_path, self.dataset_name, self.method_name)
            self.log_path = os.path.join(self.log_root_path, 'test')
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)

    def save_model(self, model_name, model):
        self._model_path = os.path.join(self.log_path, model_name + '.pth')
        torch.save(model.state_dict(), self._model_path)

    def load_envs_info(self, mode='train'):
        episode_metrics_result_list = []
        env_num = self.method_conf['env_num']
        if mode == 'test':
            env_num = self.method_conf['test_num']
        for env_id in range(env_num):
            episode_metrics_result_list.append(
                np.load(os.path.join(self.log_path, 'process_' + str(env_id), 'episode_metrics_result.npy'),
                        allow_pickle=True)[()])
        for key in episode_metrics_result_list[0]:
            self.envs_info[key] = [episode_metrics_result_list[env_id][key] for env_id in range(env_num)]
        for key in self.envs_info:
            self.envs_info[key] = np.concatenate(
                [np.expand_dims(np.array(info_list), axis=1) for info_list in self.envs_info[key]], axis=1)

    def load_sub_rollout_dict(self, env_id):
        sub_rollout_dict_path = os.path.join(self.log_path, 'process_' + str(env_id), 'sub_rollout_dict.npy')
        sub_rollout_dict = np.load(sub_rollout_dict_path, allow_pickle=True)[()]
        return sub_rollout_dict

    def delete_sub_rollout_dict(self, env_id):
        sub_rollout_dict_path = os.path.join(self.log_path, 'process_' + str(env_id), 'sub_rollout_dict.npy')
        if os.path.exists(sub_rollout_dict_path):
            os.remove(sub_rollout_dict_path)
