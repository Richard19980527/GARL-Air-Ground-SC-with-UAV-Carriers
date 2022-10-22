from log.mainlog import *
from .subp_test import *


def main():
    dataset_conf = get_global_dict_value('dataset_conf')
    env_conf = get_global_dict_value('env_conf')
    method_conf = get_global_dict_value('method_conf')
    log_conf = get_global_dict_value('log_conf')

    mp.set_start_method("spawn", force=True)
    mainlog = MainLog(mode='test')

    shared_ifdone_list = [mp.Value('b', False) for _ in range(method_conf['test_num'])]

    simulator_processes = []
    for env_id in range(method_conf['test_num']):
        p = mp.Process(target=subp_test,
                       args=(env_id,
                             mainlog.log_root_path,
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

    for iter_id in range(1):
        ################################## gen samples ####################################
        while True:
            global_ifdone = 0
            for shared_ifdone in shared_ifdone_list:
                if shared_ifdone.value:
                    global_ifdone += 1
                else:
                    break
            if global_ifdone == method_conf['test_num']:
                ################################## mainlog work ####################################
                mainlog.load_envs_info(mode='test')
                eff = np.mean(mainlog.envs_info['eff'][iter_id])
                fairness = np.mean(mainlog.envs_info['fairness'][iter_id])
                dcr = np.mean(mainlog.envs_info['dcr'][iter_id])
                ecr = np.mean(mainlog.envs_info['ecr'][iter_id])
                cor = np.mean(mainlog.envs_info['cor'][iter_id])

                report_str = '#' * 100 \
                             + '\n' \
                             + 'iter: ' + str(iter_id) \
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
