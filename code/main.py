from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help='the name of dataset (KAIST or UCLA)')
    parser.add_argument('method_name', type=str, help='the name of method (GARL)')
    parser.add_argument('mode', type=str, help='train or test')
    args = parser.parse_args()

    dataset_conf = importlib.import_module('datasets.' + args.dataset_name + '.conf_temp').DATASET_CONF
    env_conf = importlib.import_module('env.conf_temp').ENV_CONF
    method_conf = importlib.import_module('methods.' + args.method_name + '.conf_temp').METHOD_CONF
    log_conf = importlib.import_module('log.conf_temp').LOG_CONF

    log_conf['dataset_name'] = args.dataset_name
    log_conf['method_name'] = args.method_name

    global_dict_init()
    set_global_dict_value('dataset_conf', dataset_conf)
    set_global_dict_value('env_conf', env_conf)
    set_global_dict_value('method_conf', method_conf)
    set_global_dict_value('log_conf', log_conf)

    fix_random_seed(get_global_dict_value('method_conf')['seed'])
    if args.mode == 'train':
        main = importlib.import_module('methods.' + args.method_name + '.main')
        main.main()
    elif args.mode == 'test':
        main_test = importlib.import_module('methods.' + args.method_name + '.main_test')
        main_test.main()
