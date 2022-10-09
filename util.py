import os

os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from bs4 import BeautifulSoup as bs
from collections import deque
import copy
import datetime
from datetime import datetime
import glob
import imageio
import importlib
import joblib
import math
from matplotlib import animation
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.path as mplpath
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import os.path
import pandas as pd
import paramiko
from PIL import Image
import random
import re
from scipy.linalg import fractional_matrix_power
from scipy.stats import pearsonr
import shapefile  # 使用pyshp
import shutil
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import socket
import stat
from statistics import mean
import sys
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
import traceback
import utm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

project_name = os.path.abspath(__file__).split('/')[-2]
min_value = 1e-5


def dis_p2p(point1, point2):
    dis = math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2))
    return dis


def vector_dot_product(vector1, vector2):
    result = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    return result


def vector_length_counter(vector):
    length = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))
    return length


def vector1_project2_vector2(vector1, vector2):
    u = vector_dot_product(vector1, vector2) / (vector_length_counter(vector2)) ** 2
    target_vector = (vector2[0] * u, vector2[1] * u)
    return target_vector


def global_dict_init():  # 初始化
    global _global_dict
    _global_dict = {}


def set_global_dict_value(key, value):
    # 定义一个全局变量
    _global_dict[key] = value


def get_global_dict_value(key):
    # 获得一个全局变量，不存在则提示读取对应变量失败
    try:
        return _global_dict[key]
    except:
        print(datetime.now(), '读取' + key + '失败\r\n')


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def set_dict_value(mydict, keys, val):
    mydict_tmp = mydict
    lastkey = keys[-1]
    for key in keys[:-1]:
        mydict_tmp = mydict_tmp[key]
    if val == 'True':
        mydict_tmp[lastkey] = True
    elif val == 'False':
        mydict_tmp[lastkey] = False
    else:
        mydict_tmp[lastkey] = type(mydict_tmp[lastkey])(val)


def check_dict_key(mydict, keys):
    mydict_tmp = mydict
    flag = True
    for key in keys:
        if not isinstance(mydict_tmp, dict) or key not in mydict_tmp:
            flag = False
            break
        else:
            mydict_tmp = mydict_tmp[key]
    return flag


def gen_conf(args, conf_temp):
    conf = copy.deepcopy(conf_temp)
    for attr in dir(args):
        if attr == 'param_name':
            param_list = getattr(args, attr).split('___')
            for param in param_list:
                if param == 'param_name':
                    if check_dict_key(conf, [param]):
                        set_dict_value(conf, [param], getattr(args, attr))
                    continue
                keys = param.split('__')[:-1]
                val = param.split('__')[-1]
                if check_dict_key(conf, keys):
                    set_dict_value(conf, keys, val)
        else:
            keys = attr.split('__')
            if check_dict_key(conf, keys):
                set_dict_value(conf, keys, getattr(args, attr))
    return conf

