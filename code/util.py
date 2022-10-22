import os

os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import copy
import importlib
import math
import numpy as np
import random
from scipy.linalg import fractional_matrix_power
import shutil
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import List

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


def global_dict_init():
    global _global_dict
    _global_dict = {}


def set_global_dict_value(key, value):
    _global_dict[key] = value


def get_global_dict_value(key):
    return _global_dict[key]


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


