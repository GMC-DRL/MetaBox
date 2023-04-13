from Agent import basic_Agent, metalearning,gleet
from Optimizer import basic_optimizer,learnable_optimizer
from Environment import basic_environment
from Problem import basic_problem,cec_test_func

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from reward import binary
from Problem.cec_dataset import Training_Dataset
from config import get_config
class Experimentmanager():
    def __init__(self,problem, optimizer, agent, reward_function):
        self.env = basic_environment.PBO_Env(problem,optimizer,reward_function)
        self.agent = agent

        pass

    def run(self):
        # 1.已通过参数初始化agent和env
        # 2.agent得到action
        # 3.env根据action得到reward、更新种群
        # 4.重复2、3直到满足终止条件
        pass