import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch

class learnable_Agent():
    def __init__(self,config):
        self.config = config
        self.net = None


        pass

    def get_feature(self):

        pass


    def inference(self,need_gd):
        # get aciton/fitness
        pass


    def cal_loss(self):
        pass


    def learning(self):
        # cal_loss
        # update nets
        pass


    def memory(self):
        # record some info
        pass
