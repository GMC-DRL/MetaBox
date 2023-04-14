import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        # self.net should be a tuple/list/dict of several nets
        self.net = None

        self.memory = Memory()

        pass

    def get_feature(self,env):
        # get feature from env.state to feed net
        pass


    def inference(self,env,need_gd):
        # get_feature
        # use feature to get aciton
        pass


    def cal_loss(self,env):

        pass


    def learning(self):
        # select optimizer(Adam,SGD...)
        # cal_loss
        # update nets

        pass


    def memory(self):
        # record some info
        return self.memory
        pass
