import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from L2OBench.Agent.basic_Agent import learnable_Agent
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

# 实现一个简单的metalearning的网络结构  用于改进pbo算法

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

class meta_Agent(learnable_Agent):
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
