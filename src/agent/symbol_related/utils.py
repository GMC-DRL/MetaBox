import torch
import torch.nn as nn

from .expression import *
from torch.distributions import Categorical
import numpy as np
import math

binary_code_len=4

class Memory():
    def __init__(self) -> None:
        self.position=[]
        self.x_in=[]
        self.mask=[]
        self.working_index=[]
        self.seq=None
        self.c_seq=None
        self.c_index=[]
        self.filter_index=[]
    
    def get_dict(self):
        return {
            'position':self.position,
            'x_in':self.x_in,
            'mask':self.mask,
            'working_index':self.working_index,
            'seq':self.seq,
            'c_seq':self.c_seq,
            'c_index':self.c_index,
            'filter_index':self.filter_index
        }

def get_binary(action):
    bs=action.shape[0]
    binary_code=torch.zeros((bs,binary_code_len))
    for i in range(bs):
        binary=bin(int(action[i]+1))[2:]
        l=list(map(int,str(binary)))
        while len(l)<binary_code_len:
            l.insert(0,0)
        binary_code[i]=torch.tensor(l)
    return binary_code

def get_choice(output,mask,fix_choice=None):
    # output:(bs,1,output_size)
    bs,_,output_size=output.size()
    output=output.squeeze(1)
    

    # apply mask
    output[mask==0]=-math.inf
    # print(f'pre:{prob}')
    prob=torch.softmax(output,dim=-1)

    policy=Categorical(prob)
    if fix_choice is not None:
        action=fix_choice
        log_prob=policy.log_prob(action)
        return log_prob
    else:
        action=policy.sample()
    log_prob=policy.log_prob(action)

    # get binary code
    binary_code=get_binary(action)
    return log_prob,action,binary_code

def get_c(output,min_c,interval,fix_c=None):
    # output:(bs,1,output_size)
    output=output.squeeze(1)
    device = output.device
    
    bs,output_size=output.size()
    # print(f'output.size:{output.shape}')
    prob=torch.softmax(output,dim=-1)

    policy=Categorical(prob)
    if fix_c is not None:
        choice=(fix_c-min_c)//interval
        log_prob=policy.log_prob(choice.to(device))
        return log_prob
    else:
        choice=policy.sample()
    log_prob=policy.log_prob(choice)
    
    choice=min_c+choice*interval

    return log_prob,choice



class MLP(nn.Module):
    def __init__(self, config):
        """
        :param config: a list of dicts like
                 [{'in':2,'out':4,'drop_out':0.5,'activation':'ReLU'},
                  {'in':4,'out':8,'drop_out':0,'activation':'Sigmoid'},
                  {'in':8,'out':10,'drop_out':0,'activation':'None'}],
                and the number of dicts is customized.
        """
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        self.net_config = config
        for layer_id, layer_config in enumerate(self.net_config):
            linear = nn.Linear(layer_config['in'], layer_config['out'])
            self.net.add_module(f'layer{layer_id}-linear', linear)
            drop_out = nn.Dropout(layer_config['drop_out'])
            self.net.add_module(f'layer{layer_id}-drop_out', drop_out)
            if layer_config['activation'] != 'None':
                activation = eval('nn.'+layer_config['activation'])()
                self.net.add_module(f'layer{layer_id}-activation', activation)

    def forward(self, x):
        return self.net(x)