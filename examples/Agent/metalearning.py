import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import math

from L2OBench.Agent.basic_Agent import learnable_Agent
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.hidden = []
        self.momentum = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.hidden[:]

def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch

# 实现一个简单的metalearning的网络结构  用于改进pbo算法

class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.population_size = config.population_size
        self.hidden_dim = config.hidden_dim
        print(self.hidden_dim)
        # self.intra_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        #
        # self.inter_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        self.memory = Memory()
        # 每个lstm的hidden都是由两个（1，1，dim）组成的tuple
        for k in range(self.population_size):
            self.memory.hidden.append((torch.rand(1,1,self.hidden_dim),torch.rand(1,1,self.hidden_dim)))

        # print("self.memory.hidden:",self.memory.hidden.shape)
        self.lstm = []
        for k in range(self.population_size):
            self.lstm.append(nn.LSTM(input_size=config.dim,hidden_size=config.hidden_dim, num_layers=1))

    def get_parameter_number(self):

        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    # state shape:list population_size [4,embedding_dim]
    def forward(self, state):
        # intra attention
        # intra_output, _ = self.intra_attention(state, state, state) #(q k v) = (4*n 1*n 1*n)
        # # inter attention
        # inter_output, _ = self.inter_attention(state, state, state)

        attention_states = [] #k*n
        for k in range(len(state)):
            attention_state = torch.sum(state[k],dim=0) #1*n
            # print("attention_state:",attention_state.shape)
            attention_states.append(attention_state)
            # print("attention_states:",attention_states[k].shape)

        # attention_states = torch.FloatTensor(attention_states).to(self.config.device)
        # lstm

        lstm_output = []
        hiddens = []
        for(k,_lstm) in enumerate(self.lstm):
            # print("attention_states[k]:",attention_states[k].shape)
            out, hidden = _lstm(attention_states[k].view(1, 1, -1), self.memory.hidden[k])

            # out,hidden = lstm(attention_states[k],self.memory.hidden[k])
            hiddens.append(hidden)
            lstm_output.append(out)
            # print("lstm_output:",lstm_output[k].shape)
        # output
        output = torch.cat(lstm_output, dim=0)
        self.memory.hidden = hiddens
        return output

class meta_Agent(learnable_Agent):
    def __init__(self,config):
        self.config = config
        self.nets = [Actor(config)]
        self.memory = Memory()
        self.dim = config.dim
        self.beta = config.beta
        self.alpha = config.alpha
        # self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)
        for i in range(len(self.config.population_size)):
            self.memory.momentum.append(torch.rand(self.dim))

        self.optimizer = torch.optim.Adam(
            [{'params': self.nets[0].parameters(), 'lr': config.lr_model}]
        )


    def get_feature(self,env):
        def distance(x, y):
            r = np.linalg.norm(x - y)
            return r
        # get feature from env
        # env.optimizer 中包含了pop_size个particle，每个particle的feature应该为4*n，n为problem的dim  4*n
        # 输出的feature是一个list，每个元素是一个particle的feature  k*4*n
        fitness = []
        for(i,particle) in enumerate(env.optimizer.population):
            fitness.append(env.problem.eval(particle))
        features = []
        gradients = []
        for i in range(self.config.population_size):
            # gradient = torch.rand(1, 1,self.dim)
            # momentum = torch.rand(1, 1,self.dim)
            # velocity = torch.rand(1, 1,self.dim)
            attraction = torch.zeros(1, 1,self.dim)
            # 计算近似梯度
            for j in range(self.config.dim):
                delta_x = torch.zeros(self.dim)
                delta_x[j] = self.config.h
                x_minus = env.optimizer.population[i] - delta_x
                x_plus = env.optimizer.population[i] + delta_x
                gradient = (env.problem.func(x_plus) - env.problem.func(x_minus)) / ( 2 * delta_x )
                gradients.append(gradient)

            gradient = torch.cat(gradients, dim=0)

            self.memory.momentum[i] = self.beta * self.memory.momentum[i] + (1 - self.beta) * gradient

            velocity = self.optimizer.population[i] - env.optimizer.gbest

            for j in range(self.config.population_size):
                a,b = 0,0
                if fitness[j] < fitness[i]:
                    a += math.exp(-self.alpha * distance(env.optimizer.population[i], env.optimizer.population[j]))*(
                                env.optimizer.population[i] - env.optimizer.population[j])
                    b += math.exp(-self.alpha * distance(env.optimizer.population[i], env.optimizer.population[j]))
                attraction[0][0] = a / b
            # 这个feature的维度为4*n
            # feature = np.concatenate((gradient, momentum, velocity, attraction), axis=0)
            feature = torch.cat((gradient, self.memory.momentum[i], velocity, attraction), dim=0)
            # print("feature:",feature.shape)
            features.append(feature)
            # print(feature.shape)

        return features



    def inference(self,env,need_gd):
        # get aciton/fitness
        # state_feature.shape (pop_size,4,n)
        state_feature = self.get_feature(env)
        # state_feature = torch.FloatTensor(state_feature).to(self.config.device)
        # action为更新种群的动作   new_pop = old_pop + action
        # so action.shape should be (pop_size,n)
        action = self.nets[0](state_feature)
        print("action:",action.shape)
        return action



    def cal_loss(self,env):
        # cal loss
        total_loss = 0


        pass


    def learning(self,env):
        # cal_loss
        loss = self.cal_loss(env)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update nets


    def memory(self):
        # record some info
        return self.memory


class reNN(nn.Module):
    def __init__(self, config):
        super(reNN, self).__init__()
        self.config = config
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.hidden = (torch.rand(1,1,self.hidden_dim),torch.rand(1,1,self.hidden_dim))
        self.lstm = nn.LSTM(input_size=config.dim,hidden_size=config.hidden_dim, num_layers=1)
        self.fc = nn.Linear(config.hidden_dim, config.dim)

    def forward(self, x):
        next_x,self.hidden = self.lstm(x,self.hidden)
        out = self.fc(next_x)
        return out

class l2l_agent():
    def __init__(self,config):
        self.config = config
        self.nets = [reNN(config)]
        self.memory = Memory()
        self.dim = config.dim
        # self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)

        self.optimizer = torch.optim.Adam(
            [{'params': self.nets[0].parameters(), 'lr': config.lr_model}]
        )

    def get_feature(self,env):
        x = env.optimizer.population[0]
        x = torch.FloatTensor(x).to(self.config.device)
        # print("x:",x.shape)
        return x


    def inference(self,env,need_gd):
        feature = self.get_feature(env)
        feature = torch.unsqueeze(feature,0)
        feature = torch.unsqueeze(feature,0)
        # print("feature:",feature.shape)
        delta_x = self.nets[0](feature)
        return delta_x


    def cal_loss(self,env):
        # # cal loss
        # total_loss = 0
        # feature = self.get_feature(env)
        # feature = torch.unsqueeze(feature, 0)
        # delta_x = self.nets[0](feature)
        # delta_x = torch.squeeze(delta_x,0)
        # new_x = env.optimizer.population + delta_x
        #
        # return loss
        pass


    def learning(self,env):
        # loss = self.cal_loss(env)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        pass

    def memory(self):
        return self.memory
