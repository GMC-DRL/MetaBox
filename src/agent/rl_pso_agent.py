import torch
from torch import nn
from torch.distributions import Normal
from agent.basic_agent import Basic_Agent
from agent.networks import MLP
from .utils import *


class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()

        net_config = [{'in': config.feature_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 32, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 8, 'out': config.action_dim, 'drop_out': 0, 'activation': 'None'}]

        self.__mu_net = MLP(net_config)
        self.__sigma_net = MLP(net_config)

        self.__max_sigma = config.max_sigma
        self.__min_sigma = config.min_sigma

    def forward(self, x_in, require_entropy=False, require_musigma=False):
        mu = self.__mu_net(x_in)
        mu = (torch.tanh(mu) + 1.) / 2.
        sigma = self.__sigma_net(x_in)
        sigma = (torch.tanh(sigma) + 1.) / 2.
        sigma = torch.clamp(sigma, min=self.__min_sigma, max=self.__max_sigma)

        policy = Normal(mu, sigma)
        action = policy.sample()

        filter = torch.abs(action - 0.5) >= 0.5
        action = torch.where(filter, (action + 3 * sigma.detach() - mu.detach()) * (1. / 6 * sigma.detach()), action)
        log_prob = policy.log_prob(action)

        if require_entropy:
            entropy = policy.entropy()

            out = (action, log_prob, entropy)
        else:
            if require_musigma:
                out = (action, log_prob, mu, sigma)
            else:
                out = (action, log_prob)

        return out


class RL_PSO_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        # add specified config
        config.feature_dim = 2*config.dim
        config.action_dim = 1
        config.action_shape = (1,)
        config.max_sigma = 0.7
        config.min_sigma = 0.01
        config.lr = 1e-5
        self.__config = config

        self.__device = config.device
        self.__nets = PolicyNetwork(config).to(self.__device)

        # optimizer
        self.__optimizer = torch.optim.Adam([{'params': self.__nets.parameters(), 'lr': config.lr}])
        self.__learning_time = 0
        
        self.__cur_checkpoint=0

        # save init agent
        if self.__cur_checkpoint==0:
            save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
            self.__cur_checkpoint+=1

    def train_episode(self, env):

        # input action_dim should be : bs, ps
        # action in (0,1) the ratio to learn from pbest & gbest
        state = env.reset()
        state = torch.FloatTensor(state).to(self.__device)
        
        exceed_max_ls = False
        R = 0
        while True:
            action, log_prob = self.__nets(state)
            action = action.reshape(self.__config.action_shape)
            action = action.cpu().numpy()
            
            state, reward, is_done = env.step(action)
            R += reward
            state = torch.FloatTensor(state).to(self.__device)
            
            policy_gradient = -log_prob*reward
            loss = policy_gradient.mean()

            self.__optimizer.zero_grad()
            loss.mean().backward()
            self.__optimizer.step()
            self.__learning_time += 1
            if self.__learning_time >= (self.__config.save_interval * self.__cur_checkpoint):
                save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
                self.__cur_checkpoint+=1

            if self.__learning_time >= self.__config.max_learning_step:
                exceed_max_ls = True
                break

            if is_done:
                break
        return exceed_max_ls, {'normalizer': env.optimizer.cost[0],
                               'gbest': env.optimizer.cost[-1],
                               'return': R,
                               'learn_steps': self.__learning_time}
    
    def rollout_episode(self, env):
        is_done = False
        state = env.reset()
        R=0
        while not is_done:
            state = torch.FloatTensor(state)
            action, _ = self.__nets(state)
            state, reward, is_done = env.step(action.cpu().numpy())
            R+=reward
        return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes,'return':R}
