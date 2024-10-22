import torch
import numpy as np
import collections
import copy
import random
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution
import os
from torch import nn
from agent.basic_agent import Basic_Agent
from agent.utils import *


def clip_grad_norms(param_groups, max_norm=np.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else np.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class Actor(nn.Module):
    def __init__(self, dim, optimizer_num, feature_dim, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList([])
        for i in range(optimizer_num):
            self.embedders.append((nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ])).to(device))
            self.embedders.append(nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ]).to(device))
        self.embedder_final = nn.Sequential(*[
            nn.Linear(feature_dim + optimizer_num * 2, 64), nn.Tanh(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(64, 16), nn.Tanh(),
            nn.Linear(16, optimizer_num),  nn.Softmax(),
        ]).to(device)

    def forward(self, obs, test=False):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(self.embedders[i](torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)))
        moves = torch.cat(moves, dim=-1)
        batch = obs.shape[0]
        feature = torch.cat((feature, moves), dim=-1).view(batch, -1)
        feature = self.embedder_final(feature)
        logits = self.model(feature)
        if test:
            out = (feature.detach().cpu().tolist(), logits)
        else:
            out = logits
        return out


class PPO_critic(nn.Module):
    def __init__(self, dim, optimizer_num, feature_dim, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList([])
        for i in range(optimizer_num):
            self.embedders.append((nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ])).to(device))
            self.embedders.append(nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ]).to(device))
        self.embedder_final = nn.Sequential(*[
            nn.Linear(feature_dim + optimizer_num * 2, 64), nn.Tanh(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(64, 16), nn.Tanh(),
            nn.Linear(16, 1), # nn.Softmax(),
        ]).to(device)

    def forward(self, obs):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(self.embedders[i](torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)))
        moves = torch.cat(moves, dim=-1)
        batch = obs.shape[0]
        feature = torch.cat((feature, moves), dim=-1).view(batch, -1)
        feature = self.embedder_final(feature)
        batch = obs.shape[0]
        bl_val = self.model(feature.view(batch, -1))
        return bl_val


class RL_DAS_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        
        config.feature_dim = 9
        self.device = config.device
        self.actor = Actor(config.dim, 3, config.feature_dim, self.device)
        self.actor_softmax = torch.nn.Softmax().to(self.device)
        self.critic = PPO_critic(config.dim, 3, config.feature_dim, self.device)
        # self.init_parameters()
        self.optimizer = torch.optim.Adam(
            [{'params': self.actor.parameters(), 'lr': 1e-5}] +
            [{'params': self.critic.parameters(), 'lr': 1e-5}])
        self.gamma = 0.99
        self.eps_clip = 0.1
        self.max_grad_norm = 0.1
        self.max_learning_step = config.max_learning_step

        # init learning time
        self.__learning_time=0

        self.__cur_checkpoint=0

        self.__config = config
        # save init agent
        if self.__cur_checkpoint==0:
            save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
            self.__cur_checkpoint+=1

    def update_setting(self, config):
        self.max_learning_step = config.max_learning_step
        self.__config.agent_save_dir = config.agent_save_dir
        self.__learning_time = 0
        save_class(self.__config.agent_save_dir, 'checkpoint0', self)
        self.__config.save_interval = config.save_interval
        self.__cur_checkpoint = 1

    def init_parameters(self):
        for param in self.actor.parameters():
            stdv = 1. / np.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        for param in self.critic.parameters():
            stdv = 1. / np.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def save(self, log_path, epoch, run_time):
        torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                },
                os.path.join(log_path, 'PPO-' + run_time + f'-{epoch}.pth'))

    def load(self, path):
        models = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(models['actor'])
        self.critic.load_state_dict(models['critic'])

    def actor_forward(self, x, test=False, to_critic=False):
        return self.actor(x, test)

    def actor_forward_without_grad(self, x, test=False):
        with torch.no_grad():
            return self.actor_forward(x, test)

    def actor_sample(self, probs, fix_action=None):
        # probs = self.actor_softmax(probs)
        policy = Categorical(probs)
        if fix_action is None:
            actions = policy.sample()
        else:
            actions = fix_action
        select_probs = policy.log_prob(actions)
        return probs, select_probs, actions

    def critic_forward(self, x):
        bl_val = self.critic(x)
        baseline_val_detached = bl_val.detach()
        return baseline_val_detached, bl_val

    def critic_forward_without_grad(self, x):
        with torch.no_grad():
            return self.critic_forward(x)

    def learn(self, memory, k_epoch):
        length = len(memory['rewards'])
        old_states = memory['states']  # episode length * batch_size * state dim
        old_logprobs = []
        for tt in range(length):
            old_logprobs.append(memory['logprobs'][tt])
        old_logprobs = torch.cat(old_logprobs).view(-1)
        actions = memory['actions']
        old_value = None
        grads = []
        for k in range(k_epoch):
            if k == 0:
                logprobs = []
                bl_val_detached = []
                bl_val = []
                for tt in range(length):
                    logprobs.append(memory['logprobs'][tt])
                    bl_val_detached.append(memory['bl_val_detached'][tt])
                    bl_val.append(memory['bl_val'][tt])
            else:
                logprobs = []
                bl_val_detached = []
                bl_val = []
                for tt in range(length):
                    logits = self.actor_forward(old_states[tt])
                    _, batch_log_likelyhood, batch_action = self.actor_sample(logits, actions[tt])
                    logprobs.append(batch_log_likelyhood)
                    baseline_val_detached, baseline_val = self.critic_forward(old_states[tt])
                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)

            logprobs = torch.cat(logprobs).view(-1)
            bl_val_detached = torch.cat(bl_val_detached).view(-1)
            bl_val = torch.cat(bl_val).view(-1)
            Reward = []
            reward_reversed = memory['rewards'][::-1]

            R = self.critic_forward(old_states[-1])[0].view(-1)
            for r in range(len(reward_reversed)):
                R = R * self.gamma + torch.tensor(reward_reversed[r], dtype=torch.float32).to(self.device)
                Reward.append(R)
            # Reward = torch.stack(Reward[::-1], 0)  # n_step, bs
            # Reward = Reward.view(-1)
            Reward = torch.cat(Reward[::-1]).view(-1)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = Reward - bl_val_detached
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            reinforce_loss = -torch.min(surr1, surr2).mean()
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - self.eps_clip, self.eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()

            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0

            loss = baseline_loss + reinforce_loss  # - 1e-5 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            grad = []
            for idx, group in enumerate(self.optimizer.param_groups):
                grad.append(group['params'][0].detach().view(-1))
            grad = torch.cat(grad)
            grads.append(grad)
            grad_norms = 0
            if self.max_grad_norm > 0:
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.max_grad_norm)
            self.optimizer.step()
            self.__learning_time += 1

            if self.__learning_time >= (self.__config.save_interval * self.__cur_checkpoint):
                    save_class(self.__config.agent_save_dir, 'checkpoint'+str(self.__cur_checkpoint), self)
                    self.__cur_checkpoint += 1

    def train_episode(self, env):
        # print(self.max_learning_step)
        obs = np.array([env.reset(),], dtype=object)
        is_done = False
        memory = {'states': [],
                'logprobs': [],
                'actions': [],
                'rewards': [],
                'bl_val': [],
                'bl_val_detached': [],
                }
        memory['states'].append(obs)
        R = 0
        while not is_done:
            probs, log_probs, act = self.actor_sample(self.actor_forward(obs))
            actions = act[0].cpu()
            baseline_val_detached, baseline_val = self.critic_forward(obs)
            obs_next, rewards, is_done, info = env.step(actions)
            obs = np.array([obs_next,], dtype=object)
            memory['states'].append(obs)
            memory['logprobs'].append(log_probs)
            memory['actions'].append(act)
            memory['rewards'].append(np.array(rewards, dtype=np.float32))
            memory['bl_val'].append(baseline_val)
            memory['bl_val_detached'].append(baseline_val_detached)

            if is_done:
                memory['rewards'] = np.array(memory['rewards'])
                award = env.optimizer.MaxFEs / env.optimizer.FEs
                memory['rewards'] *= award
                R = np.sum(memory['rewards'])

        k_epoch = int(0.3*(env.optimizer.MaxFEs // env.optimizer.period))
        self.learn(memory, k_epoch)

        return self.__learning_time >= self.max_learning_step, {'normalizer': env.optimizer.cost_scale_factor,
                                                        'gbest': env.optimizer.population.gbest,
                                                        'return': R,
                                                        'learn_steps': self.__learning_time}
    
    def rollout_episode(self, env):
        state = env.reset()
        done = False
        R = 0
        while not done:
            state = np.array([state,], dtype=object)
            # log_obs(logger, obs, total_steps)
            probs, log_probs, act = self.actor_sample(self.actor_forward(state))
            actions = act.cpu()
            obs_next, rewards, done, info = env.step(actions)
            state = obs_next
            R += rewards

            if done:
                award = env.optimizer.MaxFEs / env.optimizer.FEs
                R *= award

        return {'cost': env.optimizer.cost, 'fes': env.optimizer.FEs, 'return': R}