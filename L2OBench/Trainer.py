"""
This file is used to train the agent.(mainly for the kind of optimizer that is learnable)
The class Experimentmanager should have the following functions:
    1. __init__(self,problem, optimizer, agent, reward_function) to initialize the Experimentmanager
    2. run(self) to run the Experimentmanager and train the agent
"""
from L2OBench.Optimizer import learnable_optimizer
from L2OBench.Environment import basic_environment

from L2OBench.reward import binary
from L2OBench.Problem.cec_dataset import Training_Dataset
import math
import torch
import copy
import numpy as np

def clip_grad_norms(param_groups, max_norm=math.inf):
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
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


# class Experimentmanager():
#     def __init__(self, agent, env, config=None):
#         self.env = env
#         self.agent = agent
#         self.config = config
#
#     def run(self):
#         is_done = False
#         # 此处的state仅是部分env的meta info ，并不是env的state，env的state是在agent的get_feature中得到的
#         state = self.env.reset()
#         # state = torch.FloatTensor(state).to(self.config.device)
#
#
#
#         n_step = self.config.n_step
#         t = 0
#         done = False
#
#         while not done:
#             t_s = t
#             total_cost = 0
#             entropy = []
#             bl_val_detached = []
#             bl_val = []
#             temp_states = []
#             self.agent.memory.clear_memory()
#
#             while t-t_s < n_step:
#                 state_feature = self.agent.get_feature(self.env)
#                 state_feature = torch.FloatTensor(state_feature).to(self.config.device)
#                 temp_states.append(state_feature)
#                 self.agent.memory.states.append(state)
#                 action, log_lh,_to_critic,  entro_p = self.agent.inference(self.env,
#                                                                            require_entropy=True,
#                                                                            to_critic=True,
#                                                                            need_gd=True)
#                 self.agent.memory.actions.append(action.clone())
#
#                 self.agent.memory.logprobs.append(log_lh.clone())
#                 # print('old_logprobs.shape:{}'.format(torch.stack(self.agent.memory.logprobs).detach().view(-1).shape))
#                 action = action.cpu().numpy()
#
#
#                 entropy.append(entro_p.detach().cpu())
#
#                 baseline_val_detached, baseline_val = self.agent.nets[1](_to_critic)
#                 bl_val_detached.append(baseline_val_detached)
#                 bl_val.append(baseline_val)
#
#                 state, reward, is_done = self.env.step(action[0])
#                 print("fes:", state['fes'], "cost_min:", state['cost'].min(), "cost_mean:", state['cost'].mean(),
#                       "reward:",
#                       reward)
#
#                 self.agent.memory.rewards.append(reward)
#
#                 t += 1
#                 if is_done:
#                     done = True
#                     break
#
#             t_time = t - t_s
#             total_cost = total_cost / t_time
#
#             # begin update
#
#             old_actions = torch.stack(self.agent.memory.actions)
#             # old_states = torch.FloatTensor(self.agent.memory.states).detach()  # .view(t_time, bs, ps, dim_f)
#             # 使用得到了feature的state数组
#             old_states = torch.stack(temp_states).detach()  # .view(t_time, bs, ps, dim_f)
#             # old_states = torch.stack(self.agent.memory.states).detach()  # .view(t_time, bs, ps, dim_f)
#
#
#             # old_actions = all_actions.view(t_time, bs, ps, -1)
#             # print('old_actions.shape:{}'.format(old_actions.shape))
#
#             old_logprobs = torch.stack(self.agent.memory.logprobs).detach().view(-1)
#             # print('old_logprobs.shape:{}'.format(old_logprobs.shape))
#
#             # Optimize PPO policy for K mini-epochs:
#             old_value = None
#
#             for _k in range(self.config.K_epochs):
#
#                 if _k == 0:
#                     logprobs = self.agent.memory.logprobs
#
#                 else:
#                     # Evaluating old actions and values :
#                     logprobs = []
#                     entropy = []
#                     bl_val_detached = []
#                     bl_val = []
#
#                     for tt in range(t_time):
#                         # get new action_prob
#                         _, log_p, _to_critic, entro_p = self.agent.nets[0](torch.Tensor(np.expand_dims(old_states[tt], axis=0)).to(self.config.device),
#                                                                     fixed_action=old_actions[tt],
#                                                                     require_entropy=True,  # take same action
#                                                                     to_critic=True,
#                                                                     )
#
#                         # _, log_p, _to_critic, entro_p = self.agent.nets[0](old_states[tt],
#                         #                                             fixed_action=old_actions[tt],
#                         #                                             require_entropy=True,  # take same action
#                         #                                             to_critic=True
#                         #                                             )
#
#                         logprobs.append(log_p)
#                         entropy.append(entro_p.detach().cpu())
#
#                         baseline_val_detached, baseline_val = self.agent.nets[1](_to_critic)
#
#                         bl_val_detached.append(baseline_val_detached)
#                         bl_val.append(baseline_val)
#
#                 logprobs = torch.stack(logprobs).view(-1)
#                 # print('logprobs.shape:{}'.format(logprobs.shape))
#                 entropy = torch.stack(entropy).view(-1)
#                 bl_val_detached = torch.stack(bl_val_detached).view(-1)
#                 bl_val = torch.stack(bl_val).view(-1)
#
#                 # get traget value for critic
#                 Reward = []
#                 reward_reversed = self.agent.memory.rewards[::-1]
#                 # get next value
#                 cretic = self.agent.inference(self.env, only_critic=True, need_gd=True)
#                 R = self.agent.nets[1](cretic)[0]
#
#                 # R = agent.critic(x_in)[0]
#                 critic_output = R.clone()
#                 for r in range(len(reward_reversed)):
#                     R = R * self.config.gamma + reward_reversed[r]
#                     Reward.append(R)
#                 # clip the target:
#                 Reward = torch.stack(Reward[::-1], 0)
#                 Reward = Reward.view(-1)
#                 # print('Reward.shape:{}'.format(Reward.shape))
#
#                 # Finding the ratio (pi_theta / pi_theta__old):
#                 # print('logprobs.shape:{}'.format(logprobs.shape))
#                 # print('old_logprobs.shape:{}'.format(old_logprobs.shape))
#                 ratios = torch.exp(logprobs - old_logprobs.detach())
#                 # print('ratios.shape:{}'.format(ratios.shape))
#
#                 # Finding Surrogate Loss:
#                 # print('bl_val_detached.shape:{}'.format(bl_val_detached.shape))
#                 # print('Reward.shape:{}'.format(Reward.shape))
#                 advantages = Reward - bl_val_detached
#                 # print('advantages.shape:{}'.format(advantages.shape))
#
#                 surr1 = ratios * advantages
#                 surr2 = torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages
#                 reinforce_loss = -torch.min(surr1, surr2).mean()
#                 # print(reinforce_loss.shape)
#
#                 # define baseline loss
#                 if old_value is None:
#                     baseline_loss = ((bl_val - Reward) ** 2).mean()
#                     old_value = bl_val.detach()
#                 else:
#                     vpredclipped = old_value + torch.clamp(bl_val - old_value, - self.config.eps_clip, self.config.eps_clip)
#                     v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
#                     baseline_loss = v_max.mean()
#
#                 # check K-L divergence (for logging only)
#                 approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
#                 approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
#                 # calculate loss
#                 loss = baseline_loss + reinforce_loss
#                 # print(type(loss))
#                 # print(loss.shape)
#
#                 self.agent.optimizer.zero_grad()
#                 loss.backward()
#
#                 # Clip gradient norm and get (clipped) gradient norms for logging
#
#                 grad_norms = clip_grad_norms(self.agent.optimizer.param_groups, self.config.max_grad_norm)
#
#                 # perform gradient descent
#                 self.agent.optimizer.step()
#                 # self.agent.memory.clear_memory()
#                 print('loss:{}'.format(loss))
#
#                 # end update
#
#             # self.agent.learning(env=self.env)
#
#         # learn

# class Experimentmanager():
#     def __init__(self, agent, env, config=None):
#         self.env = env
#         self.agent = agent
#         self.config = config
#
#     def run(self):
#         is_done = False
#         # 此处的state仅是部分env的meta info ，并不是env的state，env的state是在agent的get_feature中得到的
#         state = self.env.reset()
#         # state = torch.FloatTensor(state).to(self.config.device)
#
#
#
#         n_step = self.config.n_step
#         t = 0
#         done = False
#
#         while not done:
#             t_s = t
#             total_cost = 0
#
#
#             self.agent.memory.clear_memory()
#             loss = 0
#             fitness = 0
#             while t-t_s < n_step:
#
#                 delta_x = self.agent.inference(self.env,need_gd=True)
#                 delta_x = torch.squeeze(delta_x, 0)
#                 delta_x = torch.squeeze(delta_x, 0)
#                 delta_x = delta_x.detach().cpu().numpy()
#                 # print("delta_x:",delta_x.shape)
#
#                 state, reward, is_done = self.env.step(delta_x)
#
#                 fitness = state['cost']
#
#                 # print('fitness:{}'.format(fitness))
#                 fitness = torch.FloatTensor(fitness).to(self.config.device)
#                 loss += fitness
#                 # print('loss:{}'.format(loss))
#
#                 t += 1
#                 if is_done:
#                     done = True
#                     break
#
#             t_time = t - t_s
#             total_cost = total_cost / t_time
#
#             # begin update
#             self.agent.optimizer.zero_grad()
#             loss.backward()
#
#             self.agent.optimizer.step()
#             # self.agent.memory.clear_memory()
#             print('loss:{}'.format(loss))
#
#             # end update


class Trainer_traditional():
    # 这里的env为env
    def __init__(self,config,env,optimizer):

        pass

    def run(self):
        pass


class Trainer_learnable():
    # 这里的env为pbo_env
    def __init__(self,config,env,agent):
        pass

    def run(self):
        pass


class ExperimentManager():
    def __init__(self,config,env,agent = None,optimizer = None):
        self.need_agent = config.need_agent
        if self.need_agent == True:
            self.Trainer = Trainer_learnable(config,env,agent)
        else:
            self.Trainer = Trainer_traditional(config,env,optimizer)
        pass

    def run(self):
        self.Trainer.run()
        pass


