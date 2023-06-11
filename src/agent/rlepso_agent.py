import torch
from torch import nn
from torch.distributions import Normal
from agent.basic_agent import Basic_Agent
from agent.networks import MLP
from agent.utils import *


class Actor(nn.Module):
    def __init__(self,
                 config,
                 ):
        super(Actor, self).__init__()
        net_config = [{'in': config.feature_dim, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 32, 'out': config.action_dim, 'drop_out': 0, 'activation': 'None'}]
        self.__mu_net = MLP(net_config)
        self.__sigma_net = MLP(net_config)
        self.__max_sigma = config.max_sigma
        self.__min_sigma = config.min_sigma

    def forward(self, x_in, fixed_action=None, require_entropy=False):  # x-in: bs*gs*9

        mu = (torch.tanh(self.__mu_net(x_in)) + 1.) / 2.
        sigma = (torch.tanh(self.__sigma_net(x_in)) + 1.) / 2. * (self.__max_sigma - self.__min_sigma) + self.__min_sigma

        policy = Normal(mu, sigma)

        if fixed_action is not None:
            action = fixed_action
        else:
            action = torch.clamp(policy.sample(), min=0, max=1)
        log_prob = policy.log_prob(action)

        log_prob = torch.sum(log_prob)

        if require_entropy:
            entropy = policy.entropy()  # for logging only bs,ps,2

            out = (action,
                   log_prob,
                   entropy)
        else:
            out = (action,
                   log_prob,
                   )
        return out


class Critic(nn.Module):
    def __init__(self,
                 config
                 ):
        super(Critic, self).__init__()
        self.__value_head = MLP([{'in': config.feature_dim, 'out': 16, 'drop_out': 0, 'activation': 'ReLU'},
                                 {'in': 16, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                                 {'in': 8, 'out': 1, 'drop_out': 0, 'activation': 'None'}])

    def forward(self, h_features):
        baseline_value = self.__value_head(h_features)
        return baseline_value.detach().squeeze(), baseline_value.squeeze()


class RLEPSO_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)

        # add specified config
        config.feature_dim=1
        config.action_dim=35
        config.action_shape=(35,)
        config.n_step=10
        config.K_epochs=3
        config.eps_clip=0.1
        config.gamma=0.999
        config.max_sigma=0.7
        config.min_sigma=0.01
        config.lr=1e-5
        # config.lr_decay=0.99
        self.__config = config

        self.__device = self.__config.device
        # figure out the actor
        self.__actor = Actor(config).to(self.__device)

        # figure out the critic
        self.__critic = Critic(config).to(self.__device)

        # figure out the optimizer
        self.__optimizer_actor = torch.optim.Adam(
            [{'params': self.__actor.parameters(), 'lr': config.lr}])
        self.__optimizer_critic = torch.optim.Adam(
            [{'params': self.__critic.parameters(), 'lr': config.lr}])
        
        # figure out the lr schedule
        # self.__lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(self.__optimizer_critic, config.lr_decay, last_epoch=-1, )
        # self.__lr_scheduler_actor = torch.optim.lr_scheduler.ExponentialLR(self.__optimizer_actor, config.lr_decay, last_epoch=-1, )

        # init learning time
        self.__learning_time=0

        self.__cur_checkpoint=0

        # save init agent
        if self.__cur_checkpoint==0:
            save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
            self.__cur_checkpoint+=1

    def update_setting(self, config):
        self.__config.max_learning_step = config.max_learning_step
        self.__config.agent_save_dir = config.agent_save_dir
        self.__learning_time = 0
        save_class(self.__config.agent_save_dir, 'checkpoint0', self)
        self.__cur_checkpoint = 1

    def train_episode(self, env):
        config = self.__config
        # setup
        memory = Memory()
        # initial instances and solutions
        state = env.reset()
        state = torch.FloatTensor(state).to(self.__device)


        # params for training
        gamma = config.gamma
        n_step = config.n_step
        K_epochs = config.K_epochs
        eps_clip = config.eps_clip
        
        t = 0
        _R = 0
        # initial_cost = obj
        is_done = False
        # sample trajectory
        while not is_done:
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []

            while t - t_s < n_step:
                # encoding the state

                memory.states.append(state.clone())
                
                # get model output
                action, log_lh,  entro_p = self.__actor(state,
                                                        require_entropy=True,
                                                        )
                action = action.reshape(config.action_shape)
                memory.actions.append(action.clone().detach())
                action = action.cpu().numpy()
                memory.logprobs.append(log_lh)

                entropy.append(entro_p.detach().cpu())

                baseline_val_detached, baseline_val = self.__critic(state)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                next_state,rewards,is_done = env.step(action)
                _R += rewards
                memory.rewards.append(torch.FloatTensor([rewards]).to(config.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))

                # store info
                # total_cost = total_cost + gbest_val

                # next
                t = t + 1
                state=next_state
                state=torch.FloatTensor(state).to(config.device)
                if is_done:
                    
                    break

            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update        =======================

            # bs, ps, dim_f = state.size()

            old_actions = torch.stack(memory.actions)
            old_states = torch.stack(memory.states).detach() 
            
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            # Optimize PPO policy for K mini-epochs:
            old_value = None
            for _k in range(K_epochs):
                if _k == 0:
                    logprobs = memory.logprobs

                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    entropy = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):

                        # get new action_prob
                        _, log_p,  entro_p = self.__actor(old_states[tt],
                                                          fixed_action=old_actions[tt],
                                                          require_entropy=True,  # take same action
                                                          )

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val_detached, baseline_val = self.__critic(old_states[tt])

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                entropy = torch.stack(entropy).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get target value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.__critic(state)[0]

                # R = agent.critic(state)[0]
                critic_output = R.clone()
                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)
                Reward = Reward.view(-1)
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()

                # check K-L divergence (for logging only)
                approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
                approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
                # calculate loss
                loss = baseline_loss + reinforce_loss

                # update gradient step
                # agent.optimizer.zero_grad()
                self.__optimizer_actor.zero_grad()
                self.__optimizer_critic.zero_grad()
                baseline_loss.backward()
                reinforce_loss.backward()
                # loss.backward()


                # perform gradient descent
                self.__optimizer_actor.step()
                self.__optimizer_critic.step()
                self.__learning_time += 1

                if self.__learning_time >= (self.__config.save_interval * self.__cur_checkpoint):
                    save_class(self.__config.agent_save_dir, 'checkpoint'+str(self.__cur_checkpoint), self)
                    self.__cur_checkpoint += 1

                if self.__learning_time >= config.max_learning_step:
                    return self.__learning_time >= config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                                              'gbest': env.optimizer.cost[-1],
                                                                              'return': _R,
                                                                              'learn_steps': self.__learning_time}
                
            memory.clear_memory()
        return self.__learning_time >= config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                                  'gbest': env.optimizer.cost[-1],
                                                                  'return': _R,
                                                                  'learn_steps': self.__learning_time}

    def rollout_episode(self, env):
        is_done = False
        state = env.reset()
        R = 0
        while not is_done:
            state = torch.FloatTensor(state).to(self.__config.device)
            action = self.__actor(state)[0].cpu().numpy()
            state, reward, is_done = env.step(action)
            R += reward
        return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes, 'return': R}
        