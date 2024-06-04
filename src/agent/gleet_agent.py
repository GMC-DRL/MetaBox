from typing import Tuple
from agent.basic_agent import Basic_Agent
import torch
import math

from torch import nn
import torch
from agent.networks import MultiHeadEncoder, MLP, EmbeddingNet, MultiHeadCompat
from torch.distributions import Normal
import torch.nn.functional as F
from agent.utils import *

# memory for recording transition during training process
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

class GLEET_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config

        # define parameters
        self.__config.embedding_dim = 16
        self.__config.encoder_head_num = 4
        self.__config.decoder_head_num = 4
        self.__config.n_encode_layers = 1
        self.__config.normalization = 'layer'
        self.__config.v_range = 6
        
        self.__config.hidden_dim = 16
        self.__config.node_dim = 9
        self.__config.hidden_dim1_actor = 32
        self.__config.hidden_dim2_actor = 8
        self.__config.max_sigma = 0.7
        self.__config.min_sigma = 0.01
        self.__config.hidden_dim1_critic = 32
        self.__config.hidden_dim2_critic = 16
        self.__config.gamma = 0.999
        self.__config.n_step = 10
        self.__config.K_epochs = 3
        self.__config.eps_clip = 0.1
        self.__config.lr_model = 1e-4
        self.__config.lr_decay = 0.9862327
        self.__config.max_grad_norm = 0.1
    
        # figure out the actor network
        self.actor = Actor(
            embedding_dim = self.__config.embedding_dim,
            hidden_dim = self.__config.hidden_dim,
            n_heads_actor = self.__config.encoder_head_num,
            n_heads_decoder = self.__config.decoder_head_num,
            n_layers = self.__config.n_encode_layers,
            normalization = self.__config.normalization,
            v_range = self.__config.v_range,
            node_dim=self.__config.node_dim,
            hidden_dim1=self.__config.hidden_dim1_actor,
            hidden_dim2=self.__config.hidden_dim2_actor,
            max_sigma=self.__config.max_sigma,
            min_sigma=self.__config.min_sigma,
        )
        
        
        input_critic=self.__config.embedding_dim
        # figure out the critic network
        self.critic = Critic(
            input_dim = input_critic,
            hidden_dim1 = self.__config.hidden_dim1_critic,
            hidden_dim2 = self.__config.hidden_dim2_critic,
        )

        # figure out the optimizer
        self.optimizer = torch.optim.Adam(
            [{'params': self.actor.parameters(), 'lr': self.__config.lr_model}] +
            [{'params': self.critic.parameters(), 'lr': self.__config.lr_model}])
        # figure out the lr schedule
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.__config.lr_decay, last_epoch=-1,)

        # move to device
        self.actor.to(self.__config.device)
        self.critic.to(self.__config.device)


        # init learning time
        self.__learning_time=0

        self.__cur_checkpoint=0

        # save init agent
        if self.__cur_checkpoint==0:
            save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
            self.__cur_checkpoint+=1



    def train_episode(self, env):
        memory = Memory()

        state = env.reset()[None, :]
        state = torch.FloatTensor(state).to(self.__config.device)
        # params for training
        gamma = self.__config.gamma
        n_step = self.__config.n_step
        
        K_epochs = self.__config.K_epochs
        eps_clip = self.__config.eps_clip
        
        t = 0
        # initial_cost = obj
        done=False
        _R = 0
        # sample trajectory
        while not done:
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step :  
                
                memory.states.append(state.clone())
                action, log_lh,_to_critic,  entro_p  = self.actor(state,
                                                        require_entropy = True,
                                                        to_critic=True
                                                        )
                

                memory.actions.append(action.clone())
                memory.logprobs.append(log_lh)
                action=action.cpu().numpy().squeeze()

                entropy.append(entro_p.detach().cpu())

                baseline_val_detached, baseline_val = self.critic(_to_critic)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)


                # state transient
                state, rewards, is_end = env.step(action)
                memory.rewards.append(torch.FloatTensor([rewards]).to(self.__config.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
                _R += rewards.squeeze()
                # store info
                # total_cost = total_cost + gbest_val

                # next
                t = t + 1

                state = torch.FloatTensor(state[None, :]).to(self.__config.device)
                
                if is_end:
                    done=True
                    break


            
            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update

            old_actions = torch.stack(memory.actions)
            old_states = torch.stack(memory.states).detach() #.view(t_time, bs, ps, dim_f)
            # old_actions = all_actions.view(t_time, bs, ps, -1)
            # print('old_actions.shape:{}'.format(old_actions.shape))
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
                        _, log_p,_to_critic,  entro_p = self.actor(old_states[tt],
                                                        fixed_action = old_actions[tt],
                                                        require_entropy = True,# take same action
                                                        to_critic = True
                                                        )

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val_detached, baseline_val = self.critic(_to_critic)

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                entropy = torch.stack(entropy).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)


                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(self.actor(state,only_critic = True))[0]

                # R = self.critic(x_in)[0]
                critic_output=R.clone()
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
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradient norm and get (clipped) gradient norms for logging
                # current_step = int(pre_step + t//n_step * K_epochs  + _k)
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.__config.max_grad_norm)

                # perform gradient descent
                self.optimizer.step()
                self.__learning_time += 1
                if self.__learning_time >= (self.__config.save_interval * self.__cur_checkpoint):
                    save_class(self.__config.agent_save_dir, 'checkpoint'+str(self.__cur_checkpoint), self)
                    self.__cur_checkpoint += 1

                if self.__learning_time >= self.__config.max_learning_step:
                    return self.__learning_time >= self.__config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                                              'gbest': env.optimizer.cost[-1],
                                                                              'return': _R,
                                                                              'learn_steps': self.__learning_time}

                
            
            memory.clear_memory()
        return self.__learning_time >= self.__config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                                  'gbest': env.optimizer.cost[-1],
                                                                  'return': _R,
                                                                  'learn_steps': self.__learning_time}
    
    def rollout_episode(self, env):
        with torch.no_grad():
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                state = torch.FloatTensor(state[None, :]).to(self.__config.device)
                action = self.actor(state)[0]
                action = action.cpu().numpy().squeeze()
                state, reward, is_done = env.step(action)
                R += reward
            return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes, 'return': R} 

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Actor(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_heads_decoder,
                 n_layers,
                 normalization,
                 v_range,
                 node_dim,
                 hidden_dim1,
                 hidden_dim2,
                 max_sigma=0.7,
                 min_sigma=1e-3,
                 ):
        super(Actor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_heads_decoder = n_heads_decoder        
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.node_dim = node_dim 

        # figure out the Actor network
        # figure out the embedder for feature embedding
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim)
        # figure out the fully informed encoder
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization,)
            for _ in range(self.n_layers))) # stack L layers

        # w/o eef for ablation study
        # figure out the embedder for exploration and exploitation feature
        self.embedder_for_decoder = EmbeddingNet(2*self.embedding_dim, self.embedding_dim)
        # figure out the exploration and exploitation decoder
        self.decoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization,)
            for _ in range(self.n_layers))) # stack L layers
        
        # figure out the mu_net and sigma_net
        mlp_config = [{'in': self.embedding_dim,'out': hidden_dim1,'drop_out': 0,'activation':'LeakyReLU'},
                  {'in': hidden_dim1,'out': hidden_dim2,'drop_out':0,'activation':'LeakyReLU'},
                  {'in': hidden_dim2,'out': 1,'drop_out':0,'activation':'None'}]
        self.mu_net = MLP(mlp_config) 
        self.sigma_net=MLP(mlp_config)
        

        self.max_sigma=max_sigma
        self.min_sigma=min_sigma
        
        print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in,fixed_action = None, require_entropy = False,to_critic=False,only_critic=False):
        
        population_feature=x_in[:,:,:self.node_dim]
        eef=x_in[:,:,self.node_dim:]
        # pass through embedder
        h_em = self.embedder(population_feature)
        # pass through encoder
        logits = self.encoder(h_em)
        # pass through the embedder to get eef embedding
        exploration_feature=eef[:,:,:self.node_dim]
        exploitation_feature=eef[:,:,self.node_dim:]
        
        exploration_eb=self.embedder(exploration_feature)   
        exploitation_eb=self.embedder(exploitation_feature)   
        x_in_decoder=torch.cat((exploration_eb,exploitation_eb),dim=-1)

        # pass through the embedder for decoder
        x_in_decoder = self.embedder_for_decoder(x_in_decoder)

        # pass through decoder
        logits = self.decoder(logits,x_in_decoder)
            
        # share logits to critic net, where logits is from the decoder output 
        if only_critic:
            return logits  # .view(bs, dim, ps, -1)
        # finally decide the mu and sigma
        mu = (torch.tanh(self.mu_net(logits))+1.)/2.
        sigma=(torch.tanh(self.sigma_net(logits))+1.)/2. * (self.max_sigma-self.min_sigma)+self.min_sigma
        

        # don't share the network between actor and critic if there is no attention mechanism
        _to_critic= logits

        policy = Normal(mu, sigma)
        

        if fixed_action is not None:
            action = torch.tensor(fixed_action)
        else:
            # clip the action to (0,1)
            action=torch.clamp(policy.sample(),min=0,max=1)
        # get log probability
        log_prob=policy.log_prob(action)

        # The log_prob of each instance is summed up, since it is a joint action for a population
        log_prob=torch.sum(log_prob,dim=1)

        
        if require_entropy:
            entropy = policy.entropy() # for logging only 
            
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   entropy)
        else:
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   )
        return out

class Critic(nn.Module):
    def __init__(self,
             input_dim,
             hidden_dim1,
             hidden_dim2
             ):
        
        super(Critic, self).__init__()
        self.input_dim = input_dim
        # for GLEET, hidden_dim1 = 32, hidden_dim2 = 16
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        mlp_config = [{'in': self.input_dim,'out': hidden_dim1, 'drop_out': 0,'activation':'LeakyReLU'},
                  {'in': hidden_dim1,'out': hidden_dim2,'drop_out':0,'activation':'LeakyReLU'},
                  {'in': hidden_dim2,'out': 1,'drop_out':0,'activation':'None'}]
        self.value_head=MLP(config=mlp_config)

    def forward(self, h_features):
        # since it's joint actions, the input should be meaned at population-dimention
        h_features=torch.mean(h_features,dim=-2)
        # pass through value_head to get baseline_value
        baseline_value = self.value_head(h_features)
        
        return baseline_value.detach().squeeze(), baseline_value.squeeze()

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