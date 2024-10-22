from .symbol_related.critic import Critic
from .symbol_related.lstm import LSTM
from .symbol_related.expression import *
from .symbol_related.tokenizer import MyTokenizer


import torch
from .basic_agent import Basic_Agent
from .utils import *

class Data_Memory():
    def __init__(self) -> None:
        self.teacher_cost=[]
        self.stu_cost=[]
        self.baseline_cost=[]
        self.gap=[]
        self.baseline_gap=[]
        self.expr=[]
    
    def clear(self):
        del self.teacher_cost[:]
        del self.stu_cost[:]
        del self.baseline_cost[:]
        del self.gap[:]
        del self.baseline_gap[:]
        del self.expr[:]

# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.gap_rewards=[]
        self.b_rewards=[]

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.gap_rewards[:]
        del self.b_rewards[:]

def symbol_config(config):
    config.init_pop = 'random'
    config.teacher = 'MadDE'
    config.population_size = 100
    config.boarder_method = 'clipping'
    config.skip_step = 5
    config.test_skip_step = 5
    config.max_c = 1.
    config.min_c = -1.
    config.c_interval = 0.4
    config.max_layer = 6
    config.value_dim = 1
    config.hidden_dim = 16
    config.num_layers = 1
    config.lr = 1e-3
    config.lr_critic=1e-3
    
    config.encoder_head_num=4
    config.decoder_head_num=4
    config.critic_head_num=4
    config.embedding_dim=16
    config.n_encode_layers=1
    config.normalization='layer'
    config.hidden_dim1_critic=32
    config.hidden_dim2_critic=16
    config.hidden_dim1_actor=32
    config.hidden_dim2_actor=8
    config.output_dim_actor=1
    # config.lr_decay=0.9862327
    config.gamma=0.99
    config.K_epoch=3
    config.eps_clip=0.1
    config.n_step=10
    config.k_epoch=3

class Symbol_Agent(Basic_Agent):
    def __init__(self, config) -> None:
        
        symbol_config(config)
        
        config.fea_dim = 9
        
        self.tokenizer = MyTokenizer()
        self.actor=LSTM(opts=config, tokenizer=self.tokenizer)
        self.critic=Critic(opts=config)

        self.config= config
        self.optimizer=torch.optim.Adam([{'params':self.actor.parameters(),'lr':config.lr}] + 
                                        [{'params':self.critic.parameters(),'lr':config.lr_critic}])
        # figure out the lr schedule
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        
        self.actor.to(self.config.device)
        self.critic.to(self.config.device)
        
        self.__learning_time = 0

        self.__cur_checkpoint=0

        # save init agent
        if self.__cur_checkpoint==0:
            save_class(self.config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
            self.__cur_checkpoint+=1


    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.__learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.__cur_checkpoint = 1

    # training for one batch 
    def train_episode(self, env):
        env.optimizer.is_train = True
        
        memory=Memory()

        state = env.reset()[None, :]
        # print(f'state: {state.shape}')
        state = torch.FloatTensor(state).to(self.config.device)


        gamma = self.config.gamma
        eps_clip = self.config.eps_clip
        n_step = self.config.n_step
        k_epoch=self.config.k_epoch

        t=0
        _R = 0

        # bat_step=self.config.max_fes//self.config.population_size//self.config.skip_step
        is_done=False
        while not is_done:
            t_s=t

            bl_val_detached = []
            bl_val = []

            while t-t_s < n_step:
                # get feature
                memory.states.append(state)

                # using model to generate expr
                self.config.require_baseline = False

                seq,const_seq,log_prob,action_dict=self.actor(state,save_data=True)
                

                # critic network
                baseline_val_detached,baseline_val=self.critic(state)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)
                
                # store reward for ppo
                memory.actions.append(action_dict)
                memory.logprobs.append(log_prob)
                
                # todo: construct action
                expr = construct_action(seq=seq, const_seq=const_seq, tokenizer=self.tokenizer)
                action = {'expr': expr, 'skip_step': self.config.skip_step}
                state, reward, is_done = env.step(action)
                memory.rewards.append(reward)
                _R += reward

                t=t+1

                state=torch.FloatTensor(state[None, :]).to(self.config.device)

                
                if is_done:
                    break

            t_time=t-t_s
            
            # debug
            # print(f"expr: x + {expr}")

            # begin updating network in PPO style
            old_actions = memory.actions
            old_states = torch.stack(memory.states).detach() 
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            old_value = None
            for _k in range(k_epoch):
                
                if _k == 0:
                    logprobs = memory.logprobs
                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):

                        # get new action_prob
                        log_p = self.actor(old_states[tt],fix_action = old_actions[tt])

                        logprobs.append(log_p)
                        
                        baseline_val_detached, baseline_val = self.critic(old_states[tt])

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)
                logprobs = torch.stack(logprobs).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
               
                R = self.critic(state)[0]
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
                # calculate loss
                loss = baseline_loss + reinforce_loss
                
                # see if loss is nan
                if torch.isnan(loss):
                    print(f'baseline_loss:{baseline_loss}')
                    print(f'reinforce_loss:{reinforce_loss}')
                    assert True, 'nan found in loss!!'

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()

                # perform gradient descent
                self.optimizer.step()
                self.__learning_time += 1
                if self.__learning_time >= (self.config.save_interval * self.__cur_checkpoint):
                    save_class(self.config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
                    self.__cur_checkpoint+=1
                # if self.__learning_time >= self.config.max_learning_step:
                #     return self.__learning_time >= self.config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                #                                                               'gbest': env.optimizer.cost[-1],
                #                                                               'return': _R,
                #                                                               'learn_steps': self.__learning_time}

                
            memory.clear_memory()

        
        # return batch step
        return self.__learning_time >= self.config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                                  'gbest': env.optimizer.cost[-1],
                                                                  'return': _R,
                                                                  'learn_steps': self.__learning_time}
    
    def rollout_episode(self, env):
        env.optimizer.is_train = False
        with torch.no_grad():
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                state = torch.FloatTensor(state[None, :]).to(self.config.device)
                seq,const_seq,log_prob=self.actor(state,save_data=False)
                expr = construct_action(seq=seq, const_seq=const_seq, tokenizer=self.tokenizer)
                action = {'expr': expr, 'skip_step': self.config.test_skip_step}
                state, reward, is_done = env.step(action)
                R += reward
            return {'cost': env.optimizer.cost, 'fes': env.optimizer.population.cur_fes, 'return': R} 


def construct_action(seq, const_seq, tokenizer):
    pre,c_pre=get_prefix_with_consts(seq[0],const_seq[0],0)
    str_expr=[tokenizer.decode(pre[i]) for i in range(len(pre))]
    success,infix=prefix_to_infix(str_expr,c_pre,tokenizer)
    assert success, 'fail to construct the update function'

    return infix