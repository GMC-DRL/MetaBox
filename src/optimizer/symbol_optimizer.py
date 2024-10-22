import numpy as np
from .symbol_related.population import Population
from .symbol_related.madde import MadDE
from .symbol_related.tokenizer import MyTokenizer
from .symbol_related.expression import expr_to_func
import re
import copy

import numpy as np

import torch
from .learnable_optimizer import Learnable_Optimizer

'''Working environment for SYMBOL'''

# preprocess the randx operand
class Myreplace(object):
    def __init__(self):
        self.count = 0 

    def replace(self, match):
        self.count += 1
        if self.count > 1:
            return self.pattern + str(self.count-1)
        else:
            return match.group()

    def process_string(self, string, pattern):
        self.pattern=pattern
        self.count = 0
        new_string = re.sub(pattern, self.replace, string)
        return new_string

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


class Symbol_Optimizer(Learnable_Optimizer):

    # todo: integrate task function into optimizer
    def __init__(self, config):
        super().__init__(config)
        
        # !add
        symbol_config(config)

        self.tokenizer=MyTokenizer()
        self.__config = config
        self.dim = config.dim

        
        self.NP = 100
        
        self.no_improve=0
        self.per_no_improve=np.zeros((self.NP,))

        self.evaling=False
        self.max_fes= config.maxFEs
        
        
        self.boarder_method= 'periodic'
        
        self.replace=Myreplace()

        self.log_interval = config.log_interval
        
        # todo: modify teacher config
        self.teacher_optimizer = eval(config.teacher)(config)

        # ! change when using symbol_optimizer
        self.is_train = True
        self.log_interval = config.log_interval


    # the interface for environment reseting
    def init_population(self, problem):
        # self.NP=self.__Nmax
        self.max_x = problem.ub
        self.min_x = problem.lb
        self.problem = problem
        if self.is_train:
            tea_pop = self.teacher_optimizer.init_population(copy.deepcopy(problem))
            self.population=Population(self.dim,self.NP,self.min_x,self.max_x,self.max_fes,copy.deepcopy(problem))
            get_init_pop(tea_pop=tea_pop, stu_pop=self.population, method=self.__config.init_pop)
        else:
            self.population=Population(self.dim,self.NP,self.min_x,self.max_x,self.max_fes,problem)
            self.population.reset()

        self.log_index = 1
        self.cost = [self.population.gbest_cost]

        # return state
        return self.observe()

    def eval(self):
        self.evaling=True

    def train(self):
        # set_seed()
        self.evaling=False

    # feature encoding
    def observe(self):
        return self.population.feature_encoding()


    # input the self.population and expr function, return the population after applying expr function
    def update(self, action, problem):
        
        expr=action['expr']
        skip_step=action['skip_step']
        # debug
        # print(f"x + {expr}")
        # record the previous gbest
        self.population.pre_gbest=self.population.gbest_cost

        cnt_randx=expr.count('randx')
        pattern = 'randx'
        expr=self.replace.process_string(expr, pattern)
        count = self.replace.count

        assert count==cnt_randx,'randx count is wrong!!'
        variables=copy.deepcopy(self.tokenizer.variables)
        for i in range(1,count):
            variables.append(f'randx{i}')
        update_function=expr_to_func(expr,variables)

        for sub_step in range(skip_step):
            x=self.population.current_position
            
            gb=self.population.gbest_position[None,:].repeat(self.NP,0)
            gw=self.population.gworst_position[None,:].repeat(self.NP,0)

            dx=self.population.delta_x
            randx=x[np.random.randint(self.NP, size=self.NP)]
            
            pbest=self.population.pbest_position
            
            names = locals()
            inputs=[x,gb,gw,dx,randx,pbest]
            for i in range(1,count):
                names[f'randx{i}']=x[np.random.randint(self.NP, size=self.NP)]
                inputs.append(eval(f'randx{i}'))
                
            assert x.shape==gb.shape==gw.shape==dx.shape==randx.shape, 'not same shape'
            
            next_position=x+update_function(*inputs)
            
            # boarder clip or what
            if self.boarder_method=="clipping":
                next_position=np.clip(next_position,self.min_x,self.max_x)
            elif self.boarder_method=="periodic":
                next_position=self.min_x+(next_position-self.max_x)%(self.max_x-self.min_x)
            else:
                raise AssertionError('this board method is not surported!')
            
            # update population
            self.population.update(next_position, filter_survive=False)
        
        if self.population.cur_fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.population.gbest_cost)

        reward = 0
        if self.is_train:
            tea_pop, _, _, _ = self.teacher_optimizer.update({'skip_step': skip_step})
            # cal reward
            reward = self.cal_reward(tea_pop, max_step=self.max_fes / self.NP / skip_step)
        else:
            reward = (self.population.pre_gbest - self.population.gbest_cost) / (self.population.init_cost-0)

        is_end = False
        # see if the end condition is satisfied
        if problem.optimum is None:
            is_end = self.population.cur_fes >= self.max_fes
        else:
            is_end = self.population.cur_fes >= self.max_fes or self.population.gbest_cost <= 1e-8

        if is_end:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.population.gbest_cost
            else:
                self.cost.append(self.population.gbest_cost)
            # print(f'problem: {self.problem.__str__()}')

        return self.observe(), reward, is_end

    def cal_reward(self, tea_pop, max_step):
        dist = cal_gap_nearest(self.population, tea_pop)
        
        imitation_r = -dist / max_step

        # base_reward = -(self.population.gbest_cost- self.problem.optimum) / (self.population.init_cost-self.problem.optimum) / max_step
        base_reward = (self.population.pre_gbest - self.population.gbest_cost) / (self.population.init_cost-0)
        
        return imitation_r + base_reward
        # return base_reward


'''forming init pop'''
def get_init_pop(tea_pop,stu_pop,method):
    if method=='best':
        sort_index=np.argsort(tea_pop.c_cost)
        init_pos=tea_pop.current_position[sort_index[:stu_pop.pop_size]]
        stu_pop.reset(init_pop=init_pos)
    elif method == 'harf':
        sort_index=np.argsort(tea_pop.c_cost)
        init_pos=np.concatenate((tea_pop.current_position[sort_index[:int(stu_pop.pop_size*0.5)]],tea_pop.current_position[sort_index[:stu_pop.pop_size-int(stu_pop.pop_size*0.5)]]),axis=0)
        stu_pop.reset(init_pop=init_pos)
    elif method == 'random':
        rand_index=np.random.randint(0,tea_pop.pop_size,size=(stu_pop.pop_size,))
        init_pos=tea_pop.current_position[rand_index]
        stu_pop.reset(init_pop=init_pos)
    elif method == 'uniform':
        sort_index=np.argsort(tea_pop.c_cost)
        init_pos=tea_pop.current_position[sort_index[::tea_pop.pop_size//stu_pop.pop_size]]
        stu_pop.reset(init_pop=init_pos)
    else:
        raise ValueError('init pop method is currently not supported!!')
    
def cal_gap_nearest(stu_pop,tea_pop):
    max_x=stu_pop.max_x

    stu_position=stu_pop.current_position
    tea_position=tea_pop.current_position
    
    norm_p1=stu_position/max_x
    norm_p1=norm_p1[None,:,:]
    norm_p2=tea_position/max_x
    norm_p2=norm_p2[:,None,:]
    dist=np.sqrt(np.sum((norm_p2-norm_p1)**2,-1))
    min_dist=np.min(dist,-1)

    gap=np.max(min_dist)
    dim=stu_position.shape[1]
    max_dist=2*np.sqrt(dim)
    return gap/max_dist