from L2OBench.Optimizer.basic_optimizer import basic_optimizer
from L2OBench.Problem.cec_dataset import Training_Dataset
from L2OBench.Environment.basic_environment import Env
from L2OBench.config import get_config
import numpy as np
import torch
import random
import math
import copy
"""
for traditional optimizer, we need to inherit the basic_optimizer class
and for now optimizer use l20bench only with problem, config (no env and agent)
"""
class sDMSPSO(basic_optimizer):
    def __init__(self,config):
        super(sDMSPSO, self).__init__()
        self.config = config
        # problem_dim pop_size max_fes max_x
        self.dim, self.ps, self.max_fes, self.max_x = config.dim, config.ps, config.max_fes, config.max_x
        # w c1 c2 m R
        self.w, self.c1, self.c2 = config.w,config.c1, config.c2
        self.m, self.R = config.m, config.R
        self.cuda = config.cuda
        self.n_swarm = self.ps // self.m
        self.fes = 0
        self.max_velocity = config.max_velocity
        # check if has agent
        self.origin, self.origin_type = config.origin, config.origin_type
        self.boarder_method = config.boarder_method
        self.reward_func = config.reward_func
        self.w_decay = config.w_decay

        self.group_index = np.zeros(config.ps, dtype=np.int8)
        self.per_no_improve = np.zeros(config.ps)
        self.lbest_no_improve = np.zeros(self.n_swarm)
        self.max_dist = np.sqrt((2 * config.max_x) ** 2 * config.dim)
        self.max_step = self.max_fes // self.ps
        assert config.ps % config.m == 0, 'population cannot be update averagely'
        for sub_swarm in range(self.n_swarm):
            if sub_swarm != self.n_swarm - 1:
                self.group_index[sub_swarm * self.m:(sub_swarm + 1) * self.m] = sub_swarm
            else:
                self.group_index[sub_swarm * self.m:] = sub_swarm
        self.node_dim = 9


    def get_cost(self,problem,position):
        ps = position.shape[0]
        self.fes += ps
        cost = problem.eval(position)
        return cost

    def initilize(self,problem):
        # randomly generate the positions and velocities
        rand_pos = np.random.uniform(low=-self.max_x, high=self.max_x, size=(self.ps, self.dim))
        rand_vel = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.ps, self.dim))

        # get the initial costs
        c_cost = self.get_costs(problem,rand_pos)

        # find out the gbest
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position = rand_pos[gbest_index]
        self.max_cost = np.min(c_cost)

        # store the population infomation
        self.particles = {'current_position': rand_pos.copy(),  # ps, dim
                          'c_cost': c_cost.copy(),  # ps
                          'pbest_position': rand_pos.copy(),  # ps, dim
                          'pbest': c_cost.copy(),  # ps
                          'gbest_position': gbest_position.copy(),  # dim
                          'gbest_val': gbest_val,  # 1
                          'velocity': rand_vel.copy(),  # ps,dim
                          }
        self.particles['lbest_cost'] = np.zeros(self.n_swarm)
        self.particles['lbest_position'] = np.zeros((self.n_swarm, self.dim))

        # the exploration and exploitation feature, they will be updated or generated in self.update_lbest()
        # self.pbest_feature = self.input_feature_encoding()
        # self.lbest_feature = np.zeros((self.n_swarm, self.node_dim))

    def reset(self,problem):
        if self.w_decay:
            self.w=0.9
        self.gen=0
        self.fes-=self.fes
        self.per_no_improve-=self.per_no_improve
        self.lbest_no_improve-=self.lbest_no_improve
        self.initilize(problem)
        self.random_regroup()
        self.update_lbest(init=True)
        self.fes_eval=np.zeros_like(self.fes)
        # return self.get_state(self.input_feature_encoding())

    # randomly regruop the population
    def random_regroup(self):
        regroup_index=torch.randperm(n=self.ps)
        self.lbest_no_improve-=self.lbest_no_improve
        self.regroup_index=regroup_index
        self.particles['current_position']=self.particles['current_position'][regroup_index]
        self.particles['c_cost']= self.particles['c_cost'][regroup_index]
        self.particles['pbest_position']=self.particles['pbest_position'][regroup_index]
        self.particles['pbest']= self.particles['pbest'][regroup_index]
        self.particles['velocity']=self.particles['velocity'][regroup_index]
        self.per_no_improve=self.per_no_improve[regroup_index]
        # self.pbest_feature=self.pbest_feature[regroup_index]


    # update lbest position, lbest cost
    def update_lbest(self,init=False):
        if init:
            # find the lbest position and lbest cost
            grouped_pbest=self.particles['pbest'].reshape(self.n_swarm,self.m)
            grouped_pbest_pos=self.particles['pbest_position'].reshape(self.n_swarm,self.m,self.dim)
            grouped_pbest_fea=self.pbest_feature.reshape(self.n_swarm,self.m,self.node_dim)
            self.particles['lbest_cost']=np.min(grouped_pbest,axis=-1)
            index=np.argmin(grouped_pbest,axis=-1)
            self.particles['lbest_position']=grouped_pbest_pos[range(self.n_swarm),index]

            # generate the lbest_feature(exploitation feature)
            # self.lbest_feature=grouped_pbest_fea[range(self.n_swarm),index]
        else:
            # update the lbest position and lbest cost
            grouped_pbest=self.particles['pbest'].reshape(self.n_swarm,self.m)
            grouped_pbest_pos=self.particles['pbest_position'].reshape(self.n_swarm,self.m,self.dim)
            lbest_cur=np.min(grouped_pbest,axis=-1)
            index=np.argmin(grouped_pbest,axis=-1)
            lbest_pos_cur=grouped_pbest_pos[range(self.n_swarm),index]
            filter_lbest=lbest_cur<self.particles['lbest_cost']
            self.particles['lbest_cost']=np.where(filter_lbest,lbest_cur,self.particles['lbest_cost'])
            self.particles['lbest_position']=np.where(filter_lbest[:,None],lbest_pos_cur,self.particles['lbest_position'])
            self.lbest_no_improve=np.where(filter_lbest,np.zeros_like(self.lbest_no_improve),self.lbest_no_improve+1)


    def train(self):

        pass

    def get_best(self):
        return self.particles['gbest_val'], self.fes_eval

    def update(self,problem):
        pre_gbest = self.particles['gbest_val']
        is_end = False
        self.gen += 1

        # linearly decreasing the coefficient of inertia w
        if self.w_decay:
            self.w -= 0.5 / (self.max_fes / self.ps)

        # if dmspso is not controlled by RL_agent, it will have two mode, one is local search, the other is the global search which is the same as GPSO
        # but if the DMSPSO is controlled by RL_agent, it will only have the local search mode
        cur_mode = 'ls'
        if self.fes >= 0.9 * self.max_fes and self.origin:
            cur_mode = 'gs'
        if self.origin_type == 'fixed':
            c1 = self.c1
            c2 = self.c2
        elif self.origin_type == 'normal':
            rand_mu = np.random.rand(self.ps)
            rand_std = np.random.rand(self.ps) * 0.7
            action = np.random.normal(loc=rand_mu, scale=rand_std)
            action = np.clip(action, a_min=0, a_max=1)
            action = action[:, None]
            c_sum = self.c1 + self.c2
            c1 = action * c_sum
            c2 = c_sum - c1
        # if self.origin:
        #     if self.origin_type == 'fixed':
        #         c1 = self.c1
        #         c2 = self.c2
        #     elif self.origin_type == 'normal':
        #         rand_mu = np.random.rand(self.ps)
        #         rand_std = np.random.rand(self.ps) * 0.7
        #         action = np.random.normal(loc=rand_mu, scale=rand_std)
        #         action = np.clip(action, a_min=0, a_max=1)
        #         action = action[:, None]
        #         c_sum = self.c1 + self.c2
        #         c1 = action * c_sum
        #         c2 = c_sum - c1
        # else:
        #     # if the algorithm is controlled by RL agent, get parameters from actions
        #     c_sum = self.c1 + self.c2
        #     action = action[:, None]
        #     c1 = action * c_sum
        #     c2 = c_sum - c1
        # update velocity
        rand1 = np.random.rand(self.ps, 1)
        rand2 = np.random.rand(self.ps, 1)
        v_pbest = rand1 * (self.particles['pbest_position'] - self.particles['current_position'])
        if cur_mode == 'ls':
            v_lbest = rand2 * (self.particles['lbest_position'][self.group_index] - self.particles['current_position'])
            new_velocity = self.w * self.particles['velocity'] + c1 * v_pbest + c2 * v_lbest
        elif cur_mode == 'gs':
            v_gbest = rand2 * (self.particles['gbest_position'][None, :] - self.particles['current_position'])
            new_velocity = self.w * self.particles['velocity'] + c1 * v_pbest + c2 * v_gbest
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        # update position
        if self.boarder_method == "clipping":
            raw_position = self.particles['current_position'] + new_velocity
            new_position = np.clip(raw_position, -self.max_x, self.max_x)
        elif self.boarder_method == "random":
            raw_position = self.particles['current_position'] + new_velocity
            filter = raw_position.abs() > self.max_x
            new_position = np.where(filter,
                                    np.random.uniform(low=-self.max_x, high=self.max_x, size=(self.ps, self.dim)),
                                    raw_position)
        elif self.boarder_method == "periodic":
            raw_position = self.particles['current_position'] + new_velocity
            new_position = -self.max_x + ((raw_position - self.max_x) % (2. * self.max_x))
        elif self.boarder_method == "reflect":
            raw_position = self.particles['current_position'] + new_velocity
            filter_low = raw_position < -self.max_x
            filter_high = raw_position > self.max_x
            new_position = np.where(filter_low, -self.max_x + (-self.max_x - raw_position), raw_position)
            new_position = np.where(filter_high, self.max_x - (new_position - self.max_x), new_position)

        # get new cost
        new_cost = self.get_costs(problem,new_position)

        # update particles
        filters = new_cost < self.particles['pbest']
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        filters_best_val = new_cbest_val < self.particles['gbest_val']

        new_particles = {'current_position': new_position,  # bs, ps, dim
                         'c_cost': new_cost,  # bs, ps
                         'pbest_position': np.where(np.expand_dims(filters, axis=-1),
                                                    new_position,
                                                    self.particles['pbest_position']),
                         'pbest': np.where(filters,
                                           new_cost,
                                           self.particles['pbest']),
                         'velocity': new_velocity,
                         'gbest_val': np.where(filters_best_val,
                                               new_cbest_val,
                                               self.particles['gbest_val']),
                         'gbest_position': np.where(np.expand_dims(filters_best_val, axis=-1),
                                                    new_position[new_cbest_index],
                                                    self.particles['gbest_position']),
                         'lbest_position': self.particles['lbest_position'],
                         'lbest_cost': self.particles['lbest_cost']
                         }

        # update the stag step for every single
        filter_per_patience = new_particles['c_cost'] < self.particles['c_cost']
        self.per_no_improve += 1
        tmp = np.where(filter_per_patience, self.per_no_improve, np.zeros_like(self.per_no_improve))
        self.per_no_improve -= tmp

        self.particles = new_particles

        # update pbest feature
        # self.update_pbest_feature(filters)

        # update lbest-related information
        self.update_lbest()
        reward = self.cal_reward(self.particles['gbest_val'], pre_gbest)

        # regroup the population periodically
        if self.gen % self.R == 0:
            self.random_regroup()
            self.update_lbest(init=True)

        # see if the end condition is satisfied
        if self.fes >= self.max_fes:
            is_end = True
        if self.particles['gbest_val'] <= 1e-8:
            is_end = True

        # update state
        # next_state = self.input_feature_encoding()
        # next_state=self.observe(next_state)
        # next_state = self.get_state(self.input_feature_encoding())
        # info = {'gbest_val': self.particles['gbest_val'], 'fes_used': self.fes_eval}

        # return (next_state, reward, is_end, info)
        return self.particles['gbest_val'], self.fes_eval


def main():
    problem = Training_Dataset(dim=10,
                               num_samples=1,
                               batch_size=1,
                               problems='Sphere',
                               shifted=True,
                               rotated=True,
                               biased=True)[0][0]
    env = Env(problem)
    config = get_config()
    optim = sDMSPSO(config)
    optim.reset(problem)
    optim.update(problem)

