import time

import numpy as np
from optimizer.rl_das_related.optimizer import JDE21, MadDE, NL_SHADE_RSP

from optimizer.rl_das_related.Population import *
import warnings
import copy
from optimizer.learnable_optimizer import Learnable_Optimizer

class RL_DAS_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.MaxFEs = config.maxFEs
        # self.period = 2500
        if config.problem in['protein','protein-torch']:
            self.period = 100
        else:
            self.period =2500
        self.max_step = self.MaxFEs // self.period
        self.sample_times = 2
        self.n_dim_obs = 6
        
        self.final_obs = None
        self.terminal_error = 1e-8
        
        self.__config = config

        self.FEs = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def init_population(self, problem):
        self.dim = problem.dim
        self.problem = problem

        optimizers = ['NL_SHADE_RSP', 'MadDE',  'JDE21']
        self.optimizers = []
        for optimizer in optimizers:
            self.optimizers.append(eval(optimizer)(self.dim))
        self.best_history = [[] for _ in range(len(optimizers))]
        self.worst_history = [[] for _ in range(len(optimizers))]

        self.population = Population(self.dim)
        self.population.initialize_costs(self.problem)
        self.cost_scale_factor = self.population.gbest
        self.FEs = self.population.NP
        self.done = False
        self.cost = [self.population.gbest]
        self.log_index = 1
        return self.observe(problem)

    def local_sample(self):
        samples = []
        costs = []
        min_len = 1e9
        sample_size = self.population.NP
        for i in range(self.sample_times):
            sample, _ = self.optimizers[np.random.randint(len(self.optimizers))].step(copy.deepcopy(self.population),
                                                                                         self.problem,
                                                                                         self.FEs,
                                                                                         self.FEs + sample_size,
                                                                                         self.MaxFEs)
            samples.append(sample)
            cost = sample.cost
            costs.append(cost)
            min_len = min(min_len, cost.shape[0])
        self.FEs += sample_size * self.sample_times
        if self.FEs >= self.MaxFEs:
            self.done = True
        for i in range(self.sample_times):
            costs[i] = costs[i][:min_len]
        return np.array(samples), np.array(costs)

    # observed env state
    def observe(self, problem):
        
        samples, sample_costs = self.local_sample()
        feature = self.population.get_feature(self.problem,
                                            sample_costs,
                                            self.cost_scale_factor,
                                            self.FEs / self.MaxFEs)
        
        # =======================================================================
        best_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        worst_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        move = np.zeros((len(self.optimizers) * 2, self.dim)).tolist()
        for i in range(len(self.optimizers)):
            if len(self.best_history[i]) > 0:
                move[i*2] = np.mean(self.best_history[i], 0).tolist()
                move[i * 2 + 1] = np.mean(self.worst_history[i], 0).tolist()
                best_move[i] = np.mean(self.best_history[i], 0).tolist()
                worst_move[i] = np.mean(self.worst_history[i], 0).tolist()
        move.insert(0, feature)
        return move

    def seed(self, seed=None):
        np.random.seed(seed)

    def update(self, action, problem):
        warnings.filterwarnings("ignore")
        act = action

        last_cost = self.population.gbest
        pre_best = self.population.gbest_solution
        pre_worst = self.population.group[np.argmax(self.population.cost)]
        period = self.period
        end = self.FEs + self.period
        while self.FEs < end and self.FEs < self.MaxFEs and self.population.gbest > self.terminal_error:                    
            optimizer = self.optimizers[act]
            FEs_end = self.FEs + period

            self.population, self.FEs = optimizer.step(self.population,
                                                        problem,
                                                        self.FEs,
                                                        FEs_end,
                                                        self.MaxFEs,
                                                        )
        end = time.time()
        pos_best = self.population.gbest_solution
        pos_worst = self.population.group[np.argmax(self.population.cost)]
        self.best_history[act].append((pos_best - pre_best) / 200)
        self.worst_history[act].append((pos_worst - pre_worst) / 200)
        if problem.optimum is None:
            self.done = self.FEs >= self.MaxFEs
        else:
            self.done = self.FEs >= self.MaxFEs or np.min(self.population.gbest) <= 1e-8
        # self.done = (self.population.gbest <= self.terminal_error or self.FEs >= self.MaxFEs)
        reward = max((last_cost - self.population.gbest) / self.cost_scale_factor, 0)

        observe = self.observe(problem)
        if self.FEs >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.population.gbest)
        
        if self.done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.population.gbest
            else:
                self.cost.append(self.population.gbest)
        return observe, reward, self.done, {} # next state, reward, is done
