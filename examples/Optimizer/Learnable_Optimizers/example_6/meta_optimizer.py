import numpy as np
from L2OBench.Optimizer import Learnable_Optimizer, clipping

class meta_optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config.dim, config.lower_bound, config.upper_bound, config.population_size, config.max_fes)

        self.config = config
        self.dim = self.config.dim
        self.NP = self.config.NP

        self.fes = 0
        self.max_fes = self.config.max_fes
        self.is_done = False

        self.name = 'meta_optimizer'
        # pop_size = 1


    def get_costs(self,problem,position):
        self.fes += self.NP
        cost = problem.eval(position) - problem.optimum
        return cost


    def init_population(self, problem):
        super().init_population(problem)


    def update(self, problem, delta_x):
        # print("popshape",self.population.shape,type(self.population))
        new_population = self.population + delta_x
        self.population = new_population
        # print("position:",self.population)
        new_cost = self.get_costs(problem, new_population)
        self.cost = new_cost