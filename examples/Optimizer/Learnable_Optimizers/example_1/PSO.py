"""
    An example of controlling hyper-parameters individual-wisely in PSO,
    while action is presented as a np.ndarray.
"""

import numpy as np
from L2OBench.Optimizer import Learnable_Optimizer, clipping


class PSO(Learnable_Optimizer):
    def __init__(self, dim, lower_bound, upper_bound, population_size, maxFEs, w_decay=True, max_velocity=10):
        super().__init__(dim, lower_bound, upper_bound, population_size, maxFEs)
        self.w_decay = w_decay
        self.max_velocity = max_velocity

        self.velocity = None
        self.w = None
        self.gbest = None
        self.pbest = None
        self.pbest_cost = None

    def init_population(self, problem):
        super().init_population(problem)
        self.velocity = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.NP, self.dim))
        self.gbest = self.population[self.cost.argmin()].copy()
        self.pbest = self.population.copy()
        self.pbest_cost = self.cost.copy()
        if self.w_decay:
            self.w = 0.9
        else:
            self.w = 0.729

    def update(self, problem, action):
        """
        :param problem: Problem instance.
        :param action: A np.ndarray of shape [2, NP], while action[0] represent c1 and action[1] represent c2.
        """
        # linearly decreasing the coefficient of inertia w
        if self.w_decay:
            self.w -= 0.5 / (self.maxFEs / self.NP)
        # generate two set of random val for pso velocity update
        rand1 = np.random.rand(self.NP, 1)
        rand2 = np.random.rand(self.NP, 1)
        # update velocity
        new_velocity = self.w * self.velocity + action[0][:, None] * rand1 * (self.pbest - self.population) + \
                       action[1][:, None] * rand2 * (self.gbest - self.population)
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)
        # get new population
        new_population = self.population + new_velocity
        new_population = clipping(new_population, self.lb, self.ub)
        # get new cost
        new_cost = problem.eval(new_population) - problem.optimum
        self.fes += self.NP
        # update particles
        pbest_filters = new_cost < self.pbest_cost
        cbest_cost = new_cost.min()
        cbest_index = new_cost.argmin()
        gbest_filter = cbest_cost < self.gbest_cost

        self.population = new_population
        self.velocity = new_velocity
        self.cost = new_cost
        self.gbest = np.where(np.expand_dims(gbest_filter, axis=-1),
                              new_population[cbest_index],
                              self.gbest)
        self.gbest_cost = np.where(gbest_filter,
                                   cbest_cost,
                                   self.gbest_cost)
        self.pbest = np.where(np.expand_dims(pbest_filters, axis=-1),
                              new_population,
                              self.pbest)
        self.pbest_cost = np.where(pbest_filters,
                                   new_cost,
                                   self.pbest_cost)
