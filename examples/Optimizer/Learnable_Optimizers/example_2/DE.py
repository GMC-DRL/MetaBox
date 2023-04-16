"""
    An example of controlling operators individual-wisely in DE,
    while action is presented as a np.ndarray.
"""

import numpy as np
from L2OBench.Optimizer import Learnable_Optimizer, reflection, rand_1, rand_2, rand_to_best_2, cur_to_best_1


class DE(Learnable_Optimizer):
    def __init__(self, dim, lower_bound, upper_bound, population_size, maxFEs, F=0.9, Cr=0.5):
        super().__init__(dim, lower_bound, upper_bound, population_size, maxFEs)
        self.F = F
        self.Cr = Cr

    def crossover(self, x, Cr):
        """
        :param Cr: An array of crossover rate of shape[NP].
        :param x: The mutated population before crossover.
        :return: Population after crossover.
        """
        r = np.random.rand(self.NP, self.dim)
        r[np.arange(self.NP), np.random.randint(low=0, high=self.dim, size=self.NP)] = 0.  # 对每个个体的dim个随机数，随机地取其中一个置0
        y = np.where(r <= Cr.reshape(-1, 1).repeat(self.dim, axis=-1),
                     x,
                     self.population)
        return y

    def update(self, problem, action):
        """
        :param problem: Problem instance.
        :param action: A list(length=NP) whose elements stand for different mutation operators.
        """
        mutated = np.zeros((self.NP, self.dim))
        mutated[action == 0.] = rand_1(self.population[action == 0.], self.F)
        mutated[action == 1.] = rand_2(self.population[action == 1.], self.F)
        mutated[action == 2.] = rand_to_best_2(self.population[action == 2.], self.population[self.cost.argmin()], self.F)
        mutated[action == 3.] = cur_to_best_1(self.population[action == 3.], self.population[self.cost.argmin()], self.F)
        mutated = reflection(mutated, self.lb, self.ub)
        trials = self.crossover(mutated, action[1])
        # Selection
        trials_cost = problem.eval(trials) - problem.optimum
        self.fes += self.NP
        surv_filters = np.array(trials_cost <= self.cost)
        self.population = np.where(surv_filters.reshape(-1, 1), trials, self.population)
        self.cost = np.where(surv_filters, trials_cost, self.cost)
        self.gbest_cost = np.minimum(self.gbest_cost, np.min(self.cost))
