import numpy as np
from typing import Union, Any

from L2OBench.Problem import Basic_Problem


class Learnable_Optimizer:
    """
    Abstract super class for learnable optimizers.
    """
    def __init__(self,
                 dim: int,
                 lower_bound: Union[int, float, np.ndarray],
                 upper_bound: Union[int, float, np.ndarray],
                 population_size: int,
                 maxFEs: int):
        self.dim = dim
        self.lb = lower_bound
        self.ub = upper_bound
        self.NP = population_size
        self.maxFEs = maxFEs

        self.population = None
        self.cost = None
        self.gbest_cost = None  # for the need of computing rewards
        self.init_cost = None   # for the need of computing rewards
        self.fes = None

    def init_population(self, problem: Basic_Problem) -> None:
        """
        Generate population randomly and compute cost.
        """
        self.population = np.random.rand(self.NP, self.dim) * (self.ub - self.lb) + self.lb  # [lb, ub]
        self.cost = problem.eval(self.population) - problem.optimum
        self.gbest_cost = self.cost.min().copy()
        self.init_cost = self.cost.copy()
        self.fes = self.NP

    def update(self, problem: Basic_Problem, action: Any) -> None:
        raise NotImplementedError
