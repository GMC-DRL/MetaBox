import numpy as np
import scipy.stats as stats


class Learnable_Optimizer:
    """
    Abstract super class for learnable optimizers.
    """
    def __init__(self,
                 dim,
                 lower_bound,
                 upper_bound,
                 population_size,
                 maxFEs,
                 boundary_ctrl_method='clipping'):
        self.dim = dim
        self.lb = lower_bound
        self.ub = upper_bound
        self.NP = population_size
        self.maxFEs = maxFEs
        self.boundary_ctrl_method = boundary_ctrl_method

        self.population = None
        self.cost = None
        self.gbest_cost = None  # for the need of computing rewards
        self.init_cost = None   # for the need of computing rewards
        self.fes = None

    def init_population(self, problem):
        """
        Generate population randomly and compute cost.
        """
        self.population = np.random.rand(self.NP, self.dim) * (self.ub - self.lb) + self.lb  # [lb, ub]
        self.cost = problem.eval(self.population) - problem.optimum
        self.gbest_cost = self.cost.min().copy()
        self.init_cost = self.cost.copy()
        self.fes = self.NP

    def boundary_ctrl(self, x):
        y = None
        if self.boundary_ctrl_method == 'clipping':
            y = np.clip(x, self.lb, self.ub)

        elif self.boundary_ctrl_method == 'random':
            cro_bnd = (x < self.lb) | (x > self.ub)  # cross boundary
            y = ~cro_bnd * x + cro_bnd * np.random.rand(self.NP, self.dim) * (self.ub - self.lb) + self.lb

        elif self.boundary_ctrl_method == 'reflection':
            cro_lb = x < self.lb
            cro_ub = x > self.ub
            no_cro = ~(cro_lb | cro_ub)
            y = no_cro * x + cro_lb * (2 * self.lb - x) + cro_ub * (2 * self.ub - x)

        elif self.boundary_ctrl_method == 'periodic':
            y = (x - self.ub) % (self.ub - self.lb) + self.lb

        elif self.boundary_ctrl_method == 'halving':
            cro_lb = x < self.lb
            cro_ub = x > self.ub
            no_cro = ~(cro_lb | cro_ub)
            y = no_cro * x + cro_lb * (x + self.lb) / 2 + cro_ub * (x + self.ub) / 2

        elif self.boundary_ctrl_method == 'parent':
            cro_lb = x < self.lb
            cro_ub = x > self.ub
            no_cro = ~(cro_lb | cro_ub)
            y = no_cro * x + cro_lb * (self.population + self.lb) / 2 + cro_ub * (self.population + self.ub) / 2

        return y

    def update(self, problem, action):
        pass

