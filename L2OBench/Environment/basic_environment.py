from typing import Any
from L2OBench.Problem import Basic_Problem
from L2OBench.Optimizer import Learnable_Optimizer


class Env:
    """
    Treated as a pure problem for traditional algorithms without any agent.
    """
    def __init__(self, problem: Basic_Problem):
        self.problem = problem


class PBO_Env(Env):
    """
    Env with problem and optimizer.
    """
    def __init__(self,
                 problem: Basic_Problem,
                 optimizer: Learnable_Optimizer,
                 reward_func):
        Env.__init__(self, problem)
        self.optimizer = optimizer
        self.reward_func = reward_func

    def reset(self):
        return self.optimizer.init_population(self.problem)

    def step(self, action: Any):
        return self.optimizer.update(action, self.problem, self.reward_func)
