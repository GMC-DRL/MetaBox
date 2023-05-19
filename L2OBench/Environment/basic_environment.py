from typing import Any
from problem.basic_problem import Basic_Problem
from optimizer.learnable_optimizer import Learnable_Optimizer


class PBO_Env:
    """
    Env with problem and optimizer.
    """
    def __init__(self,
                 problem: Basic_Problem,
                 optimizer: Learnable_Optimizer,
                 ):
        self.problem = problem
        self.optimizer = optimizer

    def reset(self):
        return self.optimizer.init_population(self.problem)

    def step(self, action: Any):
        return self.optimizer.update(action, self.problem)
