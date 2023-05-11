from typing import Any

from L2OBench.Problem import Basic_Problem


class Learnable_Optimizer:
    """
    Abstract super class for learnable optimizers.
    """
    def __init__(self, config):
        self.config = config

    def init_population(self,
                        problem: Basic_Problem):
        raise NotImplementedError

    def update(self,
               action: Any,
               problem: Basic_Problem,
               reward_func):
        raise NotImplementedError
