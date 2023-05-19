from typing import Any
from problem.basic_problem import Basic_Problem


class Learnable_Optimizer:
    """
    Abstract super class for learnable optimizers.
    """
    def __init__(self, config):
        self.__config = config

    def init_population(self,
                        problem: Basic_Problem):
        raise NotImplementedError

    def update(self,
               action: Any,
               problem: Basic_Problem,
               ):
        raise NotImplementedError
