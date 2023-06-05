import numpy as np
import torch
from optimizer.learnable_optimizer import Learnable_Optimizer

def scale(x,lb,ub):
    x=torch.sigmoid(x)
    x=lb+(ub-lb)*x
    return x


class L2L_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
    
    def init_population(self, problem):
        self.__fes=0
        

    def update(self,action,problem):
        x=action
        x=scale(x,problem.lb,problem.ub)
        
        # evaluate x
        if problem.optimum is None:
            y=problem.eval(x)
        else:
            y=problem.eval(x)-problem.optimum
        
        is_done=False
        if problem.optimum is not None and y<=1e-8:
            is_done=True
        return y,0,is_done
    