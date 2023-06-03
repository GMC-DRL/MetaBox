import numpy as np
import scipy.stats as stats
from optimizer.basic_optimizer import basic_optimizer
from tqdm import tqdm
from skopt import gp_minimize

class BayesianOptimizer(basic_optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        # self.fes = 0
        # self.log_interval = config.log_interval
        self.log_interval = 2


    def run_episode(self, problem):
        # skopt v1.0
        cost=[]
        best = None
        fes = 0
        def black_box_function(x):
            if problem.optimum is None:
                return problem.eval(x)
            else:
                return problem.eval(x)-problem.optimum

        bounds = [(-5.0, 5.0) for index in range(self.__config.dim)]
        # res = gp_minimize(problem, [(-2.0, 2.0)])
        res = gp_minimize(black_box_function,                  # the function to minimize
                  bounds,      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  # n_calls=self.__config.maxFEs,         # the number of evaluations of f
                  n_calls=100,
                  n_random_starts=5,  # the number of random initialization points
                  )   # the random seed
        # print("res:",res)
        for i in range(len(res.func_vals)):
            if best is None:
                best=res.func_vals[i]
                cost.append(best)
            elif res.func_vals[i]<best:
                best=res.func_vals[i]
            if ((i+1) % self.log_interval) == 0:
                cost.append(best)
            fes+=1
            if best<=1e-8:
                break
        # print("cost:",cost)
        # print(len(cost))
        if len(cost) >= self.__config.n_logpoint + 1:
            cost[-1] = best
        else:
            cost.append(best)
        return {'cost': cost, 'fes': fes}



