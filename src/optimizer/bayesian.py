from optimizer.basic_optimizer import Basic_Optimizer
from skopt import gp_minimize


class BayesianOptimizer(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        # self.log_interval = config.log_interval
        self.log_interval = 2

    def run_episode(self, problem):
        # skopt v1.0
        cost = []
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
                          n_calls=self.__config.bo_maxFEs,         # the number of evaluations of f
                        #   n_calls=100,
                          n_random_starts=5,  # the number of random initialization points
                          )   # the random seed
        for i in range(len(res.func_vals)):
            if best is None:
                best = res.func_vals[i]
                cost.append(best)
            elif res.func_vals[i] < best:
                best = res.func_vals[i]
            if ((i+1) % self.log_interval) == 0:
                cost.append(best)
            fes += 1
            if best <= 1e-8:
                break
        if len(cost) >= self.__config.n_logpoint + 1:
            cost[-1] = best
        else:
            cost.append(best)
        return {'cost': cost, 'fes': fes}
