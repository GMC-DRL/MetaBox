import numpy as np
import scipy.stats as stats
from optimizer.basic_optimizer import basic_optimizer
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from tqdm import tqdm
from skopt import gp_minimize


class BayesianOptimizer(basic_optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        # self.fes = 0
        # self.log_interval = config.log_interval
        self.log_interval = 4


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
                  n_calls=200,
                  n_random_starts=5,  # the number of random initialization points
                  )   # the random seed

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
        return {'cost': cost, 'fes': fes}



        # bayes_opt v2.0
        # def black_box_function(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9):
        #     """Function with unknown internals we wish to maximize.
        #
        #     This is just serving as an example, for all intents and
        #     purposes think of the internals of this function, i.e.: the process
        #     which generates its output values, as unknown.
        #     """
        #     x = np.array([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9])
        #     return -(problem.eval(x)-problem.optimum)
        #
        # pbounds = {'x'+str(index): (-5,5) for index in range(10)}
        # optimizer = BayesianOptimization(
        #     f=black_box_function,
        #     pbounds=pbounds,
        #     allow_duplicate_points=True,
        # )
        #
        #
        # utility = UtilityFunction(kind="ucb")
        #
        # optimizer.maximize(
        #     init_points=100,
        #     n_iter=100,
        #     acquisition_function=utility
        # )
        # # print(optimizer.res)
        # best=None
        # cost=[]
        # for i, res in enumerate(optimizer.res):
        #     if best is None:
        #         best=-res['target']
        #         cost.append(best)
        #     elif -res['target']<best:
        #         best=-res['target']
        #     if ((i+1) % self.log_interval) == 0:
        #         cost.append(best)
        # return {'cost': cost, 'fes': 200}

        # bayes_opt v1.0
        # def problem_eval(*args):
        #     x_eval = np.array([args[index] for index in range(problem.dim)])
        #     if problem.optimum is None:
        #         fitness = problem.eval(x)
        #     else:
        #         fitness = problem.eval(x) - problem.optimum
        #     return -1*fitness

        # # pbounds = {'x': (problem.lb, problem.ub)}
        # pbounds = {index: (problem.lb, problem.ub) for index in range(problem.dim)}
        # # print("pbounds",pbounds)
        # self.__optimizer = BayesianOptimization(
        #     # f=None,
        #     f=sphere,
        #     pbounds=pbounds,
        #     verbose=2,
        #     random_state=1,
        # )

        # # change
        # self.__optimizer.maximize(init_points=10,n_iter=50)
        # print(self.__optimizer.res)
        # torch.rand()


        # self.__utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

        # done = False
        # best = None
        # costs = []
        # self.fes = 0
        # with tqdm(range(self.__config.maxFEs),desc='bo_rollout') as pbar:
        #     while not done:
        #         x = self.__optimizer.suggest(self.__utility)
        #         # print(x)
        #         # print("x:",x)
        #         x_eval = np.array([x[index] for index in range(problem.dim)])
        #         y = problem_eval(x_eval)
        #         # print("y:",y)
        #         self.__optimizer.register(params=x, target=y)
        #         best = self.__optimizer.max
        #         cost = -best['target']
        #         print("cost:",cost)
        #         self.fes += 1
        #         pbar.update(1)
        #         if self.fes % self.log_interval == 0:
        #             costs.append(cost)

        #         if problem.optimum is None:
        #             done = self.fes >= self.__config.maxFEs
        #         else:
        #             done = self.fes >= self.__config.maxFEs or cost <= 1e-8

        #         if done:
        #             break

        

