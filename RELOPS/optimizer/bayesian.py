import numpy as np
import scipy.stats as stats
from optimizer.basic_optimizer import basic_optimizer
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
class BayesianOptimizer(basic_optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        # self.fes = 0
        self.log_interval = 4



    def run_episode(self, problem):

        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return -1*fitness #bo 默认是maximize

        # pbounds = {'x': (problem.lb, problem.ub)}
        pbounds = {index: (problem.lb, problem.ub) for index in range(problem.dim)}
        # print("pbounds",pbounds)
        self.__optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        self.__utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

        done = False
        best = None
        costs = []
        self.fes = 0

        while not done:
            x = self.__optimizer.suggest(self.__utility)
            # print("x:",x)
            x_eval = np.array([x[index] for index in range(problem.dim)])
            y = problem_eval(x_eval)
            # print("y:",y)
            self.__optimizer.register(params=x, target=y)
            best = self.__optimizer.max
            cost = best['target']
            print("cost:",-cost)
            self.fes += 1

            if self.fes % self.log_interval == 0:
                costs.append(-cost)

            if problem.optimum is None:
                done = self.fes >= self.__config.maxFEs
            else:
                done = self.fes >= self.__config.maxFEs or cost >= -1*1e-8

            if done:
                break

        return {'cost': costs, 'fes': self.fes}

