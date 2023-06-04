import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import cma
from optimizer.basic_optimizer import Basic_Optimizer


class DEAP_CMAES(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.NP = 50
        self.__config = config
        self.__toolbox = base.Toolbox()
        self.__creator = creator
        self.__algorithm = algorithms
        self.__creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        self.__creator.create("Individual", list, fitness=creator.Fitnessmin)
        self.log_interval = config.log_interval

    def run_episode(self, problem):

        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness,   # return a tuple

        self.__toolbox.register("evaluate", problem_eval)
        strategy = cma.Strategy(centroid=[problem.ub] * self.__config.dim, sigma=0.5, lambda_=self.__config.NP)
        self.__toolbox.register("generate", strategy.generate, creator.Individual)
        self.__toolbox.register("update", strategy.update)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        fes = 0
        log_index = 0
        cost = []
        while True:
            _, logbook = self.__algorithm.eaGenerateUpdate(self.__toolbox, ngen=1, stats=stats, halloffame=hof, verbose=False)
            fes += len(logbook) * self.__config.NP
            if fes >= log_index * self.log_interval:
                log_index += 1
                cost.append(hof[0].fitness.values[0])
            if problem.optimum is None:
                done = fes >= self.__config.maxFEs
            else:
                done = fes >= self.__config.maxFEs or hof[0].fitness.values[0] <= 1e-8
            if done:
                if len(cost) >= self.__config.n_logpoint + 1:
                    cost[-1] = hof[0].fitness.values[0]
                else:
                    cost.append(hof[0].fitness.values[0])
                break
        return {'cost': cost, 'fes': fes}
