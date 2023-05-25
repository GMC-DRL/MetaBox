import numpy as np
from deap import base
from deap import creator
from deap import tools
from optimizer.basic_optimizer import basic_optimizer


class DEAP_DE(basic_optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.NP = 50
        config.F = 0.5
        config.Cr = 0.5

        self.__config = config
        self.__toolbox = base.Toolbox()
        creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.Fitnessmin)
        self.__toolbox.register("select", tools.selTournament, tournsize=3)
        self.log_interval = 400

    def run_episode(self, problem):

        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness,   # return a tuple

        self.__toolbox.register("evaluate", problem_eval)
        self.__toolbox.register("attr_float", np.random.uniform, problem.lb, problem.ub)
        self.__toolbox.register("individual", tools.initRepeat, creator.Individual, self.__toolbox.attr_float, n=self.__config.dim)
        self.__toolbox.register("population", tools.initRepeat, list, self.__toolbox.individual)

        hof = tools.HallOfFame(1)

        pop = self.__toolbox.population(n=self.__config.NP)
        fitnesses = self.__toolbox.map(self.__toolbox.evaluate, pop)
        fes = self.__config.NP
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        log_index = 1
        cost = [hof[0].fitness.values[0]]

        done = False
        while not done:
            for k, agent in enumerate(pop):
                a, b, c = self.__toolbox.select(pop, 3)
                y = self.__toolbox.clone(agent)
                # mutate & crossover
                index = np.random.randint(0, self.__config.dim, 1)[0]
                for i, value in enumerate(agent):
                    if np.random.rand() < self.__config.Cr or i == index:
                        y[i] = a[i] + self.__config.F * (b[i] - c[i])
                        # BC
                        y[i] = max(problem.lb, min(y[i], problem.ub))
                y.fitness.values = self.__toolbox.evaluate(y)
                # selection
                if y.fitness.values[0] < agent.fitness.values[0]:
                    pop[k] = y

                hof.update(pop)
                fes += 1

                if fes >= log_index * self.log_interval:
                    log_index += 1
                    cost.append(hof[0].fitness.values[0])

                if problem.optimum is None:
                    done = fes >= self.__config.maxFEs
                else:
                    done = fes >= self.__config.maxFEs or hof[0].fitness.values[0] <= 1e-8

                if done:
                    break
        return {'cost': cost, 'fes': fes}
