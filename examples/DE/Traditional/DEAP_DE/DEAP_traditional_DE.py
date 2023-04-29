import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import cma
from deap import benchmarks

from L2OBench.Optimizer.basic_optimizer import basic_optimizer

# 使用deap初始化一个DE算法用于优化黑盒问题
class DEAP_Optimizer_baseDE(basic_optimizer):
    def __init__(self,problem,config):
        self.config = config
        self.problem = problem
        self.toolbox = base.Toolbox()

        creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.Fitnessmin)

        self.toolbox.register("evaluate", self.problem.eval)
        # self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.problem.config["low"], up=self.problem.config["up"], eta=20.0)
        # self.toolbox.register("mutate", tools.mutPolynomialBounded, low=self.problem.config["low"], up=self.problem.config["up"], eta=20.0, indpb=1.0/len(self.problem.config["low"]))
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # self.toolbox.register("map", map)

        self.toolbox.register("attr_float", np.random.uniform, self.config.lb, self.config.ub)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.config.dim)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.pop = self.toolbox.population(n=self.config.population_size)

        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)



    def update(self):

        self.fitnesses = self.toolbox.map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, self.fitnesses):
            ind.fitness.values = (fit,)  # transform the *fit* into a tuple like (9.9,) to match the *base.Fitness* since the return of evaluation on individual is a scalar.

        record = self.stats.compile(self.pop)
        print(record)

        for g in range(self.config.iter):
            for k,agent in enumerate(self.pop):
                a,b,c = self.toolbox.select(self.pop,3)
                y = self.toolbox.clone(agent)
                index = np.random.randint(0,self.config.dim,1)[0]
                for i, value in enumerate(agent):
                    if np.random.rand() < self.config.cr or i == index:
                        y[i] = a[i] + self.config.f*(b[i]-c[i])
                y.fitness.values = (self.toolbox.evaluate(y),)  # transform the fitness into a tuple like (9.9,) to match the *base.Fitness* since the return of evaluation on individual is a scalar.
                if y.fitness.values[0] < agent.fitness.values[0]:
                    self.pop[k] = y
            record = self.stats.compile(self.pop)
            self.hof.update(self.pop)
            print(record)
        return self.pop

    def get_best(self):
        print("best fitness: ",self.hof[0],self.hof[0].fitness.values[0])
        return self.hof[0]

    def train(self):
        self.evolve()
        self.get_best()

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class DEAP_Optimizer_DE(basic_optimizer):
    def __init__(self,problem,config):
        self.config = config
        self.problem = problem
        self.toolbox = base.Toolbox()
        self.creator = creator

        self.creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list, fitness=creator.Fitnessmin)

        self.toolbox.register("evaluate", self.problem.eval)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # self.toolbox.register("map", map)

        self.toolbox.register("attr_float", np.random.uniform, self.config.lb, self.config.ub)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float,
                              n=self.config.dim)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.pop = self.toolbox.population(n=self.config.population_size)

        def mutDE(individual, F, CR):
            size = len(individual)
            a, b, c = self.toolbox.select(self.pop, 3)
            index = np.random.randint(0, size)
            for i, value in enumerate(individual):
                if np.random.rand() < CR or i == index:
                    individual[i] = a[i] + F*(b[i]-c[i])
            return individual

        def cxDE(ind1, ind2, CR):
            size = len(ind1)
            index = np.random.randint(0, size)
            for i, value in enumerate(ind1):
                if np.random.rand() < CR or i == index:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2

        def cxDEBlend(ind1, ind2, CR):
            size = len(ind1)
            index = np.random.randint(0, size)
            for i, value in enumerate(ind1):
                if np.random.rand() < CR or i == index:
                    ind1[i] = ind1[i] + np.random.rand()*(ind2[i]-ind1[i])
            return ind1, ind2

        self.toolbox.register("mutate", mutDE, F=self.config.f, CR=self.config.cr)
        self.toolbox.register("mate", cxDEBlend, CR=self.config.cr)

        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)



    def update(self):

        self.fitnesses = self.toolbox.map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, self.fitnesses):
            ind.fitness.values = (fit,)

        record = self.stats.compile(self.pop)
        print(record)

        for g in range(self.config.iter):
            childen = []
            for agent in self.pop:
                child1 = self.toolbox.clone(agent)
                child2 = self.toolbox.clone(agent)
                child1 = self.toolbox.mutate(child1, self.config.f, self.config.cr)
                child1, child2 = self.toolbox.mate(child1, child2)
                if self.toolbox.evaluate(child1) < self.toolbox.evaluate(child2):
                    childen.append(child1)
                else:
                    childen.append(child2)
            fitnesses = self.toolbox.map(self.toolbox.evaluate, childen)
            for (i,ind), fit in zip(enumerate(childen), fitnesses):
                ind.fitness.values = (fit,)
                if ind.fitness.values[0] < self.pop[i].fitness.values[0]:
                    self.pop[i] = ind
            self.hof.update(self.pop)
            record = self.stats.compile(self.pop)
            print(record)
        return self.pop

    def get_best(self):
        print("best fitness: ",self.hof[0],self.hof[0].fitness.values[0])
        return self.hof[0]

    def train(self):
        self.evolve()
        self.get_best()

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

