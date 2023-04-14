
import deap
import numpy as np
import torch
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


# 使用deap初始化一个DE算法用于优化黑盒问题
class DEAP_Optimizer_baseDE():
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



    def evolve(self):

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


class DEAP_Optimizer_DE():
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



    def evolve(self):

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


class DEAP_Optimizer_PSO():
    def __init__(self,problem,config):
        self.config = config
        self.problem = problem
        self.toolbox = base.Toolbox()
        self.creator = creator
        self.pmax = self.config.ub
        self.pmin = self.config.lb
        self.smax = self.config.ub - self.config.lb
        self.smin = -self.smax


        self.creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        self.creator.create("Particle", np.ndarray, fitness=creator.Fitnessmin,speed=list, smin=None, smax=None, best=None)

        def generate(size, pmin, pmax, smin, smax):
            part = creator.Particle(np.random.uniform(pmin, pmax, size))
            part.speed = np.random.uniform(smin, smax, size)
            part.smin = smin
            part.smax = smax
            return part

        def updateParticle(part, best, phi1, phi2):
            u1 = np.random.uniform(0, phi1, len(part))
            u2 = np.random.uniform(0, phi2, len(part))
            v_u1 = u1 * (part.best - part)
            v_u2 = u2 * (best - part)
            part.speed += v_u1 + v_u2
            for i, speed in enumerate(part.speed):
                if speed < part.smin:
                    part.speed[i] = part.smin
                elif speed > part.smax:
                    part.speed[i] = part.smax
            part += part.speed
            for i, value in enumerate(part):
                if value < self.pmin:
                    part[i] = self.pmin
                elif value > self.pmax:
                    part[i] = self.pmax
            return part

        self.toolbox.register("evaluate", self.problem.eval)
        self.toolbox.register("particle", generate, size=self.config.dim, pmin=self.pmin, pmax=self.pmax, smin=self.smin, smax=self.smax)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", updateParticle, phi1=self.config.phi1, phi2=self.config.phi2)

        self.pop = self.toolbox.population(n=self.config.population_size)

        self.hof = tools.HallOfFame(1)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    def evolve(self):
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = (fit,)
            ind.best = np.copy(ind)
            ind.best.fitness.values = (fit,)

        self.hof.update(self.pop)
        record = self.stats.compile(self.pop)
        print(record)

        for g in range(self.config.iter):
            for part in self.pop:
                self.toolbox.update(part, self.hof[0])
                fit = self.toolbox.evaluate(part)
                if fit < part.best.fitness.values[0]:
                    part.best = np.copy(part)
                    part.best.fitness.values = (fit,)
                part.fitness.values = (fit,)
            self.hof.update(self.pop)
            record = self.stats.compile(self.pop)
            print(record)

        return self.pop, self.hof

    def get_best(self):
        print("best fitness: ",self.hof[0],self.hof[0].fitness.values[0])
        return self.hof[0]

    def train(self):
        self.evolve()
        self.get_best()

class DEAP_Optimizer_CMAES():
    def __init__(self,problem,config):
        self.config = config
        self.problem = problem
        self.toolbox = base.Toolbox()
        self.creator = creator
        self.algorithm = algorithms
        self.creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list, fitness=creator.Fitnessmin)

        self.toolbox.register("evaluate", self.problem.eval)
        np.random.seed(128)

        self.strategy = cma.Strategy(centroid=[5]*self.config.dim, sigma=0.5, lambda_=self.config.population_size)
        self.toolbox.register("generate", self.strategy.generate, creator.Individual)
        self.toolbox.register("update", self.strategy.update)

        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)



    def evolve(self):
        self.algorithm.eaGenerateUpdate(self.toolbox, ngen=self.config.iter, stats=self.stats, halloffame=self.hof)
        return self.hof[0].fitness.values[0]


    def get_best(self):
        print("best fitness: ",self.hof[0],self.hof[0].fitness.values[0])

    def train(self):
        self.evolve()
        self.get_best()

class basic_optimizer():
    def __init__(self):
        pass

    def train(self):
        pass

    def get_best(self):
        pass

    def evolve(self):
        pass

    def get_best(self):
        pass




