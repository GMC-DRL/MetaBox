"""
This file gives several traditional optimizer examples training with L2O benchmark.
"""
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

from L2OBench.Optimizer.basic_optimizer import basic_optimizer


class DEAP_Optimizer_PSO(basic_optimizer):
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
    def update(self):
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

class DEAP_Optimizer_CMAES(basic_optimizer):
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



    def update(self):
        self.algorithm.eaGenerateUpdate(self.toolbox, ngen=self.config.iter, stats=self.stats, halloffame=self.hof)
        return self.hof[0].fitness.values[0]


    def get_best(self):
        print("best fitness: ",self.hof[0],self.hof[0].fitness.values[0])

    def train(self):
        self.evolve()
        self.get_best()



