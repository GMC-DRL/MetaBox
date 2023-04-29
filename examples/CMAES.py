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

