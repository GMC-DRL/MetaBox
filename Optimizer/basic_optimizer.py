
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
from deap import gp
from deap import benchmarks


# 使用deap初始化一个DE算法用于优化黑盒问题
class DEAP_Optimizer_DE():
    def __init__(self,problem,config):
        self.config = config
        self.problem = problem
        self.toolbox = base.Toolbox()

        creator.create("Fitnessmin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.Fitnessmin)

        self.toolbox.register("evaluate", self.problem.evaluate)
        # self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.problem.config["low"], up=self.problem.config["up"], eta=20.0)
        # self.toolbox.register("mutate", tools.mutPolynomialBounded, low=self.problem.config["low"], up=self.problem.config["up"], eta=20.0, indpb=1.0/len(self.problem.config["low"]))
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # self.toolbox.register("map", map)

        self.toolbox.register("attr_float", np.random.uniform, self.config["lb"], self.config["ub"])
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.config["dim"])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.pop = self.toolbox.population(n=self.config["pop_size"])

        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)



    def evolve(self):

        self.fitnesses = self.toolbox.map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, self.fitnesses):
            ind.fitness.values = fit

        record = self.stats.compile(self.pop)
        print(record)

        for g in range(self.config["iter"]):
            for k,agent in enumerate(self.pop):
                a,b,c = self.toolbox.select(self.pop,3)
                y = self.toolbox.clone(agent)
                index = np.random.randint(0,self.config["dim"],1)[0]
                for i, value in enumerate(agent):
                    if np.random.rand() < self.config["cr"] or i == index:
                        y[i] = a[i] + self.config["f"]*(b[i]-c[i])
                y.fitness.values = self.toolbox.evaluate(y)
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






class deap_optimizer():
    def __init__(self):
        pass

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch

class learnable_Agent():
    def __init__(self,superparam,vector_env):
        self.superparam = superparam
        self.env = vector_env
        # self.net = xxx
        pass

    def get_feature(self):

        pass


    def inference(self,need_gd):
        # get aciton/fitness
        pass


    def cal_loss(self):
        pass

    def update_env(self):
        # self.env.step()
        pass

    def learning(self):
        # cal_loss
        # update nets
        pass


    def memory(self):
        # record some info
        pass
