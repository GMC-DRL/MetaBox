from optimizer.learnable_optimizer import Learnable_Optimizer
from optimizer.operators import boundary_control as BC
import numpy as np


# a function for optimizer to calculate reward
def cal_reward(f_old, f_new, d_old, d_new):
    if f_new < f_old and d_new > d_old:
        return 2
    if f_new < f_old and d_new <= d_old:
        return 1
    if f_new >= f_old and d_new > d_old:
        return 0
    if f_new >= f_old and d_new <= d_old:
        return -2


class QLPSO_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        # define hyperparameters that backbone optimizer needs
        config.NP = 30
        config.C = 1.49618
        config.W = 0.729844
        self.__config = config

        self.__C = config.C
        self.__W = config.W
        self.__NP = config.NP
        self.__dim = config.dim
        self.__maxFEs = config.maxFEs
        self.__solution_pointer = 0  # indicate which solution receive the action
        self.__population = None
        self.__pbest = None
        self.__velocity = None
        self.__cost = None
        self.__gbest_cost = None
        self.__diversity = None
        self.__state = None
        self.fes = None
        self.cost = None  # a list of costs that need to be maintained by EVERY backbone optimizers
        self.log_index = None
        self.log_interval = config.log_interval

    def __cal_diversity(self):
        return np.mean(np.sqrt(np.sum(np.square(self.__population - np.mean(self.__population,0)),1)))

    def __cal_velocity(self, action):
        i = self.__solution_pointer
        x = self.__population[i]
        v = self.__velocity[i]
        k = 0
        # calculate neighbour indexes
        if action == 0:
            k=4
        if action == 1:
            k=8
        if action == 2:
            k=16
        if action == 3:
            k=30

        nbest = None
        nbest_cost = np.inf
        for j in range(-k//2,k//2+1):
            if self.__cost[(i+j) % self.__population.shape[0]] < nbest_cost:
                nbest_cost = self.__cost[(i+j) % self.__population.shape[0]]
                nbest = self.__population[(i+j) % self.__population.shape[0]]
        return self.__W * v \
               + self.__C * np.random.rand() * (nbest - x) \
               + self.__C * np.random.rand() * (self.__pbest[i] - x)

    def init_population(self, problem):
        self.__population = np.random.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb  # [lb, ub]
        self.__pbest = self.__population.copy()
        self.__velocity = np.zeros(shape=(self.__NP, self.__dim))
        self.__diversity = self.__cal_diversity()
        if problem.optimum is None:
            self.__cost = problem.eval(self.__population)
        else:
            self.__cost = problem.eval(self.__population) - problem.optimum
        self.__gbest_cost = self.__cost.min().copy()
        self.fes = self.__NP
        self.log_index = 1
        self.cost = [self.__gbest_cost]
        self.__state = np.random.randint(low=0, high=4, size=self.__NP)
        return self.__state[self.__solution_pointer]

    def update(self, action, problem):
        self.__velocity[self.__solution_pointer] = self.__cal_velocity(action)
        self.__population[self.__solution_pointer] += self.__velocity[self.__solution_pointer]
        # Boundary control
        self.__population[self.__solution_pointer] = BC.clipping(self.__population[self.__solution_pointer], problem.lb, problem.ub)
        # calculate reward's data
        f_old = self.__cost[self.__solution_pointer]
        if problem.optimum is None:
            f_new = problem.eval(self.__population[self.__solution_pointer])
        else:
            f_new = problem.eval(self.__population[self.__solution_pointer]) - problem.optimum
        self.fes += 1
        d_old = self.__diversity
        d_new = self.__cal_diversity()
        reward = cal_reward(f_old,f_new,d_old,d_new)
        # update population information
        self.__cost[self.__solution_pointer] = f_new
        self.__diversity = d_new
        self.__gbest_cost = self.__cost.min().copy()
        if f_new < f_old:
            self.__pbest[self.__solution_pointer] = self.__population[self.__solution_pointer] #record pbest position
        self.__state[self.__solution_pointer] = action
        self.__solution_pointer = (self.__solution_pointer + 1) % self.__NP

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__gbest_cost)
        # if an episode should be terminated
        if problem.optimum is None:
            is_done = self.fes >= self.__maxFEs
        else:
            is_done = (self.fes >= self.__maxFEs or self.__cost.min() <= 1e-8)
        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__gbest_cost
            else:
                self.cost.append(self.__gbest_cost)
        return self.__state[self.__solution_pointer], reward, is_done
