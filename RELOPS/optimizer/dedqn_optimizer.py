from optimizer.learnable_optimizer import Learnable_Optimizer
from optimizer.operators import boundary_control as BC
from optimizer.operators import mutate as Mu
from optimizer.operators import crossover as Cross
import numpy as np


def cal_fdc(sample, fitness):
    best = np.argmin(fitness)
    distance = np.linalg.norm(sample-sample[best], axis=-1)
    cfd = np.mean((fitness - np.mean(fitness)) * (distance - np.mean(distance)))
    return cfd / (np.var(distance)*np.var(fitness) + 1e-6)


def cal_rie(fitness):
    epsilon_star = 0
    for i in range(1,len(fitness)):
        if (fitness[i] - fitness[i-1]) > epsilon_star:
            epsilon_star = fitness[i] - fitness[i-1]
    # cal rie
    hs = []
    for k in range(9):
        epsilon = 0
        if k < 8:
            epsilon = epsilon_star / (2 ** k)
        s = []
        for i in range(len(fitness) - 1):
            if (fitness[i+1] - fitness[i]) < -epsilon:
                s.append(-1)
            elif (fitness[i+1] - fitness[i]) > epsilon:
                s.append(1)
            else:
                s.append(0)
        freq = np.zeros(6)
        for i in range(len(fitness) - 2):
            if s[i] == -1 and s[i+1] == 0:
                freq[0] += 1
            elif s[i] == -1 and s[i+1] == 1:
                freq[1] += 1
            elif s[i] == 0 and s[i+1] == 1:
                freq[2] += 1
            elif s[i] == 0 and s[i+1] == -1:
                freq[3] += 1
            elif s[i] == 1 and s[i+1] == -1:
                freq[4] += 1
            else:
                freq[5] += 1
        freq[freq == 0] = len(fitness)
        freq /= len(fitness)
        entropy = -np.sum(freq * np.log(freq) / np.log(6))
        hs.append(entropy)
    return max(hs)


def cal_acf(fitness):
    avg_f = np.mean(fitness)
    a = np.sum((fitness - avg_f) ** 2) + 1e-6
    acf = 0
    for i in  range(len(fitness) - 1):
        acf += (fitness[i] - avg_f) * (fitness[i + 1] - avg_f)
    acf /= a
    return acf


def cal_nop(sample, fitness):
    best = np.argmin(fitness)
    distance = np.linalg.norm(sample - sample[best], axis=-1)
    data = np.stack([fitness, distance], axis=0)
    data = data.T
    data = data[np.argsort(data[:, 1]), :]
    fitness_sorted = data[:,0]
    r = 0
    for i in range(len(fitness) - 1):
        if fitness_sorted[i+1] < fitness_sorted[i]:
            r += 1
    return r / len(fitness)


def random_walk_sampling(population, dim, steps):
    pmin = np.min(population, axis=0)
    pmax = np.max(population, axis=0)
    walks = []
    start_point = np.random.rand(dim)
    walks.append(start_point.tolist())
    for _ in range(steps - 1):
        move = np.random.rand(dim)
        start_point = (start_point + move) % 1
        walks.append(start_point.tolist())
    return pmin + (pmax - pmin) * np.array(walks)


def cal_reward(survival, pointer):
    reward = 0
    for i in range(len(survival)):
        if i == pointer:
            if survival[i] == 1:
                reward += 1
        else:
            reward += 1/survival[i]
    return reward / len(survival)


class DEDQN_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.NP = 100
        config.F = 0.5
        config.Cr = 0.5
        config.rwsteps = config.NP
        self.__config = config

        self.__dim = config.dim
        self.__NP = config.NP
        self.__F = config.F
        self.__Cr = config.Cr
        self.__maxFEs = config.maxFEs
        self.__rwsteps = config.rwsteps
        self.__solution_pointer = 0 #indicate which solution receive the action
        self.__population = None
        self.__cost = None
        self.__gbest = None
        self.__gbest_cost = None
        self.__state = None
        self.__survival = None
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __cal_feature(self, problem):
        samples = random_walk_sampling(self.__population, self.__dim, self.__rwsteps)
        if problem.optimum is None:
            samples_cost = problem.eval(self.__population)
        else:
            samples_cost = problem.eval(self.__population) - problem.optimum
        # calculate fdc
        fdc = cal_fdc(samples, samples_cost)
        rie = cal_rie(samples_cost)
        acf = cal_acf(samples_cost)
        nop = cal_nop(samples, samples_cost)
        self.fes += self.__rwsteps
        return np.array([fdc, rie, acf, nop])

    def init_population(self, problem):
        self.__population = np.random.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb  # [lb, ub]
        self.__survival = np.ones(self.__population.shape[0])
        if problem.optimum is None:
            self.__cost = problem.eval(self.__population)
        else:
            self.__cost = problem.eval(self.__population) - problem.optimum
        self.__gbest = self.__population[self.__cost.argmin()]
        self.__gbest_cost = self.__cost.min()
        self.fes = self.__NP
        self.log_index = 1
        self.cost = [self.__gbest_cost]
        self.__state = self.__cal_feature(problem)
        return self.__state

    def update(self, action, problem):
        # mutate first
        if action == 0:
            u = Mu.rand_1_single(self.__population, self.__F, self.__solution_pointer)
        elif action == 1:
            u = Mu.cur_to_rand_1_single(self.__population, self.__F, self.__solution_pointer)
        elif action == 2:
            u = Mu.best_2_single(self.__population, self.__gbest, self.__F, self.__solution_pointer)
        else:
            raise ValueError(f'action error: {action}')
        # BC
        u = BC.clipping(u, problem.lb, problem.ub)
        # then crossover
        u = Cross.binomial(self.__population[self.__solution_pointer], u, self.__Cr)
        # select from u and x
        if problem.optimum is None:
            u_cost = problem.eval(u)
        else:
            u_cost = problem.eval(u) - problem.optimum
        self.fes += self.__NP
        if u_cost <= self.__cost[self.__solution_pointer]:
            self.__population[self.__solution_pointer] = u
            self.__cost[self.__solution_pointer] = u_cost
            self.__survival[self.__solution_pointer] = 1
            if u_cost < self.__gbest_cost:
                self.__gbest = u
                self.__gbest_cost = u_cost
        else:
            self.__survival[self.__solution_pointer] += 1
        self.__state = self.__cal_feature(problem)

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__gbest_cost)

        reward = cal_reward(self.__survival, self.__solution_pointer)

        if problem.optimum is None:
            is_done = self.fes >= self.__maxFEs
        else:
            is_done = self.fes >= self.__maxFEs or self.__cost.min() <= 1e-8
        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__gbest_cost
            else:
                self.cost.append(self.__gbest_cost)
        self.__solution_pointer = (self.__solution_pointer + 1) % self.__population.shape[0]
        return self.__state, reward, is_done
