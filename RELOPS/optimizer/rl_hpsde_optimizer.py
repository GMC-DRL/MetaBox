import numpy as np
import scipy.stats as stats
from optimizer.learnable_optimizer import Learnable_Optimizer
from optimizer.operators import clipping, binomial, cur_to_rand_1, cur_to_best_1


class Population:
    def __init__(self, config):
        self.Nmax = config.NP_max                      # the upperbound of population size
        self.Nmin = config.NP_min                      # the lowerbound of population size
        self.NP = self.Nmax                            # the population size
        self.dim = config.dim                          # the dimension of individuals
        self.group = None                              # the population
        self.cost = None                               # the cost of individuals
        self.gbest = None                              # the global best cost
        self.gbest_solution = None                     # the individual with global best cost
        self.F = config.F
        self.Cr = config.Cr
        self.MF = np.ones(int(self.dim * config.Hm)) * self.F       # the set of step length of DE
        self.MCr = np.ones(int(self.dim * config.Hm)) * self.Cr     # the set of crossover rate of DE
        self.k = 0                                     # the index of updating element in MF and MCr

        self.init_best = None

    # generate an initialized population with size(default self population size)
    def initialize_group(self, lb, ub, size=-1):
        if size < 0:
            size = self.NP
        self.group = np.random.rand(size, self.dim) * (ub - lb) + lb

    # initialize cost
    def initialize_costs(self, problem):
        if problem.optimum is None:
            self.cost = problem.eval(self.group)
        else:
            self.cost = problem.eval(self.group) - problem.optimum
        self.gbest = np.min(self.cost)
        self.gbest_solution = self.group[np.argmin(self.cost)]
        self.init_best = np.min(self.cost)

    # sort former 'size' population in respect to cost
    def sort(self, size, reverse=False):
        # new index after sorting
        r = -1 if reverse else 1
        ind = np.concatenate((np.argsort(r * self.cost[:size]), np.arange(self.NP)[size:]))
        self.group = self.group[ind]
        self.cost = self.cost[ind]
        self.gbest = np.min(self.cost)
        self.gbest_solution = self.group[np.argmin(self.cost)]

    # randomly choose step length nad crossover rate from MF and MCr
    def choose_F_Cr(self, F_dist):
        # generate Cr
        gs = self.NP
        ind_r = np.random.randint(0, self.MF.shape[0], size=gs)  # index
        Cr = np.minimum(1, np.maximum(0, np.random.normal(loc=self.MCr[ind_r], scale=0.1, size=gs)))  # 0~1
        # generate F
        locs = self.MF[ind_r]
        F = None
        if F_dist == 'cauchy':
            F = stats.cauchy.rvs(loc=locs, scale=0.1, size=gs)
        elif F_dist == 'levy':
            F = stats.levy.rvs(loc=locs, scale=0.1, size=gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * locs[err] - F[err]
        F = np.minimum(1, F)
        return Cr, F

    # calculate wL mean
    def mean_wL(self, df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # update MF and MCr, join new value into the set if there are some successful changes or set it to initial value
    def update_M_F_Cr(self, SF, SCr, df):
        if SF.shape[0] > 0:
            mean_wL = self.mean_wL(df, SF)
            self.MF[self.k] = mean_wL
            mean_wL = self.mean_wL(df, SCr)
            self.MCr[self.k] = mean_wL
            self.k = (self.k + 1) % self.MF.shape[0]
        else:
            self.MF[self.k] = 0.5
            self.MCr[self.k] = 0.5

    # linearly population size reduction
    def LPSR(self, fes, maxFEs):
        self.sort(self.NP)
        N = max(int(self.Nmax + np.round(self.Nmin - self.Nmax) * fes / maxFEs), 1)
        if N < self.NP:
            self.NP = N
            self.group = self.group[:N]
            self.cost = self.cost[:N]
            self.gbest = np.min(self.cost)
            self.gbest_solution = self.group[np.argmin(self.cost)]


class RL_HPSDE_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.F = 0.5
        config.Cr = 0.5
        config.NP_max = 18 * config.dim
        config.NP_min = 4
        config.Hm = 0.5
        config.rw_steps = 200
        config.step_size = 10
        self.__config = config

        self.__population = None
        self.__dim = config.dim
        self.__rw_steps = config.rw_steps
        self.__step_size = config.step_size
        self.__maxFEs = config.maxFEs
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = 400

    def init_population(self, problem):
        self.__population = Population(self.__config)
        self.__population.initialize_group(lb=problem.lb, ub=problem.ub)
        self.__population.initialize_costs(problem)
        self.__population.sort(self.__population.NP)
        self.fes = self.__population.NP
        self.log_index = 1
        self.cost = [self.__population.gbest]
        return self.__get_state(problem)

    def __simple_random_walk(self, lb, ub):
        samples = np.zeros((self.__rw_steps + 1, self.__dim))
        samples[0] = lb + np.random.random(self.__dim) * (ub - lb)
        for step in range(1, self.__rw_steps + 1):
            samples[step] = samples[step - 1] + np.random.uniform(low=-self.__step_size,
                                                                  high=self.__step_size,
                                                                  size=self.__dim)
            while True:
                outter_index = np.where(np.any([samples[step] > ub, samples[step] < lb], axis=0))[0]
                if outter_index.shape[0] > 0:
                    samples[step][outter_index] = samples[step - 1][outter_index] + np.random.uniform(low=-self.__step_size,
                                                                                                      high=self.__step_size,
                                                                                                      size=outter_index.shape[0])
                else:
                    break
        return samples

    def __progressive_random_walk(self, lb, ub):
        samples = np.zeros((self.__rw_steps + 1, self.__dim))
        startingZone = np.random.rand(self.__dim)
        startingZone[startingZone < 0.5] = -1
        startingZone[startingZone >= 0.5] = 1
        r = np.random.rand(self.__dim) * (ub - lb) / 2
        samples[0] = (ub + lb) / 2 + startingZone * r
        rD = np.random.choice(self.__dim, 1)
        if startingZone[rD] == -1:
            samples[0][rD] = lb
        else:
            samples[0][rD] = ub
        for step in range(1, self.__rw_steps + 1):
            samples[step] = samples[step - 1] + np.random.rand(self.__dim) * (-self.__step_size) * startingZone
            cro_ub = samples[step] > ub
            cro_lb = samples[step] < lb
            samples[step][cro_ub] = 2 * ub - samples[step][cro_ub]
            samples[step][cro_lb] = 2 * lb - samples[step][cro_lb]
            startingZone[np.any([cro_ub, cro_lb], axis=0)] *= -1
        return samples

    # Dynamic Fitness Distance Correlation
    def __DFDC(self, sample, cost):
        sample = sample[1:]
        cost = cost[1:]
        gbest_solution = self.__population.gbest_solution
        dist = np.linalg.norm(sample - gbest_solution, ord=2, axis=-1)
        r = np.mean((cost - cost.mean()) * (dist - dist.mean())) / (cost.std() * dist.std())
        if 0.15 < r <= 1:
            return True  # easy
        elif -1 <= r < 0.15:
            return False  # difficult
        else:
            raise ValueError(f"DFDC error: {r}, {cost.std()}, {dist.std()}")

    # Dynamic Ruggedness of Information Entropy
    def __DRIE(self, cost):
        diff = cost[1:] - cost[:self.__rw_steps]
        e_star = np.max(np.abs(diff))
        r = None
        for i, scale in enumerate([0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]):
            symbol = (diff < (-scale * e_star)) * (-1) + ((scale * e_star) < diff) * 1
            prob = np.zeros(6)
            for j in range(self.__rw_steps - 1):
                if symbol[j] == -1 and symbol[j + 1] == 0:
                    prob[0] += 1
                elif symbol[j] == -1 and symbol[j + 1] == 1:
                    prob[1] += 1
                elif symbol[j] == 0 and symbol[j + 1] == -1:
                    prob[2] += 1
                elif symbol[j] == 0 and symbol[j + 1] == 1:
                    prob[3] += 1
                elif symbol[j] == 1 and symbol[j + 1] == -1:
                    prob[4] += 1
                elif symbol[j] == 1 and symbol[j + 1] == 0:
                    prob[5] += 1
            prob /= self.__rw_steps
            prob[prob < 1e-15] = 1e-15
            if i == 0:
                r = -np.sum(prob * np.log(prob) / np.log(6))
            else:
                r = max(r, -np.sum(prob * np.log(prob) / np.log(6)))
        if 0.5 <= r <= 1:
            return True  # easy
        elif 0 <= r < 0.5:
            return False  # difficult
        else:
            raise ValueError(f"DRIE error: {r}")

    def __get_state(self, problem):
        # random walk
        # sample = self.simple_random_walk(lb=problem.lb, ub=problem.ub)
        sample = self.__progressive_random_walk(lb=problem.lb, ub=problem.ub)
        if problem.optimum is None:
            sample_cost = problem.eval(sample)
        else:
            sample_cost = problem.eval(sample) - problem.optimum
        self.fes += sample.shape[0]
        # get state
        return self.__DFDC(sample, sample_cost) * 1 + self.__DRIE(sample_cost) * 2

    def update(self, action, problem):
        population = self.__population
        NP, dim = population.NP, population.dim
        # Mu
        if action == 0:
            Cr, F = population.choose_F_Cr("cauchy")
            v = cur_to_rand_1(population.group, F)
        elif action == 1:
            Cr, F = population.choose_F_Cr("cauchy")
            v = cur_to_best_1(population.group, population.gbest_solution, F)
        elif action == 2:
            Cr, F = population.choose_F_Cr("levy")
            v = cur_to_rand_1(population.group, F)
        elif action == 3:
            Cr, F = population.choose_F_Cr("levy")
            v = cur_to_best_1(population.group, population.gbest_solution, F)
        else:
            raise ValueError(f'action error: {action}')
        # BC
        v = clipping(v, problem.lb, problem.ub)
        # Cr
        u = binomial(population.group, v, Cr)
        # Selection
        if problem.optimum is None:
            ncost = problem.eval(u)
        else:
            ncost = problem.eval(u) - problem.optimum
        self.fes += NP
        optim = np.where(ncost < population.cost)[0]
        SF = F[optim]
        SCr = Cr[optim]
        df = np.maximum(0, population.cost - ncost)[optim]
        population.update_M_F_Cr(SF, SCr, df)

        population.group[optim] = u[optim]
        population.cost = np.minimum(population.cost, ncost)
        # update gbest
        if population.cost.min() < population.gbest:
            population.gbest = population.cost.min()
            population.gbest_solution = population.group[np.argmin(population.cost)]
        population.LPSR(fes=self.fes, maxFEs=self.__maxFEs)  # be sorted at the same time

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(population.gbest)

        reward = optim.shape[0] / NP
        if problem.optimum is None:
            done = self.fes >= self.__maxFEs
        else:
            done = (self.fes >= self.__maxFEs or population.gbest <= 1e-8)
        return self.__get_state(problem), reward, done
