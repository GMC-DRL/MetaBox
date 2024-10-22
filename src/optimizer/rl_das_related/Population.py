import scipy.stats as stats
from .utils import *
import numpy as np


class Population:
    def __init__(self, dim):
        self.Nmax = 170                         # the upperbound of population size
        self.Nmin = 30                          # the lowerbound of population size
        self.NP = self.Nmax                     # the population size
        self.NA = int(self.NP * 2.1)            # the size of archive(collection of replaced individuals)
        self.dim = dim                          # the dimension of individuals
        self.cost = np.zeros(self.NP)           # the cost of individuals
        self.cbest = 1e15                       # the best cost in current population, initialize as 1e15
        self.cbest_id = -1                      # the index of individual with the best cost
        self.gbest = 1e15                       # the global best cost
        self.gbest_solution = np.zeros(dim)     # the individual with global best cost
        self.Xmin = np.ones(dim) * -5         # the upperbound of individual value
        self.Xmax = np.ones(dim) * 5          # the lowerbound of individual value
        self.group = self.initialize_group()    # the population
        self.archive = np.array([])             # the archive(collection of replaced individuals)
        self.MF = np.ones(dim * 20) * 0.2       # the set of step length of DE
        self.MCr = np.ones(dim * 20) * 0.2      # the set of crossover rate of DE
        self.k = 0                              # the index of updating element in MF and MCr
        self.F = np.ones(self.NP) * 0.5         # the set of successful step length
        self.Cr = np.ones(self.NP) * 0.9        # the set of successful crossover rate

    # generate an initialized population with size(default self population size)
    def initialize_group(self, size=-1):
        if size < 0:
            size = self.NP
        return np.random.random((size, self.dim)) * (self.Xmax - self.Xmin) + self.Xmin

    # initialize cost
    def initialize_costs(self, problem):
        if problem.optimum is not None:
            self.cost = problem.eval(self.group) - problem.optimum
        else:
            self.cost = problem.eval(self.group)
        self.gbest = self.cbest = np.min(self.cost)
        self.cbest_id = np.argmin(self.cost)
        self.gbest_solution = self.group[self.cbest_id]

    def clear_context(self):
        self.archive = np.array([])             # the archive(collection of replaced individuals)
        self.MF = np.ones(self.dim * 20) * 0.2       # the set of step length of DE
        self.MCr = np.ones(self.dim * 20) * 0.2      # the set of crossover rate of DE
        self.k = 0                              # the index of updating element in MF and MCr
        self.F = np.ones(self.NP) * 0.5         # the set of successful step length
        self.Cr = np.ones(self.NP) * 0.9        # the set of successful crossover rate

    # sort former 'size' population in respect to cost
    def sort(self, size, reverse=False):
        # new index after sorting
        r = -1 if reverse else 1
        ind = np.concatenate((np.argsort(r * self.cost[:size]), np.arange(self.NP)[size:]))
        self.cost = self.cost[ind]
        self.cbest = np.min(self.cost)
        self.cbest_id = np.argmin(self.cost)
        self.group = self.group[ind]
        self.F = self.F[ind]
        self.Cr = self.Cr[ind]

    # calculate new population size with non-linear population size reduction
    def cal_NP_next_gen(self, FEs, MaxFEs):
        NP = np.round(self.Nmax + (self.Nmin - self.Nmax) * np.power(FEs/MaxFEs, 1-FEs/MaxFEs))
        return NP

    # slice the population and its cost, crossover rate, etc
    def slice(self, size):
        self.NP = size
        self.group = self.group[:size]
        self.cost = self.cost[:size]
        self.F = self.F[:size]
        self.Cr = self.Cr[:size]
        if self.cbest_id >= size:
            self.cbest_id = np.argmin(self.cost)
            self.cbest = np.min(self.cost)

    # reduce population in JDE way
    def reduction(self, bNP):
        self.group = np.concatenate((self.group[:bNP//2], self.group[bNP:]), 0)
        self.F = np.concatenate((self.F[:bNP // 2], self.F[bNP:]), 0)
        self.Cr = np.concatenate((self.Cr[:bNP // 2], self.Cr[bNP:]), 0)
        self.cost = np.concatenate((self.cost[:bNP // 2], self.cost[bNP:]), 0)
        self.NP = bNP // 2 + 10

    # calculate wL mean
    def mean_wL(self, df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # randomly choose step length nad crossover rate from MF and MCr
    def choose_F_Cr(self):
        # generate Cr can be done simutaneously
        gs = self.NP
        ind_r = np.random.randint(0, self.MF.shape[0], size=gs)  # index
        C_r = np.minimum(1, np.maximum(0, np.random.normal(loc=self.MCr[ind_r], scale=0.1, size=gs)))
        # as for F, need to generate 1 by 1
        cauchy_locs = self.MF[ind_r]
        F = stats.cauchy.rvs(loc=cauchy_locs, scale=0.1, size=gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        # F = []
        # for i in range(gs):
        #     while True:
        #         f = stats.cauchy.rvs(loc=cauchy_locs[i], scale=0.1)
        #         if f >= 0:
        #             F.append(f)
        #             break
        return C_r, np.minimum(1, F)

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

    # non-linearly reduce population size and update it into new population
    def NLPSR(self, FEs, MaxFEs):
        self.sort(self.NP)
        N = self.cal_NP_next_gen(FEs, MaxFEs)
        A = int(max(N * 2.1, self.Nmin))
        N = int(N)
        if N < self.NP:
            self.slice(N)
        if A < self.archive.shape[0]:
            self.NA = A
            self.archive = self.archive[:A]

    # update archive, join new individual
    def update_archive(self, old_id):
        if self.archive.shape[0] < self.NA:
            self.archive = np.append(self.archive, self.group[old_id]).reshape(-1, self.dim)
        else:
            self.archive[np.random.randint(self.archive.shape[0])] = self.group[old_id]

    # collect all the features of the group  dim = 6
    def get_feature(self,
                    problem,            # the optimizing problem
                    sample_costs,
                    cost_scale_factor,  # a scale factor to normalize costs
                    progress            # the current progress of evaluations
                    ):
        gbc = self.gbest / cost_scale_factor
        fdc = cal_fdc(self.group/100, self.cost/cost_scale_factor)
        random_walk_samples = rw_sampling(self.group)
        walk_costs = problem.func(random_walk_samples)
        rf = cal_rf(walk_costs)
        acf = cal_acf(walk_costs)
        nopt = cal_nopt(random_walk_samples, walk_costs)
        # disp, disp_ratio, evp, nsc, anr, ni, nw, adf
        disp, disp_ratio = dispersion(self.group, self.cost)
        evp = population_evolvability(self.cost, sample_costs)
        nsc = negative_slope_coefficient(self.cost, sample_costs[0])
        anr = average_neutral_ratio(self.cost, sample_costs)
        ni, nw = non_improvable_worsenable(self.cost, sample_costs)
        adf = average_delta_fitness(self.cost, sample_costs)
        # return [gbc, fdc, rf, acf, nopt, disp, disp_ratio, evp, nsc, anr, ni, nw, adf, progress]  # 14
        return [gbc, fdc, disp, disp_ratio, nsc, anr, ni, nw, progress]  # 9

