"""
    An example of controlling both hyper-parameters and operators individual-wisely in MadDE,
    while action is presented as a dict.
"""

import numpy as np
from L2OBench.Optimizer import Learnable_Optimizer, parent


class MadDE(Learnable_Optimizer):
    def __init__(self, dim, lower_bound, upper_bound, population_size, maxFEs,
                 p=0.18):
        super().__init__(dim, lower_bound, upper_bound, population_size, maxFEs)
        self.archive = None       # the archive(collection of replaced individuals)
        self.NA = None            # the max size of archive(collection of replaced individuals)
        self.p = p                # p-best factor

    def init_population(self, problem):
        super().init_population(problem)
        self.archive = np.array([])
        self.NA = int(self.NP * 2.1)

    # sort former 'size' population in respect to cost
    def sort(self, size, reverse=False):
        r = -1 if reverse else 1
        ind = np.concatenate((np.argsort(r * self.cost[:size]), np.arange(self.NP)[size:]))  # new index after sorting
        self.cost = self.cost[ind]
        self.population = self.population[ind]

    # update archive, join new individual
    def update_archive(self, old_id):
        if self.archive.shape[0] < self.NA:
            self.archive = np.append(self.archive, self.population[old_id]).reshape(-1, self.dim)
        else:
            self.archive[np.random.randint(self.archive.shape[0])] = self.population[old_id]  # 随机选一个替换

    # current-to-best/1 with archive
    def ctb_w_arc(self, group_id, best, Fs):
        NP = self.NP
        sNP = group_id.shape[0]
        NB = best.shape[0]
        NA = self.archive.shape[0]

        count = 0
        rb = np.random.randint(NB, size=sNP)
        duplicate = np.where(rb == group_id)[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == group_id)[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=sNP)
        duplicate = np.where((r1 == rb) + (r1 == group_id))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == group_id))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=sNP)
        duplicate = np.where((r2 == rb) + (r2 == group_id) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == group_id) + (r2 == r1))[0]
            count += 1

        xi = self.population[group_id]
        xb = best[rb]
        x1 = self.population[r1]
        if NA > 0:
            x2 = np.concatenate((self.population, self.archive), 0)[r2]
        else:
            x2 = self.population[r2]
        v = xi + Fs * (xb - xi) + Fs * (x1 - x2)

        return v

    # current-to-rand/1 with archive
    def ctr_w_arc(self, group_id, Fs):
        NP = self.NP
        sNP = group_id.shape[0]
        NA = self.archive.shape[0]

        count = 0
        r1 = np.random.randint(NP, size=sNP)
        duplicate = np.where(r1 == group_id)[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == group_id)[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=sNP)
        duplicate = np.where((r2 == group_id) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == group_id) + (r2 == r1))[0]
            count += 1

        xi = self.population[group_id]
        x1 = self.population[r1]
        if NA > 0:
            x2 = np.concatenate((self.population, self.archive), 0)[r2]
        else:
            x2 = self.population[r2]
        v = xi + Fs * (x1 - x2)

        return v

    # weighted rand-to-best/1
    def weighted_rtb(self, group_id, best, Fs, Fas):
        NP = self.NP
        sNP = group_id.shape[0]
        NB = best.shape[0]

        count = 0
        rb = np.random.randint(NB, size=sNP)
        duplicate = np.where(rb == group_id)[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == group_id)[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=sNP)
        duplicate = np.where((r1 == rb) + (r1 == group_id))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == group_id))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=sNP)
        duplicate = np.where((r2 == rb) + (r2 == group_id) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == group_id) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = self.population[r1]
        x2 = self.population[r2]

        v = Fs * x1 + Fs * Fas * (xb - x2)

        return v

    # binomial crossover
    def binomial(self, x, v, Crs):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def update(self, problem, action):
        """
        :param problem: Problem instance.
        :param action: A dict contains 4 elements(each as np.ndarray of shape[NP]) as follows:
               action['F']: mutation factors;
               action['Cr']: crossover rates;
               action['Mo']: mutation operators (0, 1 and 2 stand for ctb_w_arc, ctr_w_arc and weighted_rtb respectively);
               action['Co']: crossover operators (0 and 1 stand for binomial and q-best binomial respectively).
        """
        self.sort(self.NP)
        NP, dim = self.NP, self.dim
        q = 2 * self.p - self.p * self.fes / self.maxFEs  # qbest: top q%
        Fa = 0.5 + 0.5 * self.fes / self.maxFEs  # attraction Factor

        # Mutation
        pbest = self.population[:max(int(self.p * NP), 2)]
        qbest = self.population[:max(int(q * NP), 2)]
        F = action['F'][:, None]
        Mo = action['Mo']
        v1 = self.ctb_w_arc(np.where(Mo == 0)[0], pbest, F[Mo == 0])
        v2 = self.ctr_w_arc(np.where(Mo == 1)[0], F[Mo == 1])
        v3 = self.weighted_rtb(np.where(Mo == 2)[0], qbest, F[Mo == 2], Fa)
        v = np.zeros((NP, dim))
        v[Mo == 0] = v1
        v[Mo == 1] = v2
        v[Mo == 2] = v3

        # Boundary control
        v = parent(v, self.lb, self.ub, self.population)

        # Crossover
        Cr = action['Cr'][:, None]
        u = np.zeros((NP, dim))
        Co = action['Co']
        u[Co == 0.] = self.binomial(self.population[Co == 0.], v[Co == 0.], Cr[Co == 0.])
        if self.archive.shape[0] > 0:
            qbest = np.concatenate((self.population, self.archive), 0)[:max(int(q * (NP + self.archive.shape[0])), 2)]
        cross_qbest = qbest[np.random.randint(qbest.shape[0], size=(Co == 1.).sum())]
        u[Co == 1.] = self.binomial(cross_qbest, v[Co == 1.], Cr[Co == 1.])

        # Selection
        ncost = problem.eval(u) - problem.optimum
        self.fes += NP

        # Update archive
        optim = np.where(ncost < self.cost)[0]
        for i in optim:
            self.update_archive(i)

        self.population[optim] = u[optim]
        self.cost = np.minimum(self.cost, ncost)
        self.gbest_cost = np.minimum(self.gbest_cost, np.min(self.cost))
