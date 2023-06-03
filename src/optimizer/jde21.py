import numpy as np
import copy
from optimizer.basic_optimizer import basic_optimizer


class JDE21(basic_optimizer):
    def __init__(self, config):
        super(JDE21, self).__init__(config)
        self.__dim = config.dim      # problem dimension
        self.__sNP = 10       # size of small population
        self.__bNP = 160      # size of big population
        self.__NP = self.__sNP + self.__bNP
        # meaning of following parameters reference from the JDE21 paper
        self.__tao1 = 0.1
        self.__tao2 = 0.1
        self.__Finit = 0.5
        self.__CRinit = 0.9
        self.__Fl_b = 0.1
        self.__Fl_s = 0.17
        self.__Fu = 1.1
        self.__CRl_b = 0.0
        self.__CRl_s = 0.1
        self.__CRu_b = 1.1
        self.__CRu_s = 0.8
        self.__eps = 1e-12
        self.__MyEps = 0.25
        # record number of operation called
        self.__nReset = 0
        self.__sReset = 0
        self.__cCopy = 0
        # self.__terminateErrorValue = 1e-8
        self.__MaxFEs = config.maxFEs
        self.__FEs = 0
        self.gbest = 1e15
        self.__F = np.ones(self.__NP) * 0.5
        self.__Cr = np.ones(self.__NP) * 0.9
        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval

    # check whether the optimization stuck(global best doesn't improve for a while)
    def __prevecEnakih(self, cost, best):
        eqs = len(cost[np.fabs(cost - best) < self.__eps])
        return eqs > 2 and eqs > len(cost) * self.__MyEps

    # crowding operation describe in JDE21
    def __crowding(self, group, vs):
        NP, dim = vs.shape
        dist = np.sum(((group * np.ones((NP, NP, dim))).transpose(1, 0, 2) - vs) ** 2, -1).transpose()
        return np.argmin(dist, -1)

    def __evaluate(self, problem, Xs):
        if problem.optimum is None:
            cost = problem.eval(Xs)
        else:
            cost = problem.eval(Xs) - problem.optimum
        # cost[cost < self.__terminateErrorValue] = 0.0
        return cost

    def __sort(self):
        # new index after sorting
        ind = np.argsort(self.__cost)
        self.__cost = self.__cost[ind]
        self.__population = self.__population[ind]

    def __reinitialize(self, size, ub, lb):
        return np.random.random((size, self.__dim)) * (ub - lb) + ub

    def __init_population(self, problem):
        self.__sNP = 10
        self.__bNP = 160
        self.__NP = self.__sNP + self.__bNP
        self.__population = np.random.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb
        self.__cost = self.__evaluate(problem, self.__population)
        self.__FEs = self.__NP
        self.__cbest = self.gbest = np.min(self.__cost)
        self.__cbest_id = np.argmin(self.__cost)
        self.__F = np.ones(self.__NP) * 0.5
        self.__Cr = np.ones(self.__NP) * 0.9

        self.log_index = 1
        self.cost = [self.gbest]

    def __update(self,
                 problem,       # the problem instance
                 ):
        # initialize population
        NP = self.__NP
        dim = self.__dim
        sNP = self.__sNP
        bNP = NP - sNP
        age = 0

        def __mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big):
            if big:
                xNP = bNP
                randF = np.random.rand(xNP) * self.__Fu + self.__Fl_b
                randCr = np.random.rand(xNP) * self.__CRu_b + self.__CRl_b
                pF = self.__F[:xNP]
                pCr = self.__Cr[:xNP]
            else:
                xNP = sNP
                randF = np.random.rand(xNP) * self.__Fu + self.__Fl_s
                randCr = np.random.rand(xNP) * self.__CRu_b + self.__CRl_s
                pF = self.__F[bNP:]
                pCr = self.__Cr[bNP:]

            rvs = np.random.rand(xNP)
            F = np.where(rvs < self.__tao1, randF, pF)
            rvs = np.random.rand(xNP)
            Cr = np.where(rvs < self.__tao2, randCr, pCr)
            Fs = F.repeat(dim).reshape(xNP, dim)
            Cr[Cr > 1] = 0
            Crs = Cr.repeat(dim).reshape(xNP, dim)
            v = self.__population[r1] + Fs * (self.__population[r2] - self.__population[r3])
            v[v > problem.ub] = (v[v > problem.ub] - problem.lb) % (problem.ub - problem.lb) + problem.lb
            v[v < problem.lb] = (v[v < problem.lb] - problem.ub) % (problem.ub - problem.lb) + problem.lb
            # v = np.clip(v, problem.lb, problem.ub)
            jrand = np.random.randint(dim, size=xNP)
            u = np.where(np.random.rand(xNP, dim) < Crs, v, (self.__population[:bNP] if big else self.__population[bNP:]))
            u[np.arange(xNP), jrand] = v[np.arange(xNP), jrand]
            cost = self.__evaluate(problem, u)
            if big:
                crowding_ids = self.__crowding(self.__population[:xNP], u)
            else:
                crowding_ids = np.arange(xNP) + bNP
            age += xNP
            for i in range(xNP):
                id = crowding_ids[i]
                if cost[i] < self.__cost[id]:
                    # update and record
                    self.__population[id] = u[i]
                    self.__cost[id] = cost[i]
                    self.__F[id] = F[i]
                    self.__Cr[id] = Cr[i]
                    SF = np.append(SF, F[i])
                    SCr = np.append(SCr, Cr[i])
                    d = (self.__cost[i] - cost[i]) / (self.__cost[i] + 1e-9)
                    df = np.append(df, d)
                    if cost[i] < self.__cbest:
                        age = 0
                        self.__cbest_id = id
                        self.__cbest = cost[i]
                        if cost[i] < self.gbest:
                            self.gbest = cost[i]

            return SF, SCr, df, age

        # self.__sort()
        # initialize temp records
        # small population evaluates same times as big one thus the total evaluations for a loop is doubled big one
        df = np.array([])
        SF = np.array([])
        SCr = np.array([])
        if self.__prevecEnakih(self.__cost[:bNP], self.gbest) or age > self.__MaxFEs / 10:
            self.__nReset += 1
            self.__population[:bNP] = self.__reinitialize(bNP, problem.ub, problem.lb)
            self.__F[:bNP] = self.__Finit
            self.__Cr[:bNP] = self.__CRinit
            self.__cost[:bNP] = 1e15
            age = 0
            self.__cbest = np.min(self.__cost)
            self.__cbest_id = np.argmin(self.__cost)

        if self.__FEs < self.__MaxFEs / 3:
            mig = 1
        elif self.__FEs < 2 * self.__MaxFEs / 3:
            mig = 2
        else:
            mig = 3

        r1 = np.random.randint(bNP, size=bNP)
        count = 0
        duplicate = np.where((r1 == np.arange(bNP)) * (r1 == self.__cbest_id))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(bNP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(bNP)) * (r1 == self.__cbest_id))[0]
            count += 1

        r2 = np.random.randint(bNP + mig, size=bNP)
        count = 0
        duplicate = np.where((r2 == np.arange(bNP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(bNP + mig, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(bNP)) + (r2 == r1))[0]
            count += 1

        r3 = np.random.randint(bNP + mig, size=bNP)
        count = 0
        duplicate = np.where((r3 == np.arange(bNP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r3[duplicate] = np.random.randint(bNP + mig, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(bNP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        SF, SCr, df, age = __mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big=True)
        self.__FEs += bNP

        if self.__cbest_id >= bNP and self.__prevecEnakih(self.__cost[bNP:], self.__cbest):
            self.__sReset += 1
            cbest = self.__cbest
            cbest_id = self.__cbest_id
            tmp = copy.deepcopy(self.__population[cbest_id])
            self.__population[bNP:] = self.__reinitialize(sNP, problem.ub, problem.lb)
            self.__F[bNP:] = self.__Finit
            self.__Cr[bNP:] = self.__CRinit
            self.__cost[bNP:] = 1e15
            self.__cbest = cbest
            self.__cbest_id = cbest_id
            self.__population[cbest_id] = tmp
            self.__cost[cbest_id] = cbest

        if self.__cbest_id < bNP:
            self.__cCopy += 1
            self.__cost[bNP] = self.__cbest
            self.__population[bNP] = self.__population[self.__cbest_id]
            self.__cbest_id = bNP

        for i in range(bNP // sNP):

            r1 = np.random.randint(sNP, size=sNP) + bNP
            count = 0
            duplicate = np.where(r1 == (np.arange(sNP) + bNP))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r1[duplicate] = np.random.randint(sNP, size=duplicate.shape[0]) + bNP
                duplicate = np.where(r1 == (np.arange(sNP) + bNP))[0]
                count += 1

            r2 = np.random.randint(sNP, size=sNP) + bNP
            count = 0
            duplicate = np.where((r2 == (np.arange(sNP) + bNP)) + (r2 == r1))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r2[duplicate] = np.random.randint(sNP, size=duplicate.shape[0]) + bNP
                duplicate = np.where((r2 == (np.arange(sNP) + bNP)) + (r2 == r1))[0]
                count += 1

            r3 = np.random.randint(sNP, size=sNP) + bNP
            count = 0
            duplicate = np.where((r3 == (np.arange(sNP) + bNP)) + (r3 == r1) + (r3 == r2))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r3[duplicate] = np.random.randint(sNP, size=duplicate.shape[0]) + bNP
                duplicate = np.where((r3 == (np.arange(sNP) + bNP)) + (r3 == r1) + (r3 == r2))[0]
                count += 1

            SF, SCr, df, age = __mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big=False)
            self.__FEs += sNP

        # update and record information for NL-SHADE-RSP and reduce population
        self.gbest = np.min(self.__cost)
        if self.__FEs - self.__NP <= 0.25 * self.__MaxFEs <= self.__FEs or self.__FEs - self.__NP <= 0.5 * self.__MaxFEs <= self.__FEs or self.__FEs - self.__NP <= 0.75 * self.__MaxFEs <= self.__FEs:
            self.__bNP //= 2
            self.__population = self.__population[self.__bNP:]
            self.__cost = self.__cost[self.__bNP:]
            self.__F = self.__F[self.__bNP:]
            self.__Cr = self.__Cr[self.__bNP:]
            self.__NP = self.__bNP + self.__sNP
            self.__cbest_id = np.argmin(self.__cost)

        if self.__FEs >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest)

        if problem.optimum is None:
            return False
        else:
            return self.gbest <= 1e-8

    def run_episode(self, problem):
        self.__init_population(problem)
        while self.__FEs < self.__MaxFEs:
            is_done = self.__update(problem)
            if is_done:
                break
        if len(self.cost) >= self.__n_logpoint + 1:
            self.cost[-1] = self.gbest
        else:
            self.cost.append(self.gbest)
        return {'cost': self.cost, 'fes': self.__FEs}
