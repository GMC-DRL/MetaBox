from .Population import *
import time
import numpy as np
import copy


class NL_SHADE_RSP:
    def __init__(self, dim, error=1e-8):
        self.pb = 0.4   # rate of best individuals in mutation
        self.pa = 0.5   # rate of selecting individual from archive
        self.dim = dim  # dimension of problem
        self.error = error

    def evaluate(self, problem, u):
        if problem.optimum is not None:
            cost = problem.eval(u) - problem.optimum
            cost[cost < self.error] = 0.0
        else:
            cost = problem.eval(u)
        return cost

    # Binomial crossover
    def Binomial(self, x, v, cr):
        dim = len(x)
        jrand = np.random.randint(dim)
        u = np.where(np.random.random(dim) < cr, v, x)
        u[jrand] = v[jrand]
        return u

    # Binomial crossover
    def Binomial_(self, x, v, cr):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < cr.repeat(dim).reshape(NP, dim), v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    # Exponential crossover
    def Exponential(self, x, v, cr):
        dim = len(x)
        u = x.copy()
        L = np.random.randint(dim)
        for i in range(L, dim):
            if np.random.random() < cr:
                u[i] = v[i]
            else:
                break
        return u

    # Exponential crossover
    def Exponential_(self, x, v, cr):
        NP, dim = x.shape
        u = x.copy()
        L = np.random.randint(dim, size=NP).repeat(dim).reshape(NP, dim)
        L = L <= np.arange(dim)
        rvs = np.random.rand(NP, dim)
        L = np.where(rvs > cr.repeat(dim).reshape(NP, dim), L, 0)
        u = u * (1 - L) + v * L
        return u

    # update pa according to cost changes
    def update_Pa(self, fa,fp,na,NP):
        if na == 0 or fa == 0:
            self.pa = 0.5
            return
        self.pa = (fa / (na + 1e-15)) / ((fa / (na + 1e-15)) + (fp / (NP-na + 1e-15)))
        self.pa = np.minimum(0.9, np.maximum(self.pa, 0.1))

    # step method for ensemble, optimize population for a few times
    def step(self,
             population,    # an initialized or half optimized population, the method will optimize it
             problem,       # the problem instance
             FEs,           # used number of evaluations, also the starting of current step
             FEs_end,       # the ending evaluation number of step, step stop while reaching this limitation
                            # i.e. user wants to run a step with 1000 evaluations period, it should be FEs + 1000
             MaxFEs,         # the max number of evaluations
             ):
        # initialize population and archive
        NP, dim = population.NP, population.dim
        NA = int(NP * 2.1)
        if NA < population.archive.shape[0]:
            population.archive = population.archive[:NA]
        self.pa = 0.5
        population.sort(population.NP)
        # start optimization loop
        while FEs < FEs_end and FEs < MaxFEs:
            t1 = time.time()
            # select crossover rate and step length
            Cr, F = population.choose_F_Cr()
            Cr = np.sort(Cr)
            # initialize some record values
            fa = 0                                      # sum of cost improvement using archive
            fp = 0                                      # sum of cost improvement without archive
            ap = np.zeros(NP, bool)                     # record of whether a individual update with archive
            df = np.array([])                           # record of cost improvement of each individual
            pr = np.exp(-(np.arange(NP) + 1) / NP)      # calculate the rate of individuals at different positions being selected in others' mutation
            pr /= np.sum(pr)
            na = 0                                      # the number of archive usage
            SF = np.array([])                           # the set records successful step length
            SCr = np.array([])                          # the set records successful crossover rate
            u = np.zeros((NP, dim))                     # trail vectors
            # randomly select a crossover method for the population
            CrossExponential = np.random.random() < 0.5
            t2 = time.time()
            pb_upper = int(np.maximum(2, NP * self.pb))  # the range of pbest selection
            pbs = np.random.randint(pb_upper, size=NP)   # select pbest for all individual
            count = 0
            duplicate = np.where(pbs == np.arange(NP))[0]
            while duplicate.shape[0] > 0 and count < 1:
                pbs[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
                duplicate = np.where(pbs == np.arange(NP))[0]
                count += 1
            xpb = population.group[pbs]
            t3 = time.time()
            r1 = np.random.randint(NP, size=NP)
            count = 0
            duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
                duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
                count += 1
            x1 = population.group[r1]
            t4 = time.time()
            rvs = np.random.rand(NP)
            r2_pop = np.where(rvs >= self.pa)[0]  # the indices of mutation with population
            r2_arc = np.where(rvs < self.pa)[0]   # the indices of mutation with archive
            use_arc = np.zeros(NP, dtype=bool)    # a record for archive usage, used in parameter updating
            use_arc[r2_arc] = 1
            if population.archive.shape[0] < 25:   # if the archive is empty, indices above are canceled
                r2_pop = np.arange(NP)
                r2_arc = np.array([], dtype=np.int32)
            r2 = np.random.choice(np.arange(NP), size=r2_pop.shape[0], p=pr)
            count = 0
            duplicate = np.where((r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop]))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r2[duplicate] = np.random.choice(np.arange(NP), size=duplicate.shape[0], p=pr)
                duplicate = np.where((r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop]))[0]
                count += 1
            x2 = np.zeros((NP, self.dim))
            t5 = time.time()
            # scatter indiv from population and archive into x2
            if r2_pop.shape[0] > 0:
                x2[r2_pop] = population.group[r2]
            if r2_arc.shape[0] > 0:
                x2[r2_arc] = population.archive[np.random.randint(np.minimum(population.archive.shape[0], NA), size=r2_arc.shape[0])]
            Fs = F.repeat(self.dim).reshape(NP, self.dim)   # adjust shape for batch processing
            vs = population.group + Fs * (xpb - population.group) + Fs * (x1 - x2)
            # crossover rate for Binomial crossover has a different way for calculating
            Crb = np.zeros(NP)
            tmp_id = np.where(np.arange(NP) + FEs < 0.5 * MaxFEs)[0]
            Crb[tmp_id] = 2 * ((FEs + tmp_id) / MaxFEs - 0.5)
            if CrossExponential:
                Cr = Crb
                us = self.Binomial_(population.group, vs, Cr)
            else:
                us = self.Exponential_(population.group, vs, Cr)
            # reinitialize values exceed valid range
            us = us * ((-100 <= us) * (us <= 100)) + ((us > 100) + (us < -100)) * (np.random.rand(NP, dim) * 200 - 100)
            t6 = time.time()
            cost = self.evaluate(problem, us)
            optim = np.where(cost < population.cost)[0]  # the indices of indiv whose costs are better than parent
            for i in range(optim.shape[0]):
                population.update_archive(i)
            population.F[optim] = F[optim]
            population.Cr[optim] = Cr[optim]
            SF = F[optim]
            SCr = Cr[optim]
            df = (population.cost[optim] - cost[optim]) / (population.cost[optim] + 1e-9)
            arc_usage = use_arc[optim]
            fp = np.sum(df[arc_usage])
            fa = np.sum(df[np.array(1 - arc_usage, dtype=bool)])
            na = np.sum(arc_usage)
            population.group[optim] = us[optim]
            population.cost[optim] = cost[optim]
            t7 = time.time()

            if np.min(cost) < population.gbest:
                population.gbest = np.min(cost)
                population.gbest_solution = population.group[np.argmin(cost)]

            FEs += NP
            # adaptively adjust parameters
            self.pb = 0.4 - 0.2 * (FEs / MaxFEs)
            population.NLPSR(FEs, MaxFEs)
            population.update_M_F_Cr(SF, SCr, df)
            self.update_Pa(fa, fp, na, NP)
            NP = population.NP
            NA = population.NA
            if np.min(cost) < self.error:
                return population, min(FEs, MaxFEs)

        return population, min(FEs, MaxFEs)


class JDE21:
    def __init__(self, dim, error=1e-8):
        self.dim = dim      # problem dimension
        self.sNP = 10       # size of small population
        self.bNP = 160      # size of big population
        # meaning of following parameters reference from the JDE21 paper
        self.tao1 = 0.1
        self.tao2 = 0.1
        self.Finit = 0.5
        self.CRinit = 0.9
        self.Fl_b = 0.1
        self.Fl_s = 0.17
        self.Fu = 1.1
        self.CRl_b = 0.0
        self.CRl_s = 0.1
        self.CRu_b = 1.1
        self.CRu_s = 0.8
        self.eps = 1e-12
        self.MyEps = 0.25
        # record number of operation called
        self.nReset = 0
        self.sReset = 0
        self.cCopy = 0
        self.terminateErrorValue = error

    # check whether the optimization stuck(global best doesn't improve for a while)
    def prevecEnakih(self, cost, best):
        eqs = len(cost[np.fabs(cost - best) < self.eps])
        return eqs > 2 and eqs > len(cost) * self.MyEps

    # crowding operation describe in JDE21
    def crowding(self, group, v):
        dist = np.sum((group - v) ** 2, -1)
        return np.argmin(dist)

    def crowding_(self, group, vs):
        NP, dim = vs.shape
        dist = np.sum(((group * np.ones((NP, NP, dim))).transpose(1, 0, 2) - vs) ** 2, -1).transpose()
        return np.argmin(dist, -1)

    def evaluate(self, Xs, problem):
        if problem.optimum is not None:
            cost = problem.eval(Xs) - problem.optimum
        else:
            cost = problem.eval(Xs)
        cost[cost < self.terminateErrorValue] = 0.0
        return cost

    def step(self,
             population,    # an initialized or half optimized population, the method will optimize it
             problem,       # the problem instance
             FEs,           # used number of evaluations, also the starting of current step
             FEs_end,       # the ending evaluation number of step, step stop while reaching this limitation
             MaxFEs,         # the max number of evaluations
             ):
        # initialize population
        NP = population.NP
        dim = population.dim
        sNP = self.sNP
        bNP = NP - sNP
        age = 0

        def mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big):
            if big:
                xNP = bNP
                randF = np.random.rand(xNP) * self.Fu + self.Fl_b
                randCr = np.random.rand(xNP) * self.CRu_b + self.CRl_b
                pF = population.F[:xNP]
                pCr = population.Cr[:xNP]
            else:
                xNP = sNP
                randF = np.random.rand(xNP) * self.Fu + self.Fl_s
                randCr = np.random.rand(xNP) * self.CRu_b + self.CRl_s
                pF = population.F[-sNP:]
                pCr = population.Cr[-sNP:]

            rvs = np.random.rand(xNP)
            F = np.where(rvs < self.tao1, randF, pF)
            rvs = np.random.rand(xNP)
            Cr = np.where(rvs < self.tao2, randCr, pCr)
            Fs = F.repeat(dim).reshape(xNP, dim)
            Crs = Cr.repeat(dim).reshape(xNP, dim)
            v = population.group[r1] + Fs * (population.group[r2] - population.group[r3])
            v = np.clip(v, population.Xmin, population.Xmax)
            jrand = np.random.randint(dim, size=xNP)
            u = np.where(np.random.rand(xNP, dim) < Crs, v, (population.group[:bNP] if big else population.group[bNP:]))
            u[np.arange(xNP), jrand] = v[np.arange(xNP), jrand]
            cost = self.evaluate(u, problem)
            if big:
                crowding_ids = self.crowding_(population.group[:xNP], u)
            else:
                crowding_ids = np.arange(xNP) + bNP
            age += xNP
            for i in range(xNP):
                id = crowding_ids[i]
                if cost[i] < population.cost[id]:
                    # update and record
                    population.update_archive(id)
                    population.group[id] = u[i]
                    population.cost[id] = cost[i]
                    population.F[id] = F[i]
                    population.Cr[id] = Cr[i]
                    SF = np.append(SF, F[i])
                    SCr = np.append(SCr, Cr[i])
                    d = (population.cost[i] - cost[i]) / (population.cost[i] + 1e-9)
                    df = np.append(df, d)
                    if cost[i] < population.cbest:
                        age = 0
                        population.cbest_id = id
                        population.cbest = cost[i]
                        if cost[i] < population.gbest:
                            population.gbest = cost[i]
                            population.gbest_solution = u[i]

            return SF, SCr, df, age

        population.sort(NP, True)
        # check record point lest missing it
        while FEs < FEs_end:
            # initialize temp records
            v = np.zeros((NP, dim))
            F = np.random.random(NP)
            Cr = np.random.random(NP)
            # small population evaluates same times as big one thus the total evaluations for a loop is doubled big one
            N = bNP * 2
            I = -1
            df = np.array([])
            SF = np.array([])
            SCr = np.array([])
            if self.prevecEnakih(population.cost[:bNP], population.gbest) or age > MaxFEs / 10:
                self.nReset += 1
                population.group[:bNP] = population.initialize_group(bNP)
                population.F[:bNP] = self.Finit
                population.Cr[:bNP] = self.CRinit
                population.cost[:bNP] = 1e15
                age = 0
                population.cbest = np.min(population.cost)
                population.cbest_id = np.argmin(population.cost)

            if FEs < MaxFEs / 3:
                mig = 1
            elif FEs < 2 * MaxFEs / 3:
                mig = 2
            else:
                mig = 3

            r1 = np.random.randint(bNP, size=bNP)
            count = 0
            duplicate = np.where((r1 == np.arange(bNP)) * (r1 == population.cbest_id))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r1[duplicate] = np.random.randint(bNP, size=duplicate.shape[0])
                duplicate = np.where((r1 == np.arange(bNP)) * (r1 == population.cbest_id))[0]
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

            SF, SCr, df, age = mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big=True)
            FEs += bNP

            if np.min(population.cost) < self.terminateErrorValue:
                return population, min(FEs, MaxFEs)
            if FEs >= FEs_end or FEs >= MaxFEs:
                return population, min(FEs, MaxFEs)

            if population.cbest_id >= bNP and self.prevecEnakih(population.cost[bNP:], population.cbest):
                self.sReset += 1
                cbest = population.cbest
                cbest_id = population.cbest_id
                tmp = population.group[cbest_id]
                population.group[bNP:] = population.initialize_group(sNP)
                population.F[bNP:] = self.Finit
                population.Cr[bNP:] = self.CRinit
                population.cost[bNP:] = 1e15
                population.cbest = cbest
                population.cbest_id = cbest_id
                population.group[cbest_id] = tmp
                population.cost[cbest_id] = cbest

            if population.cbest_id < bNP:
                self.cCopy += 1
                population.cost[bNP] = population.cbest
                population.group[bNP] = population.group[population.cbest_id]
                population.cbest_id = bNP

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

                SF, SCr, df, age = mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big=False)
                FEs += sNP

                if np.min(population.cost) < self.terminateErrorValue:
                    return population, min(FEs, MaxFEs)
                if FEs >= FEs_end or FEs >= MaxFEs:
                    return population, min(FEs, MaxFEs)

            # update and record information for NL-SHADE-RSP and reduce population

            population.update_M_F_Cr(SF, SCr, df)
            NP = int(population.cal_NP_next_gen(FEs, MaxFEs))
            population.NP = NP
            # population.sort(NP, True)
            population.group = population.group[-NP:]
            population.cost = population.cost[-NP:]
            population.F = population.F[-NP:]
            population.Cr = population.Cr[-NP:]
            population.cbest_id = np.argmin(population.cost)
            population.cbest = np.min(population.cost)
            bNP = NP - sNP

        return population, min(FEs, MaxFEs)

    # a testing method which runs a complete optimization on a population and show its performance, similar to step


class MadDE:
    def __init__(self, dim, error=1e-8):
        self.dim = dim
        self.p = 0.18
        self.PqBX = 0.01
        self.F0 = 0.2
        self.Cr0 = 0.2
        self.pm = np.ones(3) / 3
        self.error = error

    def ctb_w_arc(self, group, best, archive, Fs):
        NP, dim = group.shape
        NB = best.shape[0]
        NA = archive.shape[0]

        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (xb - group) + Fs * (x1 - x2)

        return v

    def ctr_w_arc(self, group, archive, Fs):
        NP, dim = group.shape
        NA = archive.shape[0]

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (x1 - x2)

        return v

    def weighted_rtb(self, group, best, Fs, Fas):
        NP, dim = group.shape
        NB = best.shape[0]

        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        x2 = group[r2]
        v = Fs * x1 + Fs * Fas * (xb - x2)

        return v

    def binomial(self, x, v, Crs):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def step(self, population, problem, FEs, FEs_end, MaxFEs):
        population.sort(population.NP)
        while FEs < FEs_end and FEs < MaxFEs:
            NP, dim = population.NP, population.dim
            q = 2 * self.p - self.p * FEs / MaxFEs
            Fa = 0.5 + 0.5 * FEs / MaxFEs
            Cr, F = population.choose_F_Cr()
            mu = np.random.choice(3, size=NP, p=self.pm)
            p1 = population.group[mu == 0]
            p2 = population.group[mu == 1]
            p3 = population.group[mu == 2]
            pbest = population.group[:max(int(self.p * NP), 2)]
            qbest = population.group[:max(int(q * NP), 2)]
            Fs = F.repeat(dim).reshape(NP, dim)
            v1 = self.ctb_w_arc(p1, pbest, population.archive, Fs[mu == 0])
            v2 = self.ctr_w_arc(p2, population.archive, Fs[mu == 1])
            v3 = self.weighted_rtb(p3, qbest, Fs[mu == 2], Fa)
            v = np.zeros((NP, dim))
            v[mu == 0] = v1
            v[mu == 1] = v2
            v[mu == 2] = v3
            v[v < -100] = (population.group[v < -100] - 100) / 2
            v[v > 100] = (population.group[v > 100] + 100) / 2
            rvs = np.random.rand(NP)
            Crs = Cr.repeat(dim).reshape(NP, dim)
            u = np.zeros((NP, dim))
            if np.sum(rvs <= self.PqBX) > 0:
                qu = v[rvs <= self.PqBX]
                if population.archive.shape[0] > 0:
                    qbest = np.concatenate((population.group, population.archive), 0)[:max(int(q * (NP + population.archive.shape[0])), 2)]
                cross_qbest = qbest[np.random.randint(qbest.shape[0], size=qu.shape[0])]
                qu = self.binomial(cross_qbest, qu, Crs[rvs <= self.PqBX])
                u[rvs <= self.PqBX] = qu
            bu = v[rvs > self.PqBX]
            bu = self.binomial(population.group[rvs > self.PqBX], bu, Crs[rvs > self.PqBX])
            u[rvs > self.PqBX] = bu
            if problem.optimum is not None:
                ncost = problem.eval(u) - problem.optimum
            else:
                ncost = problem.eval(u)
            FEs += NP
            optim = np.where(ncost < population.cost)[0]
            for i in optim:
                population.update_archive(i)
            SF = F[optim]
            SCr = Cr[optim]
            df = np.maximum(0, population.cost - ncost)
            population.update_M_F_Cr(SF, SCr, df[optim])
            count_S = np.zeros(3)
            for i in range(3):
                count_S[i] = np.mean(df[mu == i] / population.cost[mu == i])
            if np.sum(count_S) > 0:
                self.pm = np.maximum(0.1, np.minimum(0.9, count_S / np.sum(count_S)))
                self.pm /= np.sum(self.pm)
            else:
                self.pm = np.ones(3) / 3

            population.group[optim] = u[optim]
            population.cost = np.minimum(population.cost, ncost)
            population.NLPSR(FEs, MaxFEs)
            if np.min(population.cost) < population.gbest:
                population.gbest = np.min(population.cost)
                population.gbest_solution = population.group[np.argmin(population.cost)]

            if np.min(population.cost) < self.error:
                return population, min(FEs, MaxFEs)
        return population, min(FEs, MaxFEs)
