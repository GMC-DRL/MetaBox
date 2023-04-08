import numpy as np
import scipy.stats as stats


class Learnable_Optimizer:
    """
    Abstract super class for learnable optimizers.
    """
    def __init__(self,
                 dim,
                 lower_bound,
                 upper_bound,
                 population_size,
                 maxFEs,
                 boundary_ctrl_method='clipping'):
        self.dim = dim
        self.lb = lower_bound
        self.ub = upper_bound
        self.NP = population_size
        self.maxFEs = maxFEs
        self.boundary_ctrl_method = boundary_ctrl_method

        self.population = None
        self.cost = None
        self.gbest_cost = None  # for the need of computing rewards
        self.init_cost = None   # for the need of computing rewards
        self.fes = None

    def init_population(self, problem):
        """
        Generate population randomly and compute cost.
        """
        self.population = np.random.rand(self.NP, self.dim) * (self.ub - self.lb) + self.lb  # [lb, ub]
        self.cost = problem.eval(self.population) - problem.optimum
        self.gbest_cost = self.cost.min().copy()
        self.init_cost = self.cost.copy()
        self.fes = self.NP

    def boundary_ctrl(self, x):
        y = None
        if self.boundary_ctrl_method == 'clipping':
            y = np.clip(x, self.lb, self.ub)

        elif self.boundary_ctrl_method == 'random':
            cro_bnd = (x < self.lb) | (x > self.ub)  # cross boundary
            y = ~cro_bnd * x + cro_bnd * np.random.rand(self.NP, self.dim) * (self.ub - self.lb) + self.lb

        elif self.boundary_ctrl_method == 'reflection':
            cro_lb = x < self.lb
            cro_ub = x > self.ub
            no_cro = ~(cro_lb | cro_ub)
            y = no_cro * x + cro_lb * (2 * self.lb - x) + cro_ub * (2 * self.ub - x)

        elif self.boundary_ctrl_method == 'periodic':
            y = (x - self.ub) % (self.ub - self.lb) + self.lb

        elif self.boundary_ctrl_method == 'halving':
            cro_lb = x < self.lb
            cro_ub = x > self.ub
            no_cro = ~(cro_lb | cro_ub)
            y = no_cro * x + cro_lb * (x + self.lb) / 2 + cro_ub * (x + self.ub) / 2

        elif self.boundary_ctrl_method == 'parent':
            cro_lb = x < self.lb
            cro_ub = x > self.ub
            no_cro = ~(cro_lb | cro_ub)
            y = no_cro * x + cro_lb * (self.population + self.lb) / 2 + cro_ub * (self.population + self.ub) / 2

        return y

    def evolve(self, problem, action):
        pass


class DE(Learnable_Optimizer):
    def __init__(self, dim, lower_bound, upper_bound, population_size, maxFEs, boundary_ctrl_method='clipping'):
        super().__init__(dim, lower_bound, upper_bound, population_size, maxFEs, boundary_ctrl_method)

    def generate_random_int(self, cols, check_index=True):
        """
        :param cols: the number of random int generated for each individual.

        :param check_index: whether to check the population indexes appeal in their own ''cols'' elements for each individual.
               For example, if ''check_index'' is True, 0 won't appeal in any element in r[:, 0, :].

        :return: a random int matrix in shape[''NP'', ''cols''], and elements are in a range of [0, ''population_size''-1].
                 The ''cols'' elements at dimension[2] are different from each other.
        """

        r = np.random.randint(low=0, high=self.NP, size=(self.NP, cols))
        # validity checking and modification for r
        for col in range(0, cols):
            while True:
                is_repeated = [np.equal(r[:, col], r[:, i]) for i in range(col)]  # 检查当前列与其前面所有列有无重复
                if check_index:
                    is_repeated.append(np.equal(r[:, col], np.arange(self.NP)))  # 检查当前列是否与该个体编号重复
                repeated_index = np.nonzero(np.any(np.stack(is_repeated), axis=0))[0]  # 获取重复随机数的个体下标
                repeated_sum = repeated_index.size  # 重复随机数的个数
                if repeated_sum != 0:
                    r[repeated_index[:], col] = np.random.randint(low=0, high=self.NP, size=repeated_sum)  # 重新生成并替换
                else:
                    break
        return r

    def mutate(self, mutate_strategy, F):
        """
        :param mutate_strategy: Mutation strategy will be used in this generation.
        :param F: An array of mutation factor of shape[NP].
        :return: Population mutated.
        """
        F = np.array(F).reshape(-1, 1).repeat(self.dim, axis=-1)  # [NP, dim]
        x = self.population
        y = None

        if mutate_strategy == 'rand/1':  # rand/1
            r = self.generate_random_int(3)
            y = x[r[:, 0]] + F * (x[r[:, 1]] - x[r[:, 2]])

        elif mutate_strategy == 'rand2best/1':  # rand-to-best/1
            r = self.generate_random_int(3)
            y = x[r[:, 0]] + F * \
                     (x[self.cost.argmin()].reshape(1, -1).repeat(self.NP, axis=0) - x[r[:, 0]] +
                      x[r[:, 1]] - x[r[:, 2]])

        elif mutate_strategy == 'rand2best/2':  # rand-to-best/2
            r = self.generate_random_int(5)
            y = x[r[:, 0]] + F * \
                     (x[self.cost.argmin()].reshape(1, -1).repeat(self.NP, axis=0) - x[r[:, 0]] +
                      x[r[:, 1]] - x[r[:, 2]] +
                      x[r[:, 3]] - x[r[:, 4]])

        elif mutate_strategy == 'cur2best/1':  # current-to-best/1
            r = self.generate_random_int(2)
            y = x + F * \
                     (x[self.cost.argmin()].reshape(1, -1).repeat(self.NP, axis=0) - x +
                      x[r[:, 0]] - x[r[:, 1]])

        elif mutate_strategy == 'cur2best/2':  # current-to-best/2
            r = self.generate_random_int(4)
            y = x + F * \
                     (x[self.cost.argmin()].reshape(1, -1).repeat(self.NP, axis=0) - x +
                      x[r[:, 0]] - x[r[:, 1]] +
                      x[r[:, 2]] - x[r[:, 3]])

        return y

    def crossover(self, x, Cr):
        """
        :param Cr: An array of crossover rate of shape[NP].
        :param x: The mutated population before crossover.
        :return: Population after crossover.
        """
        r = np.random.rand(self.NP, self.dim)
        r[np.arange(self.NP), np.random.randint(low=0, high=self.dim, size=self.NP)] = 0.  # 对每个个体的dim个随机数，随机地取其中一个置0
        y = np.where(r <= Cr.reshape(-1, 1).repeat(self.dim, axis=-1),
                     x,
                     self.population)
        return y

    def evolve(self, problem, action):
        """
        :param problem: Problem instance.
        :param action: An array of shape [2, NP] with action[0] represents F and action[1] represents Cr.
        """
        mutated = self.mutate('rand/1', action[0])
        mutated = self.boundary_ctrl(mutated)
        trials = self.crossover(mutated, action[1])
        # Selection
        trials_cost = problem.eval(trials) - problem.optimum
        self.fes += self.NP
        surv_filters = np.array(trials_cost <= self.cost)
        self.population = np.where(surv_filters.reshape(-1, 1).repeat(self.dim, axis=-1), trials, self.population)
        self.cost = np.where(surv_filters, trials_cost, self.cost)
        self.gbest_cost = np.minimum(self.gbest_cost, np.min(self.cost))


class MadDE(Learnable_Optimizer):
    def __init__(self, dim, lower_bound, upper_bound, population_size, maxFEs, boundary_ctrl_method='parent',
                 p=0.18, PqBX=0.01, enable_NLPSR=False):
        super().__init__(dim, lower_bound, upper_bound, population_size, maxFEs, boundary_ctrl_method)
        self.Nmax = self.NP                   # the upperbound of population size
        self.Nmin = 4                         # the lowerbound of population size
        self.enable_NLPSR = enable_NLPSR      # enable Non-Linear Population Size Reduction

        self.archive = None       # the archive(collection of replaced individuals)
        self.NA = None            # the max size of archive(collection of replaced individuals)
        self.MF = None            # the set of step length of DE
        self.MCr = None           # the set of crossover rate of DE
        self.k = None             # the index of updating element in MF and MCr
        self.pm = None
        # core hyperparameters
        self.p = p
        self.PqBX = PqBX

    def init_population(self, problem):
        super().init_population(problem)
        self.archive = np.array([])
        self.NA = int(self.NP * 2.1)
        self.MF = np.ones(self.dim * 20) * 0.2
        self.MCr = np.ones(self.dim * 20) * 0.2
        self.k = 0
        self.pm = np.ones(3) / 3

    # sort former 'size' population in respect to cost
    def sort(self, size, reverse=False):
        r = -1 if reverse else 1
        ind = np.concatenate((np.argsort(r * self.cost[:size]), np.arange(self.NP)[size:]))  # new index after sorting
        self.cost = self.cost[ind]
        self.population = self.population[ind]

    # calculate wL mean
    def mean_wL(self, df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # randomly choose step length nad crossover rate from MF and MCr
    def choose_F_Cr(self):
        # generate Cr can be done simultaneously
        ind_r = np.random.randint(0, self.MF.shape[0], size=self.NP)  # index
        C_r = np.minimum(1, np.maximum(0, np.random.normal(loc=self.MCr[ind_r], scale=0.1, size=self.NP)))  # 0~1
        # as for F, need to generate 1 by 1
        cauchy_locs = self.MF[ind_r]
        F = stats.cauchy.rvs(loc=cauchy_locs, scale=0.1, size=self.NP)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
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

    # Non-Linearly Population Size Reduction
    def NLPSR(self):
        self.sort(self.NP)
        # calculate new population size
        N = np.round(self.Nmax + (self.Nmin - self.Nmax) * np.power(self.fes/self.maxFEs, 1-self.fes/self.maxFEs))
        A = int(max(N * 2.1, self.Nmin))
        N = int(N)
        if N < self.NP:
            self.NP = N
            self.population = self.population[:N]
            self.cost = self.cost[:N]
        if A < self.archive.shape[0]:
            self.NA = A
            self.archive = self.archive[:A]

    # update archive, join new individual
    def update_archive(self, old_id):
        if self.archive.shape[0] < self.NA:
            self.archive = np.append(self.archive, self.population[old_id]).reshape(-1, self.dim)
        else:
            self.archive[np.random.randint(self.archive.shape[0])] = self.population[old_id]  # 随机选一个替换

    # current-to-best with archive
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

    # current-to-rand with archive
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

    # weighted rand-to-best
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

    def evolve(self, problem, action):
        self.sort(self.NP)
        NP, dim = self.NP, self.dim
        q = 2 * self.p - self.p * self.fes / self.maxFEs  # qbest: top q%
        Fa = 0.5 + 0.5 * self.fes / self.maxFEs  # attraction Factor
        Cr, F = self.choose_F_Cr()
        # Mutation
        mu = np.random.choice(3, size=NP, p=self.pm)
        pbest = self.population[:max(int(self.p * NP), 2)]
        qbest = self.population[:max(int(q * NP), 2)]
        Fs = F.repeat(dim).reshape(NP, dim)
        v1 = self.ctb_w_arc(np.where(mu == 0)[0], pbest, Fs[mu == 0])
        v2 = self.ctr_w_arc(np.where(mu == 1)[0], Fs[mu == 1])
        v3 = self.weighted_rtb(np.where(mu == 2)[0], qbest, Fs[mu == 2], Fa)
        v = np.zeros((NP, dim))
        v[mu == 0] = v1
        v[mu == 1] = v2
        v[mu == 2] = v3
        # Boundary control
        v = self.boundary_ctrl(v)
        # Crossover
        rvs = np.random.rand(NP)
        Crs = Cr.repeat(dim).reshape(NP, dim)
        u = np.zeros((NP, dim))
        if np.sum(rvs <= self.PqBX) > 0:
            qu = v[rvs <= self.PqBX]
            if self.archive.shape[0] > 0:
                qbest = np.concatenate((self.population, self.archive), 0)[:max(int(q * (NP + self.archive.shape[0])), 2)]
            cross_qbest = qbest[np.random.randint(qbest.shape[0], size=qu.shape[0])]
            qu = self.binomial(cross_qbest, qu, Crs[rvs <= self.PqBX])
            u[rvs <= self.PqBX] = qu
        bu = v[rvs > self.PqBX]
        bu = self.binomial(self.population[rvs > self.PqBX], bu, Crs[rvs > self.PqBX])
        u[rvs > self.PqBX] = bu
        # Selection
        ncost = problem.eval(u) - problem.optimum
        self.fes += NP
        last_NP = NP
        # Update archive, MF and MCr
        optim = np.where(ncost < self.cost)[0]
        for i in optim:
            self.update_archive(i)
        SF = F[optim]
        SCr = Cr[optim]
        df = np.maximum(0, self.cost - ncost)
        self.update_M_F_Cr(SF, SCr, df[optim])
        # Update pm
        count_S = np.zeros(3)
        for i in range(3):
            count_S[i] = np.mean(df[mu == i] / self.cost[mu == i])
        if np.sum(count_S) > 0:
            self.pm = np.maximum(0.1, np.minimum(0.9, count_S / np.sum(count_S)))
            self.pm /= np.sum(self.pm)
        else:
            self.pm = np.ones(3) / 3

        self.population[optim] = u[optim]
        self.cost = np.minimum(self.cost, ncost)
        self.gbest_cost = np.minimum(self.gbest_cost, np.min(self.cost))
        if self.enable_NLPSR:
            self.NLPSR()


class PSO(Learnable_Optimizer):
    def __init__(self, dim, lower_bound, upper_bound, population_size, maxFEs, boundary_ctrl_method='clipping',
                 w_decay=True, c=4.1, max_velocity=10):
        super().__init__(dim, lower_bound, upper_bound, population_size, maxFEs, boundary_ctrl_method)
        self.w_decay = w_decay
        self.c = c
        self.max_velocity = max_velocity

        self.velocity = None
        self.w = None
        self.gbest = None
        self.pbest = None
        self.pbest_cost = None

    def init_population(self, problem):
        super().init_population(problem)
        self.velocity = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.NP, self.dim))
        self.gbest = self.population[self.cost.argmin()].copy()
        self.pbest = self.population.copy()
        self.pbest_cost = self.cost.copy()
        if self.w_decay:
            self.w = 0.9
        else:
            self.w = 0.729

    def evolve(self, problem, action):
        """
        :param problem: Problem instance.
        :param action: An array of shape [NP] controlling exploration-exploitation tradeoff.
        """
        # linearly decreasing the coefficient of inertia w
        if self.w_decay:
            self.w -= 0.5 / (self.maxFEs / self.NP)
        # generate two set of random val for pso velocity update
        rand1 = np.random.rand(self.NP, 1)
        rand2 = np.random.rand(self.NP, 1)
        # update velocity
        action = action[:, None]
        new_velocity = self.w * self.velocity + self.c * action * rand1 * (self.pbest - self.population) + \
                       self.c * (1 - action) * rand2 * (self.gbest[None, :] - self.population)
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)
        # get new population
        new_population = self.population + new_velocity
        new_population = self.boundary_ctrl(new_population)
        # get new cost
        new_cost = problem.eval(new_population) - problem.optimum
        self.fes += self.NP
        # update particles
        pbest_filters = new_cost < self.pbest_cost
        cbest_cost = new_cost.min()
        cbest_index = new_cost.argmin()
        gbest_filter = cbest_cost < self.gbest_cost

        self.population = new_population
        self.velocity = new_velocity
        self.cost = new_cost
        self.gbest = np.where(np.expand_dims(gbest_filter, axis=-1),
                              new_population[cbest_index],
                              self.gbest)
        self.gbest_cost = np.where(gbest_filter,
                                   cbest_cost,
                                   self.gbest_cost)
        self.pbest = np.where(np.expand_dims(pbest_filters, axis=-1),
                              new_population,
                              self.pbest)
        self.pbest_cost = np.where(pbest_filters,
                                   new_cost,
                                   self.pbest_cost)
