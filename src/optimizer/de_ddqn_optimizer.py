import numpy as np
from collections import deque
from optimizer.learnable_optimizer import Learnable_Optimizer
from optimizer.operators import rand_1_single, rand_2_single, rand_to_best_2_single, cur_to_rand_1_single, binomial, clipping


class DE_DDQN_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.F = 0.5
        config.Cr = 1.0
        config.NP = 100
        config.gen_max = 10
        config.W = 50
        self.__config = config

        self.__F = config.F
        self.__Cr = config.Cr
        self.__NP = config.NP
        self.__maxFEs = config.maxFEs
        self.__gen_max = config.gen_max
        self.__W = config.W
        self.__dim = config.dim
        self.__dim_max = config.dim
        # records
        self.__gen = None  # 记录当前第几代
        self.__pointer = None    # 记录当前代中将要更新的个体index
        self.__stagcount = None  # stagnation counter
        self.__X = None  # 种群
        self.__cost = None  # 当前种群各个个体的适应度
        self.__X_gbest = None  # 全局最优个体
        self.__c_gbest = None  # 全局最优cost
        self.__c_gworst = None  # 全局最差cost
        self.__X_prebest = None  # 父代中的最优个体
        self.__c_prebest = None  # 父代中的最优cost
        self.__OM = None  # 记录四种变异策略的近gen_max代的OM1~OM4的数据
        self.__N_succ = None  # 记录四种变异策略的近gen_max代的OM1~OM4的有效步数(OMx>0即有效)
        self.__N_tot = None  # 记录四种变异策略的近gen_max代的应用次数
        self.__OM_W = None  # 记录各变异策略共W个的最优个体，元素为六元组[strategy, OM1, OM2, OM3, OM4, cost]
        self.__r = None     # random indexes used for generating states and mutation
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def init_population(self, problem):
        # population initialization
        self.__X = np.random.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb
        if problem.optimum is None:
            self.__cost = problem.eval(self.__X)
        else:
            self.__cost = problem.eval(self.__X) - problem.optimum
        # reset records
        self.fes = self.__NP
        self.__gen = 0
        self.__pointer = 0
        self.__stagcount = 0
        self.__X_gbest = self.__X[np.argmin(self.__cost)]
        self.__c_gbest = np.min(self.__cost)
        self.__c_gworst = np.max(self.__cost)
        self.__X_prebest = self.__X[np.argmin(self.__cost)]
        self.__c_prebest = np.min(self.__cost)
        self.__OM = [[], [], [], []]  # 记录四种变异策略的近gen_max代的OM1~OM4的数据
        self.__N_succ = [[], [], [], []]  # 记录四种变异策略的近gen_max代的OM1~OM4的有效步数(OMx>0即有效)
        self.__N_tot = []  # 记录四种变异策略的近gen_max代的应用次数
        self.__OM_W = []  # 记录各变异策略共W个的最优个体，元素为六元组[strategy, OM1, OM2, OM3, OM4, cost]
        for op in range(4):
            self.__N_tot.append(deque(maxlen=self.__gen_max))
            for m in range(4):
                self.__OM[op].append(deque(maxlen=self.__gen_max))
                self.__N_succ[op].append(deque(maxlen=self.__gen_max))
        self.log_index = 1
        self.cost = [self.__c_gbest]
        return self.__get_state(problem)

    def __get_state(self, problem):
        max_dist = np.linalg.norm(np.array([problem.ub - problem.lb]).repeat(self.__dim), 2)
        features = np.zeros(99)
        features[0] = (self.__cost[self.__pointer] - self.__c_gbest) / (self.__c_gworst - self.__c_gbest)
        features[1] = (np.mean(self.__cost) - self.__c_gbest) / (self.__c_gworst - self.__c_gbest)
        features[2] = np.std(self.__cost) / ((self.__c_gworst - self.__c_gbest) / 2)
        features[3] = (self.__maxFEs - self.fes) / self.__maxFEs
        features[4] = self.__dim / self.__dim_max
        features[5] = self.__stagcount / self.__maxFEs
        self.__r = np.random.randint(0, self.__NP, 5)
        for j in range(0, 5):  # features[6] ~ features[10]
            features[j + 6] = np.linalg.norm(self.__X[self.__pointer] - self.__X[self.__r[j]], 2) / max_dist
        features[11] = np.linalg.norm(self.__X[self.__pointer] - self.__X_prebest, 2) / max_dist
        for j in range(0, 5):  # features[12] ~ features[16]
            features[j + 12] = (self.__cost[self.__pointer] - self.__cost[self.__r[j]]) / (self.__c_gworst - self.__c_gbest)
        features[17] = (self.__cost[self.__pointer] - self.__c_prebest) / (self.__c_gworst - self.__c_gbest)
        features[18] = np.linalg.norm(self.__X[self.__pointer] - self.__X_gbest, 2) / max_dist
        i = 19
        for op in range(4):  # 对应四种变异操作op
            for m in range(4):  # m = 1,2,3,4对应四种OM m
                for g in range(min(self.__gen_max, self.__gen)):
                    if self.__N_tot[op][g] > 0:
                        features[i] += self.__N_succ[op][m][g] / self.__N_tot[op][g]  # features[19] ~ features[34]
                i = i + 1
        for op in range(4):
            sum_N_tot = 0
            for g in range(min(self.__gen_max, self.__gen)):
                sum_N_tot += self.__N_tot[op][g]
            for m in range(4):
                for g in range(min(self.__gen_max, self.__gen)):
                    for k in range(self.__N_succ[op][m][g]):
                        features[i] += self.__OM[op][m][g][k]
                if sum_N_tot > 0:
                    features[i] = features[i] / sum_N_tot  # features[35] ~ features[50]
                i = i + 1
        if self.__gen >= 2:
            for op in range(4):
                for m in range(4):
                    # features[51] ~ features[66]
                    if self.__N_tot[op][0] - self.__N_tot[op][1] != 0 and self.__N_succ[op][m][0] > 0 and self.__N_succ[op][m][1] > 0:
                        features[i] = (np.max(self.__OM[op][m][0]) - np.max(self.__OM[op][m][1])) / (np.max(self.__OM[op][m][1]) * np.abs(self.__N_tot[op][0] - self.__N_tot[op][1]))
                    i = i + 1
        else:
            i = 67
        for op in range(4):
            for m in range(4):
                for g in range(min(self.__gen_max, self.__gen)):
                    if self.__N_succ[op][m][g] > 0:
                        features[i] += np.max(self.__OM[op][m][g])  # features[67] ~ features[82]
                i = i + 1
        for w in range(min(self.__W, len(self.__OM_W))):
            for m in range(4):
                features[i + self.__OM_W[w][0] * 4 + m] += self.__OM_W[w][m + 1]  # features[83] ~ features[98]
        return features

    def update(self, action, problem):
        if self.__pointer == 0:  # 进入新的一代
            # 更新cbest
            self.__X_prebest = self.__X_gbest
            self.__c_prebest = self.__c_prebest
            # 更新代数
            self.__gen = self.__gen + 1
            # 为OM、N_succ、N_tot插入新的一代
            for op in range(4):
                self.__N_tot[op].appendleft(0)
                for m in range(4):
                    self.__OM[op][m].appendleft(list())
                    self.__N_succ[op][m].appendleft(0)
        # mutation  ['rand/1', 'rand/2', 'rand-to-best/2', 'cur-to-rand/1']
        if action == 0:
            donor = rand_1_single(self.__X, self.__F, self.__pointer, self.__r)
        elif action == 1:
            donor = rand_2_single(self.__X, self.__F, self.__pointer, self.__r)
        elif action == 2:
            donor = rand_to_best_2_single(self.__X, self.__X_gbest, self.__F, self.__pointer, self.__r)
        elif action == 3:
            donor = cur_to_rand_1_single(self.__X, self.__F, self.__pointer, self.__r)
        else:
            raise ValueError('Action error')
        # BC
        donor = clipping(donor, problem.lb, problem.ub)
        # crossover
        trial = binomial(self.__X[self.__pointer], donor, self.__Cr)
        # get the cost of the trial vector
        if problem.optimum is None:
            trial_cost = problem.eval(trial)
        else:
            trial_cost = problem.eval(trial) - problem.optimum
        self.fes += 1
        # compute reward
        reward = max(self.__cost[self.__pointer] - trial_cost, 0)
        # 更新OM, N_succ, N_tot, OM_W等记录
        self.__N_tot[action][0] += 1
        om = np.zeros(4)
        om[0] = self.__cost[self.__pointer] - trial_cost
        om[1] = self.__c_prebest - trial_cost
        om[2] = self.__c_gbest - trial_cost
        om[3] = np.median(self.__cost) - trial_cost
        for m in range(4):
            if om[m] > 0:
                self.__N_succ[action][m][0] += 1
                self.__OM[action][m][0].append(om[m])
        # 更新OM_W
        if len(self.__OM_W) >= self.__W:  # OM_W已满，需要先淘汰一个
            found = False  # 标识OM_W中是否有找到该strategy的六元组
            for i in range(len(self.__OM_W)):
                if self.__OM_W[i][0] == action:
                    found = True  # 找到了该strategy的六元组（最先进去的一个，FIFO）
                    del self.__OM_W[i]  # 删除之
                    break
            if not found:  # OM_W中没有该strategy的六元组
                del self.__OM_W[np.argmax(np.array(self.__OM_W)[:, 5])]  # 淘汰适应值最大的六元组
        self.__OM_W.append([action, om[0], om[1], om[2], om[3], trial_cost])  # 尾插新的六元组
        # 更新stagcount
        if trial_cost >= self.__c_gbest:
            self.__stagcount += 1
        # selection
        if trial_cost <= self.__cost[self.__pointer]:  # better than its parent
            self.__cost[self.__pointer] = trial_cost
            self.__X[self.__pointer] = trial
            # 更新gbest、cbest记录
            if trial_cost <= self.__c_gbest:  # better than the global best
                self.__c_gbest = trial_cost
                self.__X_gbest = trial
        # 更新gworst记录
        if trial_cost > self.__c_gworst:
            self.__c_gworst = trial_cost
        # 更新指针
        self.__pointer = (self.__pointer + 1) % self.__NP

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__c_gbest)
        # 检查是否已done
        if problem.optimum is None:
            is_done = (self.fes >= self.__maxFEs)
        else:
            is_done = (self.fes >= self.__maxFEs or self.__c_gbest <= 1e-8)
        # get next state
        next_state = self.__get_state(problem)

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__c_gbest
            else:
                self.cost.append(self.__c_gbest)
        return next_state, reward, is_done
