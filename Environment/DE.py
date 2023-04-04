import torch
from basic_environment import PBO_Env


class DE_env(PBO_Env):
    """
    Original DE
    """
    def __init__(self,
                 problem_instance,
                 dim,
                 lower_bound,
                 upper_bound,
                 population_size,
                 FEs,
                 mutate_strategy=0,
                 boundary_ctrl_method='clipping',
                 reward_definition=0.,
                 ):
        super().__init__(problem_instance, dim, lower_bound, upper_bound, population_size, FEs)
        self.mutate_strategy = mutate_strategy  # 0: rand/1;  1: rand-to-best/1; 2: rand-to-best/2
        self.boundary_ctrl_method = boundary_ctrl_method
        self.reward_definition = reward_definition
        self.fes = 0

    def init_population(self):
        """
        Generate population randomly and compute costs.
        """
        self.population = torch.rand(size=(self.NP, self.dim)) * (self.ub - self.lb) + self.lb  # [lb, ub]
        self.cost = self.problem_instance.func(self.population)

    def reset(self):
        # reset all counters
        self.fes = 0
        self.init_population()

    def reinit_population(self):  # 保留bsf个体，其余重新随机生成
        new_pop = torch.rand(size=(self.NP, self.dim)) * (self.ub - self.lb) + self.lb  # [lb, ub]
        new_pop[self.cost.argmin()] = self.population[self.cost.argmin()]
        self.population = new_pop
        self.cost = self.problem_instance.func(self.population)

    def get_feature(self):
        pass

    def generate_random_int(self, cols, check_index=True):
        """
        :param cols: the number of random int generated for each individual.

        :param check_index: whether to check the population indexes appeal in their own ''cols'' elements for each individual.
               For example, if ''check_index'' is True, 0 won't appeal in any element in r[:, 0, :].

        :return: a random int matrix in shape[''NP'', ''cols''], and elements are in a range of [0, ''population_size''-1].
                 The ''cols'' elements at dimension[2] are different from each other.
        """

        r = torch.randint(high=self.NP, size=[self.NP, cols])

        # validity checking and modification for r
        if check_index:
            pop_index = torch.arange(self.NP)
            for col in range(0, cols):
                while True:
                    is_repeated = [torch.eq(r[:, col], r[:, i]) for i in range(col)]  # 检查当前列与其前面所有列有无重复
                    is_repeated.append(torch.eq(r[:, col], pop_index))  # 检查当前列是否与该个体编号重复
                    repeated_index = torch.nonzero(
                        torch.any(torch.stack(is_repeated), dim=0))  # 获取重复随机数的下标[population_index]
                    repeated_sum = repeated_index.size(0)  # 重复随机数的个数
                    if repeated_sum != 0:
                        r[repeated_index[:, 0], col] = torch.randint(high=self.NP, size=[repeated_sum])  # 重新生成并替换
                    else:
                        break
        else:
            for col in range(1, cols):
                while True:
                    is_repeated = [torch.eq(r[:, col], r[:, i]) for i in range(col)]
                    repeated_index = torch.nonzero(torch.any(torch.stack(is_repeated), dim=0))
                    repeated_sum = repeated_index.size(0)
                    if repeated_sum != 0:
                        r[repeated_index[:, 0], col] = torch.randint(high=self.NP, size=[repeated_sum])
                    else:
                        break
        return r

    def mutate(self, F):
        """
        :param F: An array of Mutation factor of shape[NP].
        :return: Population mutated.
        """
        F = F.unsqueeze(-1).repeat(1, self.dim)
        x = self.population
        y = None

        if self.mutate_strategy == 0:  # rand/1
            r = self.generate_random_int(3)
            y = x[r[:, 0]] + F * (x[r[:, 1]] - x[r[:, 2]])

        elif self.mutate_strategy == 1:  # rand-to-best/1
            r = self.generate_random_int(3)
            y = x[r[:, 0]] + F * \
                     (x[self.cost.argmin()].unsqueeze(0).repeat(self.NP, 1) - x[r[:, 0]] +
                      x[r[:, 1]] - x[r[:, 2]])

        elif self.mutate_strategy == 2:  # rand-to-best/2
            r = self.generate_random_int(5)
            y = x[r[:, 0]] + F * \
                     (x[self.cost.argmin()].unsqueeze(0).repeat(self.NP, 1) - x[r[:, 0]] +
                      x[r[:, 1]] - x[r[:, 2]] +
                      x[r[:, 3]] - x[r[:, 4]])

        elif self.mutate_strategy == 3:  # current-to-best/1
            r = self.generate_random_int(2)
            y = x + F * \
                     (x[self.cost.argmin()].unsqueeze(0).repeat(self.NP, 1) - x +
                      x[r[:, 0]] - x[r[:, 1]])

        elif self.mutate_strategy == 4:  # current-to-best/2
            r = self.generate_random_int(4)
            y = x + F * \
                     (x[self.cost.argmin()].unsqueeze(0).repeat(self.NP, 1) - x +
                      x[r[:, 0]] - x[r[:, 1]] +
                      x[r[:, 2]] - x[r[:, 3]])

        return y

    def boundary_ctrl(self, x):
        y = None
        if self.boundary_ctrl_method == 'clipping':
            y = torch.clamp(x, self.lb, self.ub)

        elif self.boundary_ctrl_method == 'random':
            cro_bnd = (x < self.lb) | (x > self.ub)  # cross_boundary
            y = ~cro_bnd * x + cro_bnd * torch.rand(size=[self.NP, self.dim]) * (self.ub - self.lb) + self.lb

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

    def crossover(self, Cr, x):
        """
        :param Cr: An array of crossover rate of shape[NP].
        :param x: The mutated population before crossover.
        :return: Population after crossover.
        """
        r = torch.rand(self.NP, self.dim)
        r[torch.arange(self.NP), torch.randint(high=self.dim, size=[self.NP])] = 0.  # 对每个个体的dim个随机数，随机地取其中一个置0
        y = torch.where(torch.tensor(r <= Cr.unsqueeze(-1).repeat(1, self.dim)),
                        x,
                        self.population)
        return y

    def get_reward(self):
        # reward = None

        # if self.reward_definition == 0.:
        #     reward = (population['c_bsf'] - new_population['c_bsf']) / new_population['c_wsf'] * 100
        #
        # elif self.reward_definition == 0.1:
        #     reward = (population['c_bsf'] - new_population['c_bsf']) / population['cost'].max(1)[0]
        #
        # elif self.reward_definition == 0.2:
        #     reward = (population['cost'].max(1)[0] - new_population['cost'].max(1)[0]) / population['cost'].max(1)[0]

        # definition 1
        # reward = (population['cost'] - new_population['cost']).mean(1) / new_population['c_wsf']

        # definition 1.5
        # reward = (population['cost'] - new_population['cost']).mean(1) / population['c_bsf']

        # definition 1.6
        # reward = (population['cost'] - new_population['cost']).mean(1) / population['cost'].max(1)[0]

        # definition 2
        # reward = torch.where(new_population['cost'] < population['cost'],
        #                      torch.ones(bs, ps).to(old_pos.device),
        #                      torch.zeros(bs, ps).to(old_pos.device))
        # reward = torch.where(new_population['cost'] < population['c_bsf'].unsqueeze(-1).repeat(1, ps),
        #                      torch.ones(bs, ps).to(old_pos.device) * 10,
        #                      reward).mean(1)

        # elif self.reward_definition == 3.:
        #     reward = - new_population['c_bsf'] / new_population['c_wsf']  # - current_best(best-so-far) / worst-so-far
        #
        # elif self.reward_definition == 3.1:
        #     reward = - new_population['cost'].max(1)[0] / new_population['c_wsf']  # - current_worst / worst-so-far
        #
        # elif self.reward_definition == 4.:
        #     new_population['bonus'] = (new_population['init_best'] - new_population['c_bsf']) / new_population[
        #         'init_best']
        #     reward = (new_population['bonus'] + population['bonus']) * (
        #                 new_population['bonus'] - population['bonus']) * 100
        pass

    def step(self, action):  # todo: 兼容选算子而不是选参数的agent
        """
        :param action:
        """
        mutated = self.mutate(action[0])
        mutated = self.boundary_ctrl(mutated)
        trials = self.crossover(action[1], mutated)
        # Selection
        trials_cost = self.problem_instance.func(trials)
        surv_filters = torch.tensor(trials_cost <= self.cost)  # survive_filter is true if the offspring is better than or equal to its parent
        self.population = torch.where(surv_filters.unsqueeze(-1).repeat(1, self.dim), trials, self.population)
        self.cost = torch.where(surv_filters, trials_cost, self.cost)

        self.fes += self.NP
        # todo: get_feature的过程应该放在env里还是agent里？
        # 放在agent里，agent的state就是env的feature，
        state = self.get_feature()
        reward = self.get_reward()
        is_done = self.fes >= self.FEs or self.cost.min() <= 1e-8
        return state, reward, is_done
