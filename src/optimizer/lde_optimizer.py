import numpy as np
import torch
from optimizer.learnable_optimizer import Learnable_Optimizer


class LDE_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.__config.NP = 50
        self.__config.BINS = 5
        self.__config.MEMO_SIZE = 5
        self.__config.P_INI = 1
        self.__config.P_NUM_MIN = 2
        self.__config.P_MIN = self.__config.P_NUM_MIN/self.__config.NP
        self.__BATCH_SIZE = 1
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __get_cost(self, batch, pop):
        bs = len(batch)
        cost = []
        for p in range(bs):
            if batch[p].optimum is None:
                cost.append(batch[p].eval(pop[p]))  # [NP]
            else:
                cost.append(batch[p].eval(pop[p]) - batch[p].optimum)  # [NP]
        return np.vstack(cost)

    def __modifyChildwithParent(self, cross_pop, parent_pop, x_max, x_min):
        cro_lb = cross_pop < x_min
        cro_ub = cross_pop > x_max
        no_cro = ~(cro_lb | cro_ub)

        cross_pop = no_cro * cross_pop + cro_lb * (parent_pop + x_min) / 2. + cro_ub * (parent_pop + x_max) / 2.

        return cross_pop

    def __de_crosselect_random_dataset(self, pop, m_pop, fit, cr_vector, nfes, batch):
        batch_size, pop_size, problem_size = pop.shape
        
        # Crossover
        r = np.random.uniform(size=(batch_size, pop_size, problem_size))
        r[np.arange(batch_size)[:, None].repeat(pop_size, axis=1),
        np.arange(pop_size)[None, :].repeat(batch_size, axis=0),
        np.random.randint(low=0, high=problem_size, size=[batch_size, self.__config.NP])] = 0.  # 对每个个体的dim个随机数，随机地取其中一个置0
        cross_pop = np.where(r <= cr_vector[:, :, None].repeat(problem_size, axis=-1), m_pop, pop)

        # Boundary Control
        cross_pop = self.__modifyChildwithParent(cross_pop, pop, batch.ub, batch.lb)

        # Get costs
        cross_fit = self.__get_cost([batch], cross_pop)

        nfes += pop_size

        # Selection
        surv_filters = cross_fit <= fit  # survive_filter is true if the offspring is better than or equal to its parent
        n_pop = np.where(surv_filters[:, :, None].repeat(problem_size, axis=-1), cross_pop, pop)
        n_fit = np.where(surv_filters, cross_fit, fit)

        return n_pop, n_fit, nfes

    def __mulgenerate_pop(self, p, NP, input_dimension, x_min, x_max, same_per_problem):
        if same_per_problem:
            pop = x_min + np.random.uniform(size=(NP, input_dimension)) * (x_max - x_min)
            pop = pop[None, :, :].repeat(p, axis=0)
        else:
            pop = x_min + np.random.uniform(size=(p, NP, input_dimension)) * (x_max - x_min)
        return pop

    def __order_by_f(self, pop, fit):
        batch_size, pop_size = pop.shape[0], pop.shape[1]
        sorted_array = np.argsort(fit, axis=1)
        temp_pop = pop[np.arange(batch_size)[:, None].repeat(pop_size, axis=1), sorted_array]
        temp_fit = fit[np.arange(batch_size)[:, None].repeat(pop_size, axis=1), sorted_array]
        return temp_pop, temp_fit

    def __maxmin_norm(self, a):
        batch_size = a.shape[0]
        normed = np.zeros_like(a)
        for b in range(batch_size):
            if np.max(a[b]) != np.min(a[b]):
                normed[b] = (a[b] - np.min(a[b])) / (np.max(a[b]) - np.min(a[b]))
        return normed

    def __con2mat_current2pbest_Nw(self, mutation_vector, p):
        batch_size, pop_size = mutation_vector.shape[0], mutation_vector.shape[1]
        p_index_array = np.random.randint(0, int(np.ceil(pop_size*p)), size=(batch_size, pop_size))
        mutation_mat = np.zeros((batch_size, pop_size, pop_size))
        for i in range(pop_size):
            mutation_mat[:, i, i] = 1 - mutation_vector[:, i]
            for b in range(batch_size):
                if p_index_array[b, i] != i:
                    mutation_mat[b, i, p_index_array[b, i]] = mutation_vector[b, i]
                else:
                    mutation_mat[b, i, i] = 1
        return mutation_mat

    def __con2mat_rand2pbest_Nw(self, mutation_vector, nfes, MaxFEs):
        #        ( 0.4  -   1  ) * nfes/MAXFE + 1
        p_rate = (self.__config.P_MIN - self.__config.P_INI) * nfes/MaxFEs + self.__config.P_INI
        mutation_mat = self.__con2mat_current2pbest_Nw(mutation_vector, max(0, p_rate))
        return mutation_mat

    def __add_random(self, m_pop, pop, mu):
        batch_size = pop.shape[0]
        r = torch.randint(high=self.__config.NP, size=[batch_size, self.__config.NP, 2])

        # validity checking and modification for r
        pop_index = torch.arange(self.__config.NP)
        for col in range(0, 2):
            while True:
                is_repeated = [torch.eq(r[:, :, col], r[:, :, i]) for i in range(col)]  # 检查当前列与其前面所有列有无重复
                is_repeated.append(torch.eq(r[:, :, col], pop_index))  # 检查当前列是否与该个体编号重复
                repeated_index = torch.nonzero(torch.any(torch.stack(is_repeated), dim=0))  # 获取重复随机数的下标[batch_index, population_index]
                repeated_sum = repeated_index.size(0)  # 重复随机数的个数
                if repeated_sum != 0:
                    r[repeated_index[:, 0], repeated_index[:, 1], col] = torch.randint(high=self.__config.NP,
                                                                                    size=[repeated_sum])  # 重新生成并替换
                else:
                    break
        r = r.numpy()

        batch_index = np.arange(batch_size)[:, None].repeat(self.__config.NP, axis=1)

        mur_pop = m_pop + np.expand_dims(mu, -1).repeat(pop.shape[-1], axis=-1) * (pop[batch_index, r[:, :, 0]] - pop[batch_index, r[:, :, 1]])

        return mur_pop

    def init_population(self, problem):
        self.__pop = self.__mulgenerate_pop(self.__BATCH_SIZE, self.__config.NP, self.__config.dim, problem.lb, problem.ub, True)   # [bs, NP, dim]
        self.__fit = self.__get_cost([problem], self.__pop)
        self.gbest_cost = np.min(self.__fit)
        self.fes = self.__config.NP
        self.log_index = 1
        self.cost = [self.gbest_cost]
        self.__past_histo = (self.__config.NP/self.__config.BINS) * np.ones((self.__BATCH_SIZE, 1, self.__config.BINS))
        return self.__get_feature()

    def get_best(self):
        return self.gbest_cost

    def __get_feature(self):
        self.__pop, self.__fit = self.__order_by_f(self.__pop, self.__fit)  # fitness降序排序
        fitness = self.__maxmin_norm(self.__fit)
        hist_fit = []
        for b in range(self.__BATCH_SIZE):
            hist_fit.append(np.histogram(fitness[b], self.__config.BINS)[0])
        hist_fit = np.vstack(hist_fit)  # [bs, BINS]

        mean_past_histo = np.mean(self.__past_histo, axis=1)   # [bs, BINS]

        # [bs, NP+BINS*2]
        input_net = np.concatenate((fitness, hist_fit, mean_past_histo), axis=1)
        return input_net

    def update(self, action, problem):
        self.__pop, self.__fit = self.__order_by_f(self.__pop, self.__fit)  # fitness降序排序
        fitness = self.__maxmin_norm(self.__fit)

        # sf_cr = np.squeeze(action.cpu().numpy(), axis=0)  # [bs, NP*2]
        sf = action[:, 0:self.__config.NP]  # scale factor [bs, NP]
        cr = action[:, self.__config.NP:2*self.__config.NP]  # crossover rate  [bs, NP]
        sf_mat = self.__con2mat_rand2pbest_Nw(sf, self.fes, self.__config.maxFEs)  # [NP, NP]
        mu_pop = self.__add_random(np.matmul(sf_mat, self.__pop), self.__pop, sf)  # [NP, dim]

        pop_next, fit_next, self.fes = self.__de_crosselect_random_dataset(self.__pop, mu_pop, self.__fit, cr, self.fes, problem)  # DE
        bsf = self.__fit.min(1)
        bsf_next = fit_next.min(1)

        reward = (bsf - bsf_next)/bsf  # reward

        if problem.optimum is None:
            is_done = self.fes >= self.__config.maxFEs
        else:
            is_done = self.fes >= self.__config.maxFEs or np.min(fit_next) <= 1e-8

        self.__pop = pop_next
        self.__fit = fit_next

        hist_fit = []
        for b in range(self.__BATCH_SIZE):
            hist_fit.append(np.histogram(fitness[b], self.__config.BINS)[0])
        hist_fit = np.vstack(hist_fit)  # [bs, BINS]
        self.__past_histo = np.concatenate((self.__past_histo, hist_fit[:, None, :]), axis=1)
        self.gbest_cost = np.min(self.__fit)

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest_cost)
        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.gbest_cost
            else:
                self.cost.append(self.gbest_cost)
        return self.__get_feature(), reward, is_done
