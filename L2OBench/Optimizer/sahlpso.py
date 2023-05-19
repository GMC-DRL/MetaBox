from optimizer.basic_optimizer import basic_optimizer
from optimizer.operators import clipping
import numpy as np
# from trainer import construct_problem_set


# 种子非常重要
# Logger需要记录的：
#     Train:
#         论文里需要展示的baselines们需要记录在一起的：
#             1. 利用cost来计算的随着学习得到的return（2张图）
#                 X：learning step，Y：gbest
#                 三种模式 可以指定 只看数据集整体的/单个问题的/both
#                 按照学习次数来划定记录的区间
#                 在training的过程中就可以记录了 是对trained problem来说的
#
#             2. Return (走一个episode记一次)
#
#     Test:
#         图：
#         1. X： FES
#             1.1 每一个问题/问题集上的cost曲线（可带方差轮廓线），Y:每一个问题的用logf(x), 问题及上的用归一化的平均， agent之间比
#
#         2. X：agent
#             2.1 Y: mean[(f(x) - optim)/(baseline - optim), problem] ,  baseline使用random search， histgram， 不同agent在数据集上的优劣
#
#         3. X: problem
#             3.1 Y: (f(x) - optim)/(baseline - optim) ，baseline使用random search， histgram， 同一个agent在数据集各个问题上的优劣
#
#
#
#         表：
#           1. T0 T1 T2表，agents行 （T0 T1 T2 (T2-T1)/T0）列
#           2. Worst Best Median Mean Std(每個agent在不同function)
#           3. 大表 （AM的样式）

#
#
#
#
#
#
#
#
#
# Train:
#
#
# Tester:
#     def test():
#         config.n_runs = 51
#         optimizer
#         testset
#         logger
#         for n_runs:
#             for p in tesset:
#                 info = optimizer.run_episode(p)
#                 logger.log_ssssssssssss(info)
#
# class Sphere():
#     def __init__(self):
#         pass
#
#     def eval(self,x):
#         return np.sum(x ** 2, -1)

#
# class Random(basic_optimizer):
#     def __init__(self,config):
#         super(Random, self).__init__(config)
#
#     def run_episode(self, problem):
#         X = -5 + 10 * np.random.rand(20000, 10)
#         if problem.optimum is None:
#             Y = problem.eval(X)
#         else:
#             Y = problem.eval(X) - problem.optimum
#         return {'gbest: ': np.min(Y), 'fes:': 20000}
#
#
# class PSO(basic_optimizer):
#     def __init__(self,config):
#         super(PSO, self).__init__(config)
#
#     def run_episode(self,problem):
#         NP = 50
#         V = -1+ 2 * np.random.rand(NP, 10)
#         X = -5 + 10 * np.random.rand(NP, 10)
#         if problem.optimum is None:
#             f_X = problem.eval(X)
#         else:
#             f_X = problem.eval(X) - problem.optimum
#         pBest, pBest_cost = X.copy(), f_X.copy()
#         gBest = X[np.argmin(f_X)]
#         gBest_cost = np.min(f_X)
#         fes = NP
#         while fes < 20000:
#             V =1 * V + 2 * (pBest - X) + 2 * (gBest - X)
#             V = clipping(V, -2.5, 2.5)
#             X = X + V
#             X = clipping(X, -5, 5)
#             if problem.optimum is None:
#                 f_X = problem.eval(X)
#             else:
#                 f_X = problem.eval(X) - problem.optimum
#             mask = f_X < pBest_cost
#             pBest[mask] = X[mask]
#             pBest_cost[mask] = f_X[mask]
#             gBest_cost = np.min(pBest_cost)
#             gBest = pBest[np.argmin(pBest_cost)]
#             fes += NP
#             if problem.optimum is not None:
#                 if gBest_cost <= 1e-8:
#                     break
#         return {'gbest: ':gBest_cost,'fes:':fes}
#

from . import mutate, boundary_control, crossover

#
# class DE(basic_optimizer):
#     def __init__(self, config):
#         super(DE, self).__init__(config)
#
#     def run_episode(self,problem):
#         NP = 50
#         X = -5 + 10 * np.random.rand(NP, 10)
#         if problem.optimum is None:
#             f_X = problem.eval(X)
#         else:
#             f_X = problem.eval(X) - problem.optimum
#         gBest_cost = np.min(f_X)
#         fes = NP
#         while fes < 20000:
#             U = mutate.rand_1(X, 0.5)
#             U = crossover.binomial(X,U, 0.5)
#             U = boundary_control.clipping(U, -5, 5)
#             if problem.optimum is None:
#                 U_cost = problem.eval(U)
#             else:
#                 U_cost = problem.eval(U) - problem.optimum
#             fes += NP
#             mask = U_cost < f_X
#             X[mask] = U[mask]
#             f_X[mask] = U_cost[mask]
#             gBest_cost = np.min(f_X)
#             if problem.optimum is not None:
#                 if gBest_cost <= 1e-8:
#                     break
#         return {'gbest: ': gBest_cost, 'fes:': fes}


class SAHLPSO(basic_optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dim, self.NP, self.maxFEs, self.lb, self.ub, self.v_max = config.dim, 40, config.maxFEs, -5, 5, 1
        self.H_cr = 5
        self.M_cr = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
        self.H_ls = 15
        self.M_ls = range(1,16)
        self.LP = 5
        self.Lg = 0.2
        self.p=0.2
        self.c1 = 1.49445
        self.log_interval = 400

    def run_episode(self,problem):
        G, fes, NP, H_cr, H_ls = 1, 0, self.NP, self.H_cr, self.H_ls
        V = -self.v_max + 2 * self.v_max * np.random.rand(NP, self.dim)
        X = self.lb + (self.ub - self.lb) * np.random.rand(NP, self.dim)
        if problem.optimum is None:
            f_X = problem.eval(X)
        else:
            f_X = problem.eval(X) - problem.optimum
        w = 0.9 * np.ones(NP)
        fes += NP
        pBest,pBest_cost,A = X.copy(), f_X.copy(),[]
        for i in range(NP):
            A.append([pBest[i]])
        gBest = X[np.argmin(f_X)]
        gBest_cost = np.min(f_X)

        log_index = 1
        cost = [gBest_cost]

        P_cr = np.ones(H_cr) / H_cr
        nf_cr = np.zeros(H_cr)
        ns_cr = np.zeros(H_cr)
        P_ls = np.ones(H_ls) / H_ls
        nf_ls = np.zeros(H_ls)
        ns_ls = np.zeros(H_ls)
        remain_index = np.array(range(NP))
        selected_indiv_index = np.random.permutation(remain_index)[:int(self.Lg * NP)]
        while fes < self.maxFEs and NP >= 4:

            for i in remain_index:
                cr, ls = 0, 0 # start generate exemplar
                cr_index, ls_index = 0,0
                if (G%self.LP) or (G==1):
                    cr_index = np.random.choice(range(H_cr),p=P_cr)
                    cr = self.M_cr[cr_index]
                    ls_index = np.random.choice(range(H_ls),p=P_ls)
                    ls = self.M_ls[ls_index]
                #exploration indiv
                if i in selected_indiv_index:
                    mn = np.random.choice(remain_index,2)
                    m,n=mn[0],mn[1]
                    o = pBest[m] if f_X[m] < f_X[n] else pBest[n]
                    if (G-ls) >= 0:
                        history_pbest = A[i][G-ls-1]
                    else:
                        history_pbest = A[i][-1]
                else:
                    best_p_index = np.argsort(pBest_cost)[:max(1, int(self.p *NP))]
                    o = pBest[np.random.choice(best_p_index)]
                    history_pbest = pBest[i]
                # execute crossover in each dimensional
                mask_crossover = np.random.rand(self.dim) < cr
                e = history_pbest
                e[mask_crossover] = o[mask_crossover]
                if i not in selected_indiv_index:
                    rnd1 = np.random.rand(self.dim)
                    e =  rnd1 * e + (1 - rnd1) * gBest # end generate exemplar

                # update cr and ls selection times
                nf_cr[cr_index] += 1
                nf_ls[ls_index] += 1
                # generate velocity V and BC(V)
                V[i] = w[i] * V[i] + self.c1 * np.random.rand(self.dim) * (e - X[i])
                V[i] = clipping(V[i], -self.v_max, self.v_max)
                # generate new X and BC(X)
                X[i] = clipping(X[i] + V[i], self.lb, self.ub)
                # evaluate X
                if problem.optimum is None:
                    f_X[i] = problem.eval(X[i])
                else:
                    f_X[i] = problem.eval(X[i]) - problem.optimum
                fes += 1
                if f_X[i] < pBest_cost[i]:
                    pBest[i] = X[i]
                    if f_X[i] < gBest_cost:
                        gBest = X[i]
                        gBest_cost = f_X[i]
                    # update cr and ls success times
                    ns_cr[cr_index] += 1
                    ns_ls[ls_index] += 1
                else:
                    rnd2 = np.random.rand()
                    if rnd2 < 0.5:
                        w[i] = 0.7 + 0.1 *np.random.standard_cauchy()
                    else:
                        w[i] = 0.3 + 0.1 * np.random.standard_cauchy()
                    w[i] = np.clip(w[i], 0.2,0.9)
                A[i].append(pBest[i])

                if fes >= log_index * self.log_interval:
                    log_index += 1
                    cost.append(gBest_cost)

                if problem.optimum is None:
                    done = fes >= self.maxFEs
                else:
                    done = fes >= self.maxFEs or gBest_cost <= 1e-8

                if done:
                    return {'cost': cost, 'fes': fes}
            # after a generation update related statics
            if G % self.LP == 0:

                S_cr = np.zeros(H_cr)
                mask = nf_cr != 0
                S_cr[mask] = ns_cr[mask] / nf_cr[mask]
                # update P_cr
                if np.sum(S_cr) == 0 and H_cr < len(self.M_cr):
                    H_cr += 1
                    P_cr = np.ones(H_cr) / H_cr


                else:
                    P_cr = S_cr / np.sum(S_cr)

                # update P_ls
                S_ls = np.zeros(H_ls)
                mask = nf_ls != 0
                S_ls[mask] = ns_ls[mask] / nf_ls[mask]
                if np.sum(S_ls) == 0:
                    P_ls = np.ones(H_ls) / H_ls
                else:
                    P_ls = S_ls / np.sum(S_ls)
                #P_cr = np.ones(H_cr) / H_cr
                # nf_cr = np.zeros(H_cr)
                # ns_cr = np.zeros(H_cr)
                # #P_ls = np.ones(H_ls) / H_ls
                # nf_ls = np.zeros(H_ls)
                # ns_ls = np.zeros(H_ls)

            # population size reduction
            NP_ = round((4 - self.NP) * fes / self.maxFEs + self.NP)
            if NP_ < NP:
                remain_index = np.argsort(pBest_cost)[:NP_]
                NP = NP_
            G += 1
            #print(gBest_cost)

        return {'cost': cost, 'fes': fes}


# if __name__ == '__main__':
#     config = get_config()
#     train_set, test_set = construct_problem_set(config)
#     optimizer = SAHLPSO(config)
#
#     # problem = Sphere()
#     # print(optimizer.run_episode(problem))
#     # np.random.seed(1)
#     # for problem_id, problem in enumerate(train_set):
#     #     info = optimizer.run_episode(problem)  # pbar_info -> dict
#     #     print(f'problem {problem_id}: {info}')
#
#     from optimizer import MadDE
#     optimizer = PSO(config)
#     np.random.seed(1)
#     for problem_id, problem in enumerate(train_set):
#         info = optimizer.run_episode(problem)  # pbar_info -> dict
#         print(f'problem {problem_id}: {info}')
