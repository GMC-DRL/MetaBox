import scipy.stats as stats
from .madde_population import MadDE_Population

import numpy as np

class MadDE():
    def __init__(self, config):

        self.__dim = config.dim
        self.__p = 0.18
        self.__PqBX = 0.01
        self.__F0 = 0.2
        self.__Cr0 = 0.2
        self.__pm = np.ones(3) / 3
        self.__npm = 2
        self.__hm = 10
        self.__Nmin = 4
        self.__Nmax = self.__npm * self.__dim * self.__dim
        # self.__Nmax = 200
        self.__H = self.__hm * self.__dim
        # self.__H=100
        self.__FEs = 0
        self.__MaxFEs = config.maxFEs
        
        
        self.config = config

    def __ctb_w_arc(self, group, best, archive, Fs):
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

    def __ctr_w_arc(self, group, archive, Fs):
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

    def __weighted_rtb(self, group, best, Fs, Fas):
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

    def __binomial(self, x, v, Crs):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def __sort(self):
        # new index after sorting
        ind = np.argsort(self.population.c_cost)
        

        self.population.reset_order(ind)

    def __update_archive(self, old_id):
        if self.__archive.shape[0] < self.__NA:
            self.__archive = np.append(self.__archive, self.population.current_position[old_id]).reshape(-1, self.__dim)
        else:
            self.__archive[np.random.randint(self.__archive.shape[0])] = self.population.current_position[old_id]

    def __mean_wL(self, df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # randomly choose step length nad crossover rate from MF and MCr
    def __choose_F_Cr(self):
        # generate Cr can be done simutaneously
        gs = self.__NP
        ind_r = np.random.randint(0, self.__MF.shape[0], size=gs)  # index
        C_r = np.minimum(1, np.maximum(0, np.random.normal(loc=self.__MCr[ind_r], scale=0.1, size=gs)))
        # as for F, need to generate 1 by 1
        cauchy_locs = self.__MF[ind_r]
        F = stats.cauchy.rvs(loc=cauchy_locs, scale=0.1, size=gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        return C_r, np.minimum(1, F)

    # update MF and MCr, join new value into the set if there are some successful changes or set it to initial value
    def __update_M_F_Cr(self, SF, SCr, df):
        if SF.shape[0] > 0:
            mean_wL = self.__mean_wL(df, SF)
            self.__MF[self.__k] = mean_wL
            mean_wL = self.__mean_wL(df, SCr)
            self.__MCr[self.__k] = mean_wL
            self.__k = (self.__k + 1) % self.__MF.shape[0]
        else:
            self.__MF[self.__k] = 0.5
            self.__MCr[self.__k] = 0.5

    def init_population(self, problem):

        self.problem = problem
        self.min_x = problem.lb
        self.max_x = problem.ub

        self.__NP = self.__Nmax
        self.__NA = int(2.3 * self.__NP)
        

        self.population=MadDE_Population(self.__dim,self.__NP,self.min_x,self.max_x,self.__MaxFEs, problem)
        self.population.reset()
        self.__FEs = self.__NP
        self.__archive = np.array([])
        self.__MF = np.ones(self.__H) * self.__F0
        self.__MCr = np.ones(self.__H) * self.__Cr0
        self.__k = 0
        # self.gbest = np.min(self.population.c_cost)
        return self.population


    def update(self, action):
        
        if action.get('skip_step') is not None:
            skip_step=action['skip_step']
        elif action.get('fes') is not None:
            step_fes=action['fes']
            next_fes=self.population.cur_fes+step_fes
        else:
            assert True, 'action error!!'
        

        step_end=self.population.cur_fes>=self.__MaxFEs
        if action.get('fes') is not None:
            step_end = (step_end or (self.population.cur_fes>=next_fes))
        step=0
        
        while not step_end:
            self.__sort()
            NP, dim = self.__NP, self.__dim
            q = 2 * self.__p - self.__p * self.__FEs / self.__MaxFEs
            Fa = 0.5 + 0.5 * self.__FEs / self.__MaxFEs
            Cr, F = self.__choose_F_Cr()
            mu = np.random.choice(3, size=NP, p=self.__pm)
            p1 = self.population.current_position[mu == 0]
            p2 = self.population.current_position[mu == 1]
            p3 = self.population.current_position[mu == 2]
            pbest = self.population.current_position[:max(int(self.__p * NP), 2)]
            qbest = self.population.current_position[:max(int(q * NP), 2)]
            Fs = F.repeat(dim).reshape(NP, dim)
            v1 = self.__ctb_w_arc(p1, pbest, self.__archive, Fs[mu == 0])
            v2 = self.__ctr_w_arc(p2, self.__archive, Fs[mu == 1])
            v3 = self.__weighted_rtb(p3, qbest, Fs[mu == 2], Fa)
            v = np.zeros((NP, dim))
            v[mu == 0] = v1
            v[mu == 1] = v2
            v[mu == 2] = v3

            v[v < self.min_x] = (v[v < self.min_x] + self.min_x) / 2
            v[v > self.max_x] = (v[v > self.max_x] + self.max_x) / 2
            rvs = np.random.rand(NP)
            Crs = Cr.repeat(dim).reshape(NP, dim)
            u = np.zeros((NP, dim))
            if np.sum(rvs <= self.__PqBX) > 0:
                qu = v[rvs <= self.__PqBX]
                if self.__archive.shape[0] > 0:
                    qbest = np.concatenate((self.population.current_position, self.__archive), 0)[
                            :max(int(q * (NP + self.__archive.shape[0])), 2)]
                cross_qbest = qbest[np.random.randint(qbest.shape[0], size=qu.shape[0])]
                qu = self.__binomial(cross_qbest, qu, Crs[rvs <= self.__PqBX])
                u[rvs <= self.__PqBX] = qu
            bu = v[rvs > self.__PqBX]
            bu = self.__binomial(self.population.current_position[rvs > self.__PqBX], bu, Crs[rvs > self.__PqBX])
            u[rvs > self.__PqBX] = bu
            
            ncost=self.population.get_costs(u)
            self.__FEs += NP

            optim = np.where(ncost < self.population.c_cost)[0]
            for i in optim:
                self.__update_archive(i)
            SF = F[optim]
            SCr = Cr[optim]
            df = np.maximum(0, self.population.c_cost - ncost)
            self.__update_M_F_Cr(SF, SCr, df[optim])
            count_S = np.zeros(3)
            for i in range(3):
                count_S[i] = np.mean(df[mu == i] / self.population.c_cost[mu == i])
            if np.sum(count_S) > 0:
                self.__pm = np.maximum(0.1, np.minimum(0.9, count_S / np.sum(count_S)))
                self.__pm /= np.sum(self.__pm)
            else:
                self.__pm = np.ones(3) / 3

            self.population.update(u,ncost, filter_survive=True)
            

            self.__NP = int(np.round(self.__Nmax + (self.__Nmin - self.__Nmax) * self.__FEs / self.__MaxFEs))
            self.__NP = max(self.__NP, self.__Nmin)
            self.__NA = int(2.3 * self.__NP)
            self.__sort()
            
            self.population.reset_popsize(self.__NP)

            self.__archive = self.__archive[:self.__NA]

            step+=1
            if action.get('fes') is not None:
                if self.population.cur_fes>=next_fes or self.population.cur_fes>=self.__MaxFEs:
                    step_end=True
                    break
            elif action.get('skip_step') is not None:
                if step>=skip_step:
                    step_end=True
                    break

        return self.population,0,self.__FEs>=self.__MaxFEs,{}
        
        

