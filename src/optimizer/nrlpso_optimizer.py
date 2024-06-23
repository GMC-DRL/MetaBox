from optimizer.learnable_optimizer import Learnable_Optimizer
import numpy as np
import math



class NRLPSO_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        
        self.NP = 100
        self.k = 5
        self.total_state = 4
        self.w_max = 1
        self.w_min = 0.4
        self.u = 0.6
        self.v = 0.33
        self.v_ratio = 0.1
        self.n_state = 4

        self.__maxFEs = config.maxFEs
        self.__dim = config.dim

        self.cost = None  # a list of costs that need to be maintained by EVERY backbone optimizers
        self.log_index = None
        self.log_interval = config.log_interval
        

    def init_population(self, problem):
        self.pointer = 0
        # init population
        self.__population = np.random.rand(self.NP, self.__dim) * (problem.ub - problem.lb) + problem.lb
        
        self.v_min = -0.1 * (problem.ub - problem.lb)
        self.v_max = -self.v_min
        self.__velocity = np.zeros(shape=(self.NP, self.__dim))
        if problem.optimum is None:
            self.__cost = problem.eval(self.__population)
        else:
            self.__cost = problem.eval(self.__population) - problem.optimum
        self.__pbest_pos = self.__population.copy()
        self.__pbest_cost = self.__cost.copy()

        gbest_index = np.argmin(self.__cost)
        self.__gbest_cost = self.__cost[gbest_index]
        self.__gbest_pos = self.__population[gbest_index]
        
        self.pbest_stag_count = np.zeros((self.NP, ))
        
        self.fes = self.NP
        self.log_index = 1
        self.cost = [self.__gbest_cost]

        self.r_w = np.random.rand()

        # init state
        self.__state = np.random.randint(low=0, high=self.n_state, size=self.NP)
        return self.__state[self.pointer]

    
    def update_construct_neighborhood(self):

        # pbest neb
        pbest_to_every_distance = np.sqrt(np.sum((self.__pbest_pos[None, :] - self.__population[:, None])**2, axis=-1))
        id_index = np.arange(self.NP)

        pbest_to_every_distance[id_index, id_index] = math.inf


        sort_index = np.argsort(pbest_to_every_distance, -1)

        n_index = sort_index[:, :self.k]
        self.pbest_neb_index = n_index
        self.pbest_neb = self.__population[n_index]

        # gbest neb
        gbest_to_every_distance = np.sqrt(np.sum((self.__gbest_pos[None, :] - self.__population)**2, axis=-1))

        sort_index = np.argsort(gbest_to_every_distance, -1)

        n_index = sort_index[:self.k]

        self.gbest_neb = self.__population[n_index]
        self.gbest_neb_index = n_index

    # todo: debug
    def cal_w(self):
        self.r_w = 4 * self.r_w * (1 - self.r_w)

        w = self.u - ((self.fes / self.__maxFEs) * self.r_w * self.w_min + self.v * (self.w_max - self.w_min) * (self.fes / self.__maxFEs))

        return w

    def cal_reward(self, f_new, f_old, ef_new, ef_old):
        cond1 = f_new < f_old
        cond2 = ef_new > ef_old

        r = None
        if cond1 and cond2:
            r = 2
        elif cond1 and not cond2:
            r = 1
        elif not cond1 and cond2:
            r = 0
        elif not cond1 and not cond2:
            r = -2
        return r

    def cal_ef(self, ith):
        self.update_distance()
        return (self.distance[ith] - self.d_min) / (self.d_max - self.d_min)
    
    def update_distance(self):
        p1 = self.__population[None, :]
        p2 = self.__population[:, None]
        distance = np.sqrt(np.sum((p1 - p2)**2, -1))

        self.distance = np.sum(distance, -1) / (self.NP - 1)

        self.d_min = np.min(self.distance)
        self.d_max = np.max(self.distance)

    def cal_cs(self, p1, p2):
        return np.sum(p1 * p2) / (np.sqrt(np.sum(p1**2)) * np.sqrt(np.sum(p2**2)))

    def get_p_b(self, ith):
        r_idx = np.random.randint(0, self.k)
        return self.pbest_neb[ith][r_idx]

    def get_p_a(self):
        r_idx = np.random.randint(0, self.k)
        return self.gbest_neb[r_idx]
    

    def generate_v_vector(self, action, ith, w):
        c1, c2, P1, P2 = None, None, None, None
        r1 = np.random.rand()
        r2 = np.random.rand()

        cur_p = self.__population[ith]

        cs = self.cal_cs(self.__pbest_pos[ith], self.__gbest_pos)
        # exploration
        if action == 0:
            c1 = 2.2
            c2 = 1.8
            if cs < 0:
                P1 = self.__pbest_pos[ith]
                P2 = self.get_p_a()

                self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p) + c2 * r2 * (P2 - cur_p)
            else:
                P1 = self.get_p_b(ith)
                self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p)
        # exploitation
        elif action == 1:
            c1 = 2.1
            c2 = 1.8
            if cs < 0:
                P1 = self.get_p_b(ith)
                P2 = self.__gbest_pos

                self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p) + c2 * r2 * (P2 - cur_p)
            else:
                P2 = self.get_p_a()

                self.__velocity[ith] = w * self.__velocity[ith] + c2 * r2 * (P2 - cur_p)
        # convergence
        elif action == 2:
            c1 = 2
            c2 = 2
            
            if cs < 0:
                P1 = self.__pbest_pos[ith]
                P2 = self.__gbest_pos

                self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p) + c2 * r2 * (P2 - cur_p)
            else:
                P2 = self.__gbest_pos
                self.__velocity[ith] = w * self.__velocity[ith] + c2 * r2 * (P2 - cur_p)
        
        # jumping-out
        elif action == 3:
            c1 = 1.8
            c2 = 2.2
            P1 = self.get_p_b(ith)
            P2 = self.get_p_a()
            r1 = np.random.rand(self.__dim)
            r2 = np.random.rand(self.__dim)
            self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p) + c2 * r2 * (P2 - cur_p)
        
        # clip velocity
        self.__velocity[ith] = np.clip(self.__velocity[ith], self.v_min, self.v_max)

    def cal_cost(self, x, problem):
        self.fes += 1
        if problem.optimum is None:
            cost = problem.eval(x)
        else:
            cost = problem.eval(x) - problem.optimum
        return cost

    def neb_mutation(self, ith, problem):
        # for pbest ith neibourhood
        # find out P1 P2
        distance = np.sqrt(np.sum((self.__pbest_pos[ith][None, :] - self.pbest_neb[ith])**2, axis=-1))
        sort_idx = np.argsort(distance)
        P1 = self.pbest_neb[ith][sort_idx[0]]
        P2 = self.pbest_neb[ith][sort_idx[-1]]

        P3 = self.__pbest_pos[ith] + np.random.rand(self.__dim) * (P1 - P2)
        cost = self.cal_cost(P3, problem)
        if cost < self.__pbest_cost[ith]:
            self.__pbest_pos[ith] = P3
            self.__pbest_cost[ith] = cost
        else:
            # replace P2 by P3
            P2_idx = self.pbest_neb_index[ith][sort_idx[-1]]
            self.__population[P2_idx] = P3
            self.__cost[P2_idx] = cost


        # for gbest neibourhood
        distance = np.sqrt(np.sum((self.__gbest_pos[None, :] - self.gbest_neb)**2, axis=-1))
        sort_idx = np.argsort(distance)
        P1 = self.gbest_neb[sort_idx[0]]
        P2 = self.gbest_neb[sort_idx[-1]]

        P3 = self.__gbest_pos + np.random.rand(self.__dim) * (P1 - P2)
        cost = self.cal_cost(P3, problem)
        if cost < self.__gbest_cost:
            self.__gbest_pos = P3
            self.__gbest_cost = cost
        else:
            # replace P2 by P3
            P2_idx = self.gbest_neb_index[sort_idx[-1]]
            self.__population[P2_idx] = P3
            self.__cost[P2_idx] = cost

    def update(self, action, problem):
        if self.pointer == 0:
            self.update_construct_neighborhood()
            # dynamic w
            self.w = self.cal_w()

        # generate velocity vector
        self.generate_v_vector(action, self.pointer, self.w)

        ef_old = self.cal_ef(self.pointer)
        # update position
        self.__population[self.pointer] = self.__population[self.pointer] + self.__velocity[self.pointer]
        self.__population[self.pointer] = np.clip(self.__population[self.pointer], problem.lb, problem.ub)

        ef_new = self.cal_ef(self.pointer)
        f_old = self.__cost[self.pointer]

        
        f_new = self.cal_cost(self.__population[self.pointer], problem)
        
        reward = self.cal_reward(f_new, f_old, ef_new, ef_old)
        
        self.__cost[self.pointer] = f_new
        # perform neighborhood diffenent mutation
        if f_new < self.__pbest_cost[self.pointer]:
            self.__pbest_pos[self.pointer] = self.__population[self.pointer]
            self.pbest_stag_count[self.pointer] = 0
        else:
            self.pbest_stag_count[self.pointer] += 1

        if self.pbest_stag_count[self.pointer] >= 2:
            self.neb_mutation(self.pointer, problem)

        if f_new < self.__gbest_cost:
            self.__gbest_cost = f_new
            self.__gbest_pos = self.__population[self.pointer]

        self.__state[self.pointer] = action
        self.pointer = (self.pointer + 1) % self.NP


        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__gbest_cost)
        # if an episode should be terminated
        if problem.optimum is None:
            is_done = self.fes >= self.__maxFEs
        else:
            is_done = (self.fes >= self.__maxFEs or self.__gbest_cost <= 1e-8)
        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__gbest_cost
            else:
                self.cost.append(self.__gbest_cost)

        return self.__state[self.pointer], reward, is_done

