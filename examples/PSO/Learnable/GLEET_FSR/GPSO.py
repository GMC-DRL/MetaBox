'''implementation of GPSO as an environment for DRL usage'''
# todo 实现一个能返回ppo所需要的meta info的环境 可以从PSO改来
import numpy as np
from L2OBench.Optimizer import Learnable_Optimizer, clipping


class GPSO_numpy(Learnable_Optimizer):
    def __init__(self,config):
        super().__init__(config.dim,config.lower_bound,config.upper_bound,config.population_size,config.max_fes)

        self.config = config
        self.dim = self.config.dim
        self.w_decay = self.config.w_decay
        if self.w_decay:
            self.w = 0.9
        else:
            self.w = 0.729
        self.c = self.config.c
        self.max_velocity = self.config.max_velocity
        self.max_x = self.config.max_x
        self.NP = self.config.NP
        self.n_pop = config.n_patch

        self.no_improve = np.zeros(self.n_pop)
        self.per_no_improve = np.zeros((self.n_pop, self.NP))
        self.fes = 0
        self.max_fes = self.config.max_fes
        self.max_dist = np.sqrt((2 * self.config.max_x) ** 2 * self.config.dim)
        self.is_done = False

        self.name = 'GPSO'
        # print(f'GPSO with {self.dim} dims .')

    # initialize GPSO environment
    def init_population(self,problem):
        # randomly generate the position and velocity
        self.population = np.random.rand(self.n_pop, self.NP, self.dim) * (self.ub - self.lb) + self.lb  # [lb, ub]
        self.velocity = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.n_pop, self.NP, self.dim))
        self.fes = 0

        # get the initial cost
        self.init_cost = self.get_costs(problem, self.population)  # n_pop, ps
        self.pre_cost = self.init_cost.copy()
        self.cost = self.init_cost.copy()

        # find out the gbest_val
        self.pre_gbest = self.cost.min(1)  # n_pop
        # self.gbest_cost = self.cost.min(1).copy()  # n_pop
        self.gbest_val = self.cost.min(1)  # n_pop
        self.gbest_index = self.cost.argmin(1)  # n_pop
        self.gbest_position = self.population[np.arange(self.n_pop), self.gbest_index].copy()  # n_pop, dim

        self.pbest = self.population.copy()
        self.pbest_cost = self.cost.copy()

        # record
        self.max_cost = np.min(self.cost)

        # store all the information of the paraticles
        self.particles = {'current_position': self.population.copy(),    # n_pop, ps, dim
                          'c_cost': self.cost.copy(),   # n_pop, ps
                          'pbest_position': self.population.copy(),      # n_pop, ps, dim
                          'pbest': self.cost.copy(),    # n_pop, ps
                          'gbest_position': self.gbest_position.copy(),  # n_pop, dim
                          'gbest_val': self.gbest_val,  # n_pop
                          'velocity': self.velocity.copy(),  # n_pop, ps, dim
                          'gbest_index': self.gbest_index  # n_pop
                          }
        if self.w_decay:
            self.w = 0.9

        self.no_improve -= self.no_improve
        self.fes -= self.fes
        self.per_no_improve -= self.per_no_improve


    # calculate costs of solutions
    def get_costs(self,problem,position):
        self.fes += self.NP
        cost = problem.eval(position) - problem.optimum
        return cost




    def update(self,problem,action=None):
        self.is_done = False

        # record the gbest_val in the begining
        self.pre_gbest = self.particles['gbest_val']

        # linearly decreasing the coefficient of inertia w
        if self.w_decay:
            self.w -= 0.5 / (self.max_fes / self.NP)

        # generate two set of random val for pso velocity update
        rand1 = np.random.rand(self.n_pop, self.NP, 1)
        rand2 = np.random.rand(self.n_pop, self.NP, 1)
        # print(self.NP)
        # print("population.shape = ", self.population.shape)

        action = action[:, :, None]
        # print("action.shape = ",action.shape)

        # update velocity
        new_velocity = self.w * self.particles['velocity'] + self.c * action * rand1 * (
                    self.particles['pbest_position'] - self.particles['current_position']) + \
                       self.c * (1 - action) * rand2 * (
                                   self.particles['gbest_position'][:, None, :] - self.particles['current_position'])

        # clip the velocity if exceeding the boarder
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        # update position
        # print("velocity.shape = ",new_velocity.shape)
        new_position = self.particles['current_position'] + new_velocity
        new_population = clipping(new_position, self.lb, self.ub)
        self.population = new_population
        self.velocity = new_velocity

        # calculate the new costs
        new_cost = self.get_costs(problem,new_population)
        self.pre_cost = self.cost
        self.cost = new_cost
        # update particles
        pbest_filters = new_cost < self.particles['pbest']  # n_pop, ps

        new_cbest_val = np.min(new_cost, 1)  # n_pop
        new_cbest_index = np.argmin(new_cost, 1)  # n_pop

        gbest_filters = new_cbest_val < self.particles['gbest_val']  # n_pop


        new_particles = {'current_position': new_position,
                         'c_cost': new_cost,
                         'pbest_position': np.where(np.expand_dims(pbest_filters, axis=-1),
                                                    new_position,
                                                    self.particles['pbest_position']),
                         'pbest': np.where(pbest_filters,
                                           new_cost,
                                           self.particles['pbest']),
                         'velocity': new_velocity,
                         'gbest_val': np.where(gbest_filters,
                                               new_cbest_val,
                                               self.particles['gbest_val']),
                         'gbest_position': np.where(np.expand_dims(gbest_filters, axis=-1),
                                                    new_position[np.arange(self.n_pop), new_cbest_index],
                                                    self.particles['gbest_position']),
                         'gbest_index': np.where(gbest_filters, new_cbest_index, self.particles['gbest_index'])
                         }

        # update the stagnation steps for the whole population
        self.no_improve += np.where(new_particles['gbest_val'] < self.particles['gbest_val'], 0, 1)

        # update the stagnation steps for single particle in the population
        filter_per_patience = new_particles['c_cost'] < self.particles['c_cost']
        self.per_no_improve += 1
        tmp = np.where(filter_per_patience, self.per_no_improve, np.zeros_like(self.per_no_improve))
        self.per_no_improve -= tmp



        # update the population
        self.particles = new_particles



        # see if the end condition is satisfied
        if self.fes >= self.max_fes:
            is_done = True
        if self.particles['gbest_val'].max() <= 1e-8:
            is_done = True


