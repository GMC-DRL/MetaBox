import numpy as np

from .basic_optimizer import basic_optimizer

class Random_search(basic_optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__fes=0
        self.log_index=None
        self.cost=None
        self.__dim=config.dim
        self.__max_fes=config.maxFEs
        self.__NP=100
        self.log_interval=400


    def __reset(self,problem):
        self.__fes=0
        self.cost=[]
        self.__random_population(problem,init=True)
        self.cost.append(self.gbest)
        self.log_index=1

    def __random_population(self,problem,init):
        rand_pos=np.random.uniform(low=problem.lb,high=problem.ub,size=(self.__NP,self.__dim))
        if problem.optimum is None:
            cost=problem.eval(rand_pos)
        else:
            cost=problem.eval(rand_pos)-problem.optimum
        self.__fes+=self.__NP
        if init:
            self.gbest=np.min(cost)
        else:
            if self.gbest>np.min(cost):
                self.gbest=np.min(cost)
    

    def run_episode(self, problem):
        self.__reset(problem)
        is_done = False
        while not is_done:
            self.__random_population(problem,init=False)
            if self.__fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(self.gbest)

            if problem.optimum is None:
                is_done = self.__fes>=self.__max_fes
            else:
                is_done = self.gbest<=1e-8 or self.__fes>=self.__max_fes

            if is_done:
                break
                
        return {'cost':self.cost,'fes':self.__fes}