import torch
from torch import nn
from agent.basic_agent import Basic_Agent
from agent.utils import *
from cmaes import CMA
import copy

class LES_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config

        self.meta_pop_size = 16
        self.skip_step = 50
        self.optimizer = CMA(mean = np.zeros(246), 
                             sigma = 0.1, 
                             population_size=self.meta_pop_size)
        
        # population in cmaes
        self.x_population = None
        self.meta_performances = None
        self.optimizer_step()
        self.best_x= self.x_population[0]

        self.costs = None
        self.best_les = None
        self.gbest = 1e10

        self.__learning_step=0


        self.__cur_checkpoint=0
        # save init agent
        if self.__cur_checkpoint==0:
            save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
            self.__cur_checkpoint+=1

    def update_setting(self, config):
        self.__config.max_learning_step = config.max_learning_step
        self.__config.agent_save_dir = config.agent_save_dir
        self.__learning_step = 0
        save_class(self.__config.agent_save_dir, 'checkpoint0', self)
        self.__config.save_interval = config.save_interval
        self.__cur_checkpoint = 1

    def optimizer_step(self):
        # inital sampling
        samples = []
        for _ in range(self.meta_pop_size):
            samples.append(self.optimizer.ask())
        self.x_population = np.vstack(samples)
        self.meta_performances = [[] for _ in range(self.meta_pop_size)]

    # eval task to get meta performance
    def train_episode(self, env):
        env_population = [copy.deepcopy(env) for _ in range(self.meta_pop_size)]
        
        
        # sequential
        for i, e in enumerate(env_population):
            e.reset()
            action = {'attn':self.x_population[i][:68],
                      'mlp':self.x_population[i][68:],
                      'skip_step': self.skip_step}
            sub_best, _, _, _ = e.step(action)
            
            self.meta_performances[i].append(sub_best)

        # todo: modify threshold
        self.__learning_step += 1
            
        if self.__learning_step % 10 == 0 and self.__config.problem in ['protein','protein-torch']:
            self.train_epoch()
            
        if self.__learning_step >= (self.__config.save_interval * self.__cur_checkpoint):
            save_class(self.__config.agent_save_dir, 'checkpoint'+str(self.__cur_checkpoint), self)
            self.__cur_checkpoint += 1
        
        # return exceed_max_ls
        return self.__learning_step >= self.__config.max_learning_step,{'normalizer': env_population[0].optimizer.cost[0],
                               'gbest': env_population[0].optimizer.cost[-1],
                               'return': 0,         # set to 0 since for non-RL approach there is no return
                               'learn_steps': self.__learning_step}

    # meta train, update self.x_population
    def train_epoch(self):
        scores = np.stack(self.meta_performances)
        

        self.costs = np.median((scores - np.mean(scores,axis=0)[None, :])/scores.std(axis=0)[None, :], axis=-1)
        
        # record x gbest
        if np.min(self.costs) < self.gbest:
            self.gbest = np.min(self.costs)
            self.best_les = np.argmin(self.costs)
            self.best_x=self.x_population[self.best_les]

        # update optimizer
        self.optimizer.tell(list(zip(self.x_population,self.costs)))
        self.optimizer_step()
        

    # rollout_episode need transform 
    def rollout_episode(self,env) :
        R = 0
        # use best_x to rollout
        env.reset()
        action = {'attn':self.best_x[:68],
                      'mlp':self.best_x[68:],}
        gbest, r, _, _ = env.step(action)
        R += r

        return {'cost':env.optimizer.cost,'fes': env.optimizer.FEs, 'return': R}
