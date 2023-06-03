import numpy as np
from agent.basic_agent import Basic_Agent
from .utils import *

class RL_HPSDE_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        config.n_states = 4
        config.n_actions = 4
        config.alpha_max = 0.8
        config.alpha_decay = False
        config.gamma = 0.5
        self.__config = config

        self.__q_table = np.zeros((config.n_states, config.n_actions))
        self.__gamma = config.gamma
        self.__alpha_max = config.alpha_max
        self.__alpha = config.alpha_max
        self.__alpha_decay = config.alpha_decay
        self.__n_actions = config.n_actions
        self.__max_learning_step = config.max_learning_step
        self.__global_ls = 0

        self.__cur_checkpoint=0

        # save init agent
        if self.__cur_checkpoint==0:
            save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
            self.__cur_checkpoint+=1

    def __get_action(self, state):
        exp = np.exp(self.__q_table[state])
        prob = exp / exp.sum()
        return np.random.choice(self.__n_actions, size=1, p=prob)

    def train_episode(self, env):
        state = env.reset()
        done = False
        R = 0
        while not done:
            action = self.__get_action(state)
            next_state, reward, done = env.step(action)
            R += reward
            # update Q-table
            TD_error = reward + self.__gamma * self.__q_table[next_state].max() - self.__q_table[state][action]
            self.__q_table[state][action] += self.__alpha * TD_error
            self.__global_ls += 1

            if self.__global_ls >= (self.__config.save_interval * self.__cur_checkpoint):
                save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
                self.__cur_checkpoint+=1

            if self.__global_ls >= self.__max_learning_step:
                break
            if self.__alpha_decay:
                self.__alpha = self.__alpha_max - (self.__alpha_max - 0.1) * self.__global_ls / self.__max_learning_step
            state = next_state
        return self.__global_ls >= self.__max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                              'gbest': env.optimizer.cost[-1],
                                                              'return': R,
                                                              'learn_steps': self.__global_ls}

    def rollout_episode(self, env):
        state = env.reset()
        done = False
        R=0
        while not done:
            action = self.__get_action(state)
            next_state, reward, done = env.step(action)
            R+=reward
            state = next_state
        return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes,'return':R}
