import numpy as np
import torch
from agent.basic_agent import Basic_Agent
from agent.networks import MLP
from agent.utils import ReplayBuffer


class DEDQN_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        config.state_size = 4
        config.n_act = 3
        config.mlp_config = [{'in': config.state_size, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 10, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 10, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]
        config.lr = 1e-4
        config.epsilon = 0.1
        config.gamma = 0.8
        config.memory_size = 100
        config.batch_size = 64
        config.warm_up_size = config.batch_size
        self.__config = config

        self.__device = config.device
        self.__dqn = MLP(config.mlp_config).to(self.__device)
        self.__optimizer = torch.optim.AdamW(self.__dqn.parameters(), lr=config.lr)
        self.__criterion = torch.nn.MSELoss()
        self.__n_act = config.n_act
        self.__epsilon = config.epsilon
        self.__gamma = config.gamma
        self.__replay_buffer = ReplayBuffer(config.memory_size)
        self.__warm_up_size = config.warm_up_size
        self.__batch_size = config.batch_size
        self.__max_learning_step = config.max_learning_step
        self.__global_ls = 0

    def __get_action(self, state, options=None):
        state = torch.Tensor(state).to(self.__device)
        action = None
        Q_list = self.__dqn(state)
        if options['epsilon_greedy'] and np.random.rand() < self.__epsilon:
            action = np.random.randint(low=0, high=self.__n_act)
        if action is None:
            action = int(torch.argmax(Q_list).detach().cpu().numpy())
        Q = Q_list[action].detach().cpu().numpy()
        return action, Q

    def train_episode(self, env, epoch_id=None, logger=None):
        state = env.reset()
        done = False
        R = 0
        while not done:
            action, _ = self.__get_action(state, {'epsilon_greedy': True})
            next_state, reward, done = env.step(action)
            R += reward
            self.__replay_buffer.append((state, action, reward, next_state, done))
            # backward propagation
            if len(self.__replay_buffer) >= self.__warm_up_size:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.__replay_buffer.sample(self.__batch_size)
                pred_Vs = self.__dqn(batch_obs.to(self.__device))  # [batch_size, n_act]
                action_onehot = torch.nn.functional.one_hot(batch_action.to(self.__device), self.__n_act)  # [batch_size, n_act]
                predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]
                target_Q = batch_reward.to(self.__device) + (1 - batch_done.to(self.__device)) * self.__gamma * self.__dqn(batch_next_obs.to(self.__device)).max(1)[0]
                self.__optimizer.zero_grad()
                loss = self.__criterion(predict_Q, target_Q)
                loss.backward()
                self.__optimizer.step()
                self.__global_ls += 1
                if self.__global_ls >= self.__max_learning_step:
                    break
            state = next_state
        return self.__global_ls >= self.__max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                              'gbest': env.optimizer.cost[-1],
                                                              'return': R,
                                                              'learn_steps': self.__global_ls}

    def rollout_episode(self, env, epoch_id=None, logger=None):
        state = env.reset()
        done = False
        while not done:
            action, Q = self.__get_action(state, {'epsilon_greedy': False})
            next_state, reward, done = env.step(action)
            state = next_state
        return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes}
