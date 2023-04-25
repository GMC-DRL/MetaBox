"""
This file is used to test the performance of the optimizer (mainly for the kind of optimizer that is learnable)
ComparisionManager should have the following functions:
    1. __init__(self,problem, optimizer, agent, reward_function) to initialize the ComparisionManager
    2. run(self) to run the ComparisionManager and get the performance of the optimizer
"""
from L2OBench.Optimizer import learnable_optimizer
from L2OBench.Environment import basic_environment
import torch
import numpy as np
from tqdm import tqdm
import os

from L2OBench.reward import binary
from L2OBench.Problem.cec_dataset import Training_Dataset
from L2OBench.config import get_config

# from L2OBench.Environment.env import SubprocVectorEnv,DummyVectorEnv

class ComparisionManager():
    # def __init__(self,problem, optimizer, agent, reward_function):
    #     self.env = basic_environment.PBO_Env(problem, optimizer, reward_function)
    #     self.agent = agent
    #     self.config = get_config()

    def __init__(self,agent,env,config=None):
        self.env = env
        self.agent = agent
        self.config = config

    def run(self):
        is_done = False
        i = 0
        # 此处的state仅是部分env的meta info ，并不是env的state，env的state是在agent的get_feature中得到的
        state = self.env.reset()
        # state = torch.FloatTensor(state).to(self.config.device)
        # while not is_done:
        while i < 100:
            action = self.agent.inference(self.env, need_gd=True)
            action = action.detach().cpu().numpy()
            state, reward, is_done = self.env.step(action)
            print("fes:", state['fes'], "cost_min:", state['cost'].min(), "cost_mean:", state['cost'].mean(),)
            i += 1




# class ComparisionManager():
#     # def __init__(self,problem, optimizer, agent, reward_function):
#     #     self.env = basic_environment.PBO_Env(problem, optimizer, reward_function)
#     #     self.agent = agent
#     #     self.config = get_config()
#
#     def __init__(self,agent,env,config=None):
#         self.env = env
#         self.agent = agent
#         self.config = config
#
#     def run(self):
#         is_done = False
#         # 此处的state仅是部分env的meta info ，并不是env的state，env的state是在agent的get_feature中得到的
#         state = self.env.reset()
#         # state = torch.FloatTensor(state).to(self.config.device)
#         while not is_done:
#             action,_,_,_ = self.agent.inference(self.env, need_gd=False)
#             action = action.cpu().numpy()
#             state, reward, is_done = self.env.step(action[0])
#             print("fes:", state['fes'], "cost_min:", state['cost'].min(), "cost_mean:", state['cost'].mean(),
#                   "reward:", reward)
#


    # 1.初始化agent和env
    # 2.每个batch中，agent得到action、env根据action得到reward
    # 3.计算多个batch的平均reward和gbest
    # def run(self):
    #     # 5.初始化agent的memory
    #     self.agent.memory.clear_memory()
    #     # 6.初始化env的state
    #     state = self.env.reset()
    #     # 7.初始化reward
    #     reward = 0
    #     # 8.初始化done
    #     done = False
    #
    #     # 13.初始化best_solution
    #     best_solution = None
    #
    #     T = self.config.max_fes // self.config.population_size + 1
    #
    #     # to store the whole rollout process
    #     cost_rollout = np.zeros((self.config.batch_size, T - 1))
    #
    #     time_eval = 0
    #     collect_mean = []
    #     collect_std = []
    #
    #     # 9.开始rollout
    #     for batch_id, batch in enumerate(self.env.problem):
    #         # 10.开始一个batch的rollout
    #         collect_gbest = np.zeros((self.config.batch_size, self.config.per_eval_time))
    #         # visualize the rollout process
    #         for i in range(self.config.per_eval_time):
    #             done = False
    #             state = self.env.reset()
    #             for t in tqdm(range(T), disable=self.config.no_progress_bar,
    #                           bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
    #
    #                 if self.agent:
    #                     # if RL_agent is provided, the action is from the agent.actor
    #                     action = self.agent.inference(self.env, need_gd=False)
    #                     action = action.cpu().numpy()
    #                 else:
    #                     # if RL_agent is not provided, the action is set to zeros because of the need of the parallel environment,
    #                     # but inside the backbone algorithm, the parameter controlled by action will use the default choices
    #                     action = np.zeros(self.config.batch_size)
    #
    #                 # put action into environment
    #
    #                 # next_state, rewards, done, info = env.step(action)
    #                 next_state, rewards, done = self.env.step(action)
    #                 state = next_state
    #
    #                 # store the rollout cost history
    #                 for tt in range(self.config.batch_size):
    #                     pass
    #                     # cost_rollout[tt, t] += info[tt]['gbest_val']
    #                 if done.all():
    #                     if t + 1 < T:
    #                         for tt in range(self.config.batch_size):
    #                             pass
    #                             # cost_rollout[tt, t + 1:] += info[tt]['gbest_val']
    #                     # store the final cost in the end of optimization process
    #                     for tt in range(self.config.batch_size):
    #                         pass
    #                         # collect_gbest[tt, i] = info[tt]['gbest_val']
    #                     break
    #                 # collect the mean and std of final cost
    #             collect_std.append(np.mean(np.std(collect_gbest, axis=-1)).item())
    #             collect_mean.append(np.mean(collect_gbest).item())
    #
    #         # calculate the mean of rollout cost history
    #         cost_rollout /= time_eval
    #         cost_rollout = np.mean(cost_rollout, axis=0)
    #
    #         # save rollout data to file
    #         saving_path = os.path.join(self.config.log_dir, self.config.RL_agent, "{}_{}".format(self.config.problem,
    #                                                                                    self.config.dim),
    #                                    "rollout_{}".format(self.config.run_name, epoch_id=0))
    #         # only save part of the optimization process
    #         save_list = [
    #             cost_rollout[int((self.config.dim ** (k / 5 - 3) * self.config.max_fes) // self.config.population_size - 1)].item() for
    #             k in range(15)]
    #         save_dict = {'mean': np.mean(collect_mean).item(), 'std': np.mean(collect_std).item(), 'process': save_list}
    #         np.save(saving_path, save_dict)
    #
    #         # calculate and return the mean and std of final cost
    #         return np.mean(collect_gbest).item(), np.mean(collect_std).item()



