from Agent import basic_Agent, metalearning,gleet
from Optimizer import basic_optimizer,learnable_optimizer
from Environment import basic_environment
from Problem import basic_problem,cec_test_func

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from reward import binary
from Problem.cec_dataset import Training_Dataset
from config import get_config
def main():
    # 1.初始化problem
    problem = Training_Dataset(dim=10,
                     num_samples=1,
                     batch_size=1,
                     problems='Sphere',
                     shifted=True,
                     rotated=True,
                     biased=True)[0][0]


    config = get_config()
    # 2.初始化optimizer
    optimizer = learnable_optimizer.PSO()

    # 3.初始化agent
    agent = gleet.ppo()

    # 4.初始化env
    env = basic_environment.PBO_Env(problem,optimizer,reward_func=binary())

    # 5.初始化agent的memory
    agent.memory.clear_memory()
    # 6.初始化env的state
    state = env.reset()
    # 7.初始化reward
    reward = 0
    # 8.初始化done
    done = False

    # 13.初始化best_solution
    best_solution = None

    T = config.max_fes // config.population_size + 1

    # to store the whole rollout process
    cost_rollout = np.zeros((config.batch_size, T - 1))

    time_eval = 0
    collect_mean = []
    collect_std = []

    # 9.开始rollout
    for batch_id,batch in enumerate(env.problem):
        # 10.开始一个batch的rollout
        collect_gbest = np.zeros((config.batch_size, config.per_eval_time))
        # visualize the rollout process
        for i in range(config.per_eval_time):
            done = False
            state = env.reset()
            for t in tqdm(range(T), disable=config.no_progress_bar,
                          bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

                if agent:
                    # if RL_agent is provided, the action is from the agent.actor
                    action = agent.inference(env, need_gd=False)
                    action = action.cpu().numpy()
                else:
                    # if RL_agent is not provided, the action is set to zeros because of the need of the parallel environment,
                    # but inside the backbone algorithm, the parameter controlled by action will use the default choices
                    action = np.zeros(config.batch_size)

                # put action into environment

                # next_state, rewards, done, info = env.step(action)
                next_state, rewards, done= env.step(action)
                state = next_state

                # store the rollout cost history
                for tt in range(config.batch_size):
                    pass
                    # cost_rollout[tt, t] += info[tt]['gbest_val']
                if done.all():
                    if t + 1 < T:
                        for tt in range(config.batch_size):
                            pass
                            # cost_rollout[tt, t + 1:] += info[tt]['gbest_val']
                    # store the final cost in the end of optimization process
                    for tt in range(config.batch_size):
                        pass
                        # collect_gbest[tt, i] = info[tt]['gbest_val']
                    break
                # collect the mean and std of final cost
            collect_std.append(np.mean(np.std(collect_gbest, axis=-1)).item())
            collect_mean.append(np.mean(collect_gbest).item())

        # calculate the mean of rollout cost history
        cost_rollout /= time_eval
        cost_rollout = np.mean(cost_rollout, axis=0)

        # save rollout data to file
        saving_path = os.path.join(config.log_dir, config.RL_agent, "{}_{}".format(config.problem,
                                                                               config.dim),
                                   "rollout_{}_{}".format(config.run_name,epoch_id = 0))
        # only save part of the optimization process
        save_list = [cost_rollout[int((config.dim ** (k / 5 - 3) * config.max_fes) // config.population_size - 1)].item() for
                     k in range(15)]
        save_dict = {'mean': np.mean(collect_mean).item(), 'std': np.mean(collect_std).item(), 'process': save_list}
        np.save(saving_path, save_dict)


        # calculate and return the mean and std of final cost
        return np.mean(collect_gbest).item(), np.mean(collect_std).item()