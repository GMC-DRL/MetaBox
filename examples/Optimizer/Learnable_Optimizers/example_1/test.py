import numpy as np

from PSO import PSO
from L2OBench.config import get_config
from L2OBench.Environment import PBO_Env
from L2OBench.Problem import Training_Dataset
from L2OBench.reward import triangle
from L2OBench.Tester import ComparisionManager
from examples.Agent import gleet

import torch

env = PBO_Env(Training_Dataset(10, 1, 1, problems='Sphere')[0][0],
              PSO(10, -100, 100, 100, 200000),
              triangle)

state = env.reset()

# is_done = False

config = get_config()
# Set the device, you can change it according to your actual situation
config.use_cuda = torch.cuda.is_available() and not config.no_cuda
config.device = torch.device("cuda:1" if config.use_cuda else "cpu")
agent = gleet.ppo(config)

manager = ComparisionManager(agent,env,config)

manager.run()
# cost_mean,cost_std = manager.run()


# while not is_done:
#     state, reward, is_done = env.step(np.random.rand(2, 100))
#     print(state['fes'], state['cost'].min(), state['cost'].mean(), reward)
