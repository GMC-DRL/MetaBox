from L2OBench.config import get_config
from L2OBench.Environment import PBO_Env
from L2OBench.Problem import Training_Dataset
from L2OBench.reward import triangle
from L2OBench.Tester import ComparisionManager
from L2OBench.Trainer import Experimentmanager
from examples.Agent import gleet,metalearning
from examples.Optimizer.Learnable_Optimizers.example_6.meta_optimizer import meta_optimizer
import torch
import torch.nn as nn

config = get_config()
config.max_fes = 100000
config.hidden_dim = 10
config.population_size = 1
config.h = 0.001
config.lr = 0.001
config.c = 4.1
config.max_x = 100
config.w_decay = True
config.NP = 1
env = PBO_Env(Training_Dataset(10, 1, 1, problems='Sphere')[0][0],
              meta_optimizer(config),
              triangle)

state = env.reset()

# is_done = False


# Set the device, you can change it according to your actual situation
config.use_cuda = torch.cuda.is_available() and not config.no_cuda
config.device = torch.device("cuda:1" if config.use_cuda else "cpu")
agent = metalearning.l2l_agent(config)

# manager = ComparisionManager(agent,env,config)
manager = Experimentmanager(agent,env,config)

manager.run()