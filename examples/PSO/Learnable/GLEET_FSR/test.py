from GPSO import GPSO_numpy
from L2OBench.config import get_config
from L2OBench.Environment import PBO_Env
from L2OBench.Problem import Training_Dataset
from L2OBench.reward import triangle
from L2OBench.Trainer import ExperimentManager
from examples.PSO.Learnable.GLEET.Agent import gleet
from L2OBench.Problem.protein_docking import Protein_Docking_Dataset
import torch


config = get_config()
config.max_fes = 100000
config.population_size = 100
config.c = 4.1
config.max_velocity = 10
config.max_x = 100
config.w_decay = True
config.NP = 100
config.dim = 12

# env = PBO_Env(Training_Dataset(10, 1, 1, problems='Sphere')[0][0],
#               GPSO_numpy(config),
#               triangle)

env = PBO_Env(Protein_Docking_Dataset( difficulty='rigid',num_samples=1, batch_size=1)[0][0],
              GPSO_numpy(config),
              triangle)

state = env.reset()

# is_done = False


# Set the device, you can change it according to your actual situation
config.use_cuda = torch.cuda.is_available() and not config.no_cuda
config.device = torch.device("cuda:1" if config.use_cuda else "cpu")
agent = gleet.ppo(config, env)
config.need_agent = True
# manager = ComparisionManager(agent,env,config)

manager = ExperimentManager(agent=agent,env=env,config=config)


manager.run()