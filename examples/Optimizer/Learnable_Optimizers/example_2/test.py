import numpy as np

from DE import DE
from L2OBench.Environment import PBO_Env
from L2OBench.Problem import Training_Dataset
from L2OBench.reward import triangle


env = PBO_Env(Training_Dataset(10, 1, 1, problems='Sphere')[0][0],
              DE(10, -100, 100, 100, 200000),
              triangle)

state = env.reset()
is_done = False
while not is_done:
    state, reward, is_done = env.step(np.random.choice(4, 100))
    print(state['fes'], state['cost'].min(), state['cost'].mean(), reward)
