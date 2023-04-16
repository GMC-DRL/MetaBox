"""
A example of how to use L2OBench to test the performance of the optimizer(with agent)
"""
from examples.Agent import gleet
from L2OBench.Optimizer import learnable_optimizer
from L2OBench.Environment import basic_environment

import numpy as np
from tqdm import tqdm
import os

from L2OBench.reward import binary
from L2OBench.Problem.cec_dataset import Training_Dataset
from L2OBench.config import get_config
from L2OBench.Tester import ComparisionManager


def main():
    # 1.初始化problem
    problem = Training_Dataset(dim=10,
                     num_samples=1,
                     batch_size=1,
                     problems='Sphere',
                     shifted=True,
                     rotated=True,
                     biased=True)[0][0]


    # 2.初始化optimizer
    optimizer = learnable_optimizer.PSO()

    # 3.初始化agent
    agent = gleet.ppo()

    # 4.初始化reward_function
    reward_function = binary

    # 5.初始化manager
    manager = ComparisionManager(problem,optimizer,agent,reward_function)

    # 6.开始比较
    cost_mean,cost_std = manager.run()

    print(cost_mean,cost_std)


if __name__ == '__main__':
    main()