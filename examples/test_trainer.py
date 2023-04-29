"""
A example of how to use L2OBench to train the optimizer(with agent)
"""

from examples.PSO.Learnable.GLEET.Agent import gleet
from L2OBench.Optimizer import learnable_optimizer

from L2OBench.reward import binary
from L2OBench.Problem.cec_dataset import Training_Dataset
from L2OBench.Trainer import Experimentmanager

def main():
    # 1.初始化problem
    # 2.初始化reward_function
    # 3.初始化optimizer
    # 4.初始化agent
    # 5.初始化experimentmanager
    # 6.开始实验

    problem = Training_Dataset(dim=10,
                               num_samples=1,
                               batch_size=1,
                               problems='Sphere',
                               shifted=True,
                               rotated=True,
                               biased=True)[0][0]

    optimizer = learnable_optimizer.PSO()

    agent = gleet.ppo()

    reward_function = binary

    experimentmanager = Experimentmanager(problem,optimizer,agent,reward_function)

    experimentmanager.run()

if __name__ == '__main__':
    main()
