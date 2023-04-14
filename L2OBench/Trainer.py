
from L2OBench.Optimizer import learnable_optimizer
from L2OBench.Environment import basic_environment

from L2OBench.reward import binary
from L2OBench.Problem.cec_dataset import Training_Dataset


class Experimentmanager():
    def __init__(self,problem, optimizer, agent, reward_function):
        self.env = basic_environment.PBO_Env(problem, optimizer, reward_function)
        self.agent = agent

        pass

    def run(self):
        # 1.已通过参数初始化agent和env
        # 2.agent得到action
        # 3.env根据action得到reward、更新种群
        # 4.重复2、3直到满足终止条件
        pass


