"""
A example of how to use L2OBench to test the performance of the optimizer(traditional)
"""


from examples.PSO.traditional.basic_optimizer import DEAP_Optimizer_DE
from L2OBench.Problem.cec_dataset import Training_Dataset
from L2OBench.config import get_config


def run():
    problem = Training_Dataset(dim=10,
                               num_samples=1,
                               batch_size=1,
                               problems='Sphere',
                               shifted=True,
                               rotated=True,
                               biased=True)[0][0]
    deap = DEAP_Optimizer_DE(problem, get_config())
    deap.evolve()


if __name__ == '__main__':
    run()
