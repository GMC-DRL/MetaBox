from Optimizer.basic_optimizer import DEAP_Optimizer_DE
from Problem.cec_dataset import Training_Dataset
from config import get_config


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
