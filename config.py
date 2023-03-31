import argparse


def get_config(args=None):
    parser = argparse.ArgumentParser()

    # Problem config
    parser.add_argument('--problem', default='Schwefel', choices=['Sphere', 'Schwefel', 'Ackley', 'Bent_cigar'])
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--lower_bound', type=int, default=-100)
    parser.add_argument('--upper_bound', type=int, default=100)

    # PBO config
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--FEs_interest', type=int, default=20000, help='FEs = FEs_interest * dim')
    parser.add_argument('--mutate_strategy', type=int, default=1, choices=[0, 1, 2])

    # Agent config
    parser.add_argument('--reward_definition', type=float, default=0., choices=[0., 0.1, 0.2, 3.1, 3.])

    config = parser.parse_args(args)

    return config
