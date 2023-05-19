import argparse
import time


def get_config(args=None):
    parser = argparse.ArgumentParser()
    # Common config
    parser.add_argument('--problem', default='bbob', choices=['bbob', 'bbob-noisy', 'protein'])
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--difficulty', default='easy', choices=['easy', 'difficult'])
    parser.add_argument('--validate_interval', type=int, default=3)
    parser.add_argument('--validate_runs', type=int, default=3)
    parser.add_argument('--device', default='cpu')

    # Training parameters
    parser.add_argument('--max_learning_step', type=int, default=1500000, help='the maximum learning step for training')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--train_agent', default=None, help='agent for training')
    parser.add_argument('--train_optimizer', default=None, help='optimizer for training')
    parser.add_argument('--agent_save_dir', type=str, default='agent_model/train/',
                        help='save your own trained agent model')
    parser.add_argument('--log_dir', type=str, default='output/',
                        help='logging testing output')
    parser.add_argument('--draw_interval', type=int, default=3)
    parser.add_argument('--agent_for_plot_training', type=str, nargs='+', default=['RL_HPSDE_Agent'],
                        help='learnable optimizer to compare')

    # Testing parameters
    parser.add_argument('--agent', default=None, help='None: traditional optimizer, else Learnable optimizer')
    parser.add_argument('--agent_load_dir', type=str, default='agent_model/test/',
                        help='load your own agent model')
    parser.add_argument('--optimizer', default=None, help='your own learnable or traditional optimizer')
    parser.add_argument('--agent_for_cp', type=str, nargs='+', default=['RL_HPSDE_Agent'],
                        help='learnable optimizer to compare')
    parser.add_argument('--l_optimizer_for_cp', type=str, nargs='+', default=['RL_HPSDE_Optimizer'],
                        help='learnable optimizer to compare')  # same length with "agent_for_cp"
    parser.add_argument('--t_optimizer_for_cp', type=str, nargs='+', default=['DEAP_CMAES', 'MadDE'],
                        help='traditional optimizer to compare')
    parser.add_argument('--test_batch_size', type=int, default=1)

    config = parser.parse_args(args)
    config.maxFEs = 2000 * config.dim

    if config.problem == 'protein':
        config.dim = 12
        config.maxFEs = 5000

    config.run_time = f'{time.strftime("%Y%m%dT%H%M%S")}_{config.problem}_{config.dim}D'
    config.test_log_dir = 'output/test/'+config.run_time + '/'

    return config
