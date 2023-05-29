import argparse
import time


def get_config(args=None):
    parser = argparse.ArgumentParser()
    # Common config
    parser.add_argument('--problem', default='bbob', choices=['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch'])
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--upperbound', type=float, default=5, help='the upperbound of search space')
    parser.add_argument('--difficulty', default='easy', choices=['easy', 'difficult'])
    parser.add_argument('--validate_interval', type=int, default=3)
    parser.add_argument('--validate_runs', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--train', default=None, action='store_true', help='switch to train mode')
    parser.add_argument('--test', default=None, action='store_true', help='switch to inference mode')
    parser.add_argument('--rollout', default=None, action='store_true', help='switch to rollout mode')

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
    parser.add_argument('--n_checkpoint', type=int, default=20, help='numbel of training checkpoint')

    # Testing parameters
    parser.add_argument('--agent', default=None, help='None: traditional optimizer, else Learnable optimizer')
    parser.add_argument('--agent_load_dir', type=str, default='agent_model/',
                        help='load your own agent model')
    parser.add_argument('--optimizer', default=None, help='your own learnable or traditional optimizer')
    parser.add_argument('--agent_for_cp', type=str, nargs='+', default=[],
                        help='learnable optimizer to compare')
    parser.add_argument('--l_optimizer_for_cp', type=str, nargs='+', default=[],
                        help='learnable optimizer to compare')  # same length with "agent_for_cp"
    parser.add_argument('--t_optimizer_for_cp', type=str, nargs='+', default=[],
                        help='traditional optimizer to compare')
    parser.add_argument('--test_batch_size', type=int, default=1)

    # Rollout parameters
    parser.add_argument('--agent_for_rollout', type=str, nargs='+', help='learnable optimizer for rollout')
    parser.add_argument('--optimizer_for_rollout', type=str, nargs='+', help='learnabel optimizer for rollout')

    config = parser.parse_args(args)
    config.maxFEs = 2000 * config.dim

    if config.problem in ['protein', 'protein-torch']:
        config.dim = 12
        config.maxFEs = 1000

    config.log_interval = 400

    config.run_time = f'{time.strftime("%Y%m%dT%H%M%S")}_{config.problem}_{config.difficulty}_{config.dim}D'
    config.test_log_dir = config.log_dir + '/test/' + config.run_time + '/'

    config.rollout_log_dir = config.log_dir + '/rollout/' + config.run_time + '/'

    if config.train:
        config.agent_save_dir = config.agent_save_dir + config.train_agent + '/' + config.run_time + '/'
    if config.rollout:
        config.agent_load_dir = config.agent_load_dir + '/rollout/'
    if config.test:
        config.agent_load_dir = config.agent_load_dir + '/test/'
    config.save_interval = config.max_learning_step // config.n_checkpoint

    if 'DEAP_CMAES' not in config.t_optimizer_for_cp:
        config.t_optimizer_for_cp.append('DEAP_CMAES')
    if 'Random_search' not in config.t_optimizer_for_cp:
        config.t_optimizer_for_cp.append('Random_search')

    return config
