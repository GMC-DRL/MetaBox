import argparse


def get_config(args=None):
    parser = argparse.ArgumentParser()

    # choose mode
    parser.add_argument('--train', default=None, action='store_true', help='switch to train mode')
    parser.add_argument('--test', default=None, action='store_true', help='switch to inference mode')

    # Problem config
    parser.add_argument('--problem', default='Schwefel', choices=['Sphere', 'Schwefel', 'Ackley', 'Bent_cigar'])
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--lower_bound', type=int, default=-100)
    parser.add_argument('--upper_bound', type=int, default=100)

    # PBO config
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--FEs_interest', type=int, default=20000, help='FEs = FEs_interest * dim')
    parser.add_argument('--mutate_strategy', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--boundary_ctrl', default='clipping', choices=['clipping', 'random', 'reflection', 'periodic', 'halving', 'parent'], help='boundary control method')

    # Agent config
    parser.add_argument('--reward_definition', type=float, default=0., choices=[0., 0.1, 0.2, 3.1, 3.])
    # Net(Attention Aggregation) parameters
    parser.add_argument('--v_range', type=float, default=6., help='to control the entropy')
    parser.add_argument('--encoder_head_num', type=int, default=4, help='head number of encoder')
    parser.add_argument('--decoder_head_num', type=int, default=4, help='head number of decoder')
    parser.add_argument('--critic_head_num', type=int, default=4, help='head number of critic encoder')
    parser.add_argument('--embedding_dim', type=int, default=16, help='dimension of input embeddings')  #
    parser.add_argument('--hidden_dim', type=int, default=16, help='dimension of hidden layers in Enc/Dec')  # 减小
    parser.add_argument('--n_encode_layers', type=int, default=1,
                        help='number of stacked layers in the encoder')  # 减小一点
    parser.add_argument('--normalization', default='layer', help="normalization type, 'layer' (default) or 'batch'")
    parser.add_argument('--node_dim', default=9, type=int, help='feature dimension for backbone algorithm')
    parser.add_argument('--hidden_dim1_critic', default=32, help='the first hidden layer dimension for critic')
    parser.add_argument('--hidden_dim2_critic', default=16, help='the second hidden layer dimension for critic')
    parser.add_argument('--hidden_dim1_actor', default=32, help='the first hidden layer dimension for actor')
    parser.add_argument('--hidden_dim2_actor', default=8, help='the first hidden layer dimension for actor')
    parser.add_argument('--output_dim_actor', default=1, help='output action dimension for actor')
    parser.add_argument('--lr_decay', type=float, default=0.9862327, help='learning rate decay per epoch',
                        choices=[0.998614661, 0.9862327])
    parser.add_argument('--max_sigma', default=0.7, type=float, help='upper bound for actor output sigma')
    parser.add_argument('--min_sigma', default=0.01, type=float, help='lowwer bound for actor output sigma')
    # for ablation study 消融实验
    parser.add_argument('--no_attn', action='store_true', default=False,
                        help='whether the network has attention mechanism or not')
    parser.add_argument('--no_eef', action='store_true', default=False,
                        help='whether the state has exploitation&exploration features ')

    # Training parameters
    parser.add_argument('--max_learning_step', default=4000000, help='the maximum learning step for training')
    parser.add_argument('--RL_agent', default='ppo', choices=['ppo'], help='RL Training algorithm')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor for future rewards')
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument('--T_train', type=int, default=1800, help='number of itrations for training')
    parser.add_argument('--n_step', type=int, default=10, help='n_step for return estimation')
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances per batch during training')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--epoch_end', type=int, default=100, help='maximum training epoch')
    parser.add_argument('--epoch_size', type=int, default=1024, help='number of instances per epoch during training')
    parser.add_argument('--lr_model', type=float, default=4e-5, help="learning rate for the actor network")
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='maximum L2 norm for gradient clipping')

    config = parser.parse_args(args)
    config.FEs = config.FEs_interest * config.dim

    return config
