from basic_Agent import Memory,lr_sd,learnable_Agent
import os
from torch import nn
import torch
from Agent.baseNets import MultiHeadEncoder, MLP, EmbeddingNet
from torch.distributions import Normal

from utils import torch_load_cpu, get_inner_model


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


# defination of the Actor network
class Actor(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_heads_decoder,
                 n_layers,
                 normalization,
                 v_range,
                 node_dim,
                 hidden_dim1,
                 hidden_dim2,
                 no_attn=False,
                 no_eef=False,
                 max_sigma=0.7,
                 min_sigma=1e-3,
                 ):
        super(Actor, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_heads_decoder = n_heads_decoder
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.no_attn = no_attn
        self.no_eef = no_eef
        self.node_dim = node_dim

        # figure out the Actor network
        if not self.no_attn:
            # figure out the embedder for feature embedding
            self.embedder = EmbeddingNet(
                self.node_dim,
                self.embedding_dim)
            # figure out the fully informed encoder
            self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor,
                                 self.embedding_dim,
                                 self.hidden_dim,
                                 self.normalization, )
                for _ in range(self.n_layers)))  # stack L layers

            # w/o eef for ablation study
            if not self.no_eef:
                # figure out the embedder for exploration and exploitation feature
                self.embedder_for_decoder = EmbeddingNet(2 * self.embedding_dim, self.embedding_dim)
                # figure out the exploration and exploitation decoder
                self.decoder = mySequential(*(
                    MultiHeadEncoder(self.n_heads_actor,
                                     self.embedding_dim,
                                     self.hidden_dim,
                                     self.normalization, )
                    for _ in range(self.n_layers)))  # stack L layers
            # figure out the mu_net and sigma_net
            self.mu_net = MLP(self.embedding_dim, hidden_dim1, hidden_dim2, 1, 0)
            self.sigma_net = MLP(self.embedding_dim, hidden_dim1, hidden_dim2, 1, 0)
        else:
            # w/o both
            if self.no_eef:
                self.mu_net = MLP(self.node_dim, 16, 8, 1)
                self.sigma_net = MLP(self.node_dim, 16, 8, 1, 0)
            # w/o attn
            else:
                self.mu_net = MLP(3 * self.node_dim, 16, 8, 1)
                self.sigma_net = MLP(3 * self.node_dim, 16, 8, 1, 0)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

        print(self.get_parameter_number())

    def get_parameter_number(self):

        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in, fixed_action=None, require_entropy=False, to_critic=False, only_critic=False):
        if not self.no_attn:
            population_feature = x_in[:, :, :self.node_dim]
            eef = x_in[:, :, self.node_dim:]
            # pass through embedder
            h_em = self.embedder(population_feature)
            # pass through encoder
            logits = self.encoder(h_em)
            if not self.no_eef:
                # pass through the embedder to get eef embedding
                exploration_feature = eef[:, :, :9]
                exploitation_feature = eef[:, :, 9:]
                exploration_eb = self.embedder(exploration_feature)
                exploitation_eb = self.embedder(exploitation_feature)
                x_in_decoder = torch.cat((exploration_eb, exploitation_eb), dim=-1)
                # pass through the embedder for decoder
                x_in_decoder = self.embedder_for_decoder(x_in_decoder)

                # pass through decoder
                logits = self.decoder(logits, x_in_decoder)
            # share logits to critic net, where logits is from the decoder output
            if only_critic:
                return logits  # .view(bs, dim, ps, -1)
            # finally decide the mu and sigma
            mu = (torch.tanh(self.mu_net(logits)) + 1.) / 2.
            sigma = (torch.tanh(self.sigma_net(logits)) + 1.) / 2. * (self.max_sigma - self.min_sigma) + self.min_sigma
        else:
            feature = x_in
            if self.no_eef:
                feature = x_in[:, :, :self.node_dim]
            if only_critic:
                return feature
            mu = (torch.tanh(self.mu_net(feature)) + 1.) / 2.
            sigma = (torch.tanh(self.sigma_net(feature)) + 1.) / 2. * (self.max_sigma - self.min_sigma) + self.min_sigma

        # don't share the network between actor and critic if there is no attention mechanism
        _to_critic = feature if self.no_attn else logits

        policy = Normal(mu, sigma)

        if fixed_action is not None:
            action = torch.tensor(fixed_action)
        else:
            # clip the action to (0,1)
            action = torch.clamp(policy.sample(), min=0, max=1)
        # get log probability
        log_prob = policy.log_prob(action)

        # The log_prob of each instance is summed up, since it is a joint action for a population
        log_prob = torch.sum(log_prob, dim=1)

        if require_entropy:
            entropy = policy.entropy()  # for logging only

            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   entropy)
        else:
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   )
        return out

# defination of the Critic network
class Critic(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim1,
                 hidden_dim2
                 ):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        # for GLEET, hidden_dim1 = 32, hidden_dim2 = 16
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.value_head = MLP(input_dim=self.input_dim, mid_dim1=hidden_dim1, mid_dim2=hidden_dim2, output_dim=1)

    def forward(self, h_features):
        # since it's joint actions, the input should be meaned at population-dimention
        h_features = torch.mean(h_features, dim=-2)
        # pass through value_head to get baseline_value
        baseline_value = self.value_head(h_features)

        return baseline_value.detach().squeeze(), baseline_value.squeeze()


class ppo():
    # init the network
    def __init__(self,config):
        self.config = config
        # memory store some needed information
        self.memory = Memory()
        # agent network
        self.actor = Actor(
            embedding_dim = config.embedding_dim,
            hidden_dim = config.hidden_dim,
            n_heads_actor = config.encoder_head_num,
            n_heads_decoder = config.decoder_head_num,
            n_layers = config.n_encode_layers,
            normalization = config.normalization,
            v_range = config.v_range,
            node_dim=config.node_dim,
            hidden_dim1=config.hidden_dim1_actor,
            hidden_dim2=config.hidden_dim2_actor,
            no_attn=config.no_attn,
            no_eef=config.no_eef,
            max_sigma=config.max_sigma,
            min_sigma=config.min_sigma,
        )


        if not config.test:
            # for the sake of ablation study, figure out the input_dim for critic according to setting
            if config.no_attn and config.no_eef:
                input_critic= config.node_dim
            elif config.no_attn and not config.no_eef:
                input_critic= 3 * config.node_dim
            elif config.no_eef and not config.no_attn:
                input_critic= config.node_dim
            else:
                # GLEET(default) setting, share the attention machanism between actor and critic
                input_critic= config.embedding_dim
            # figure out the critic network
            self.critic = Critic(
                input_dim = input_critic,
                hidden_dim1 = config.hidden_dim1_critic,
                hidden_dim2 = config.hidden_dim2_critic,
            )

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay, last_epoch=-1, )

    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.config.test:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})

            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.config.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.config.save_dir, 'epoch-{}.pt'.format(epoch))
        )


    # change working mode to evaling
    def eval(self):
        torch.set_grad_enabled(False)  ##
        self.actor.eval()
        if not self.opts.test: self.critic.eval()

    # change working mode to training
    def train(self):
        torch.set_grad_enabled(True)  ##
        self.actor.train()
        if not self.opts.test: self.critic.train()

    def get_feature(self,env):

        pass


    def inference(self,env,need_gd):
        # get aciton/fitness
        self.memory.states.append(env.state.clone())
        action, log_lh, _to_critic, entro_p = self.actor(env.state,
                                                          require_entropy=True,
                                                          to_critic=True
                                                          )
        self.memory.actions.append(action)
        self.memory.log_lh.append(log_lh)
        if need_gd:
            return action, entro_p
        else:
            return action


    def cal_loss(self,K_epochs=4,env=None,t_time=10):
        # cal loss

        old_actions = torch.stack(self.memory.actions)
        old_states = torch.stack(self.memory.states).detach()  # .view(t_time, bs, ps, dim_f)
        # old_actions = all_actions.view(t_time, bs, ps, -1)
        # print('old_actions.shape:{}'.format(old_actions.shape))
        old_logprobs = torch.stack(self.memory.logprobs).detach().view(-1)

        # Optimize PPO policy for K mini-epochs:
        old_value = None
        for _k in range(K_epochs):
            if _k == 0:
                logprobs = self.memory.logprobs

            else:
                # Evaluating old actions and values :
                logprobs = []
                entropy = []
                bl_val_detached = []
                bl_val = []

                for tt in range(t_time):
                    # get new action_prob
                    _, log_p, _to_critic, entro_p = self.actor(old_states[tt],
                                                                fixed_action=old_actions[tt],
                                                                require_entropy=True,  # take same action
                                                                to_critic=True
                                                                )

                    logprobs.append(log_p)
                    entropy.append(entro_p.detach().cpu())

                    baseline_val_detached, baseline_val = self.critic(_to_critic)

                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)

            logprobs = torch.stack(logprobs).view(-1)
            entropy = torch.stack(entropy).view(-1)
            bl_val_detached = torch.stack(bl_val_detached).view(-1)
            bl_val = torch.stack(bl_val).view(-1)

            # get traget value for critic
            Reward = []
            reward_reversed = self.memory.rewards[::-1]
            # get next value
            R = self.critic(self.actor(env.state, only_critic=True))[0]

            # R = agent.critic(x_in)[0]
            critic_output = R.clone()
            for r in range(len(reward_reversed)):
                R = R * self.config.gamma + reward_reversed[r]
                Reward.append(R)
            # clip the target:
            Reward = torch.stack(Reward[::-1], 0)
            Reward = Reward.view(-1)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages
            reinforce_loss = -torch.min(surr1, surr2).mean()

            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - self.config.eps_clip, self.config.eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()

            # check K-L divergence (for logging only)
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
            # calculate loss
            loss = baseline_loss + reinforce_loss
        return loss, entropy, critic_output, bl_val_detached, bl_val, Reward, approx_kl_divergence


    def learning(self,K_epochs=4,env=None,t_time=10):
        # cal_loss
        # update nets
        loss, entropy, critic_output, bl_val_detached, bl_val, Reward, approx_kl_divergence = self.cal_loss(K_epochs,env,t_time)
        loss.backward()
        # end update


