
from L2OBench.Agent import basic_Agent
import os
from torch import nn
import torch
from examples.Agent.baseNets import MultiHeadEncoder, MLP, EmbeddingNet
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
                 config
                 ):
        super(Actor, self).__init__()
        self.embedding_dim = config.embedding_dim,
        self.hidden_dim = config.hidden_dim,
        self.n_heads_actor = config.encoder_head_num,
        self.n_heads_decoder = config.decoder_head_num,
        self.n_layers = config.n_encode_layers,
        self.normalization = config.normalization,
        self.v_range = config.v_range,
        self.node_dim=config.node_dim,
        self.hidden_dim1=config.hidden_dim1_actor,
        self.hidden_dim2=config.hidden_dim2_actor,
        self.no_attn=config.no_attn,
        self.no_eef=config.no_eef,
        self.max_sigma=config.max_sigma,
        self.min_sigma=config.min_sigma,

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
            self.mu_net = MLP(self.embedding_dim, self.hidden_dim1, self.hidden_dim2, 1, 0)
            self.sigma_net = MLP(self.embedding_dim, self.hidden_dim1, self.hidden_dim2, 1, 0)
        else:
            # w/o both
            if self.no_eef:
                self.mu_net = MLP(self.node_dim, 16, 8, 1)
                self.sigma_net = MLP(self.node_dim, 16, 8, 1, 0)
            # w/o attn
            else:
                self.mu_net = MLP(3 * self.node_dim, 16, 8, 1)
                self.sigma_net = MLP(3 * self.node_dim, 16, 8, 1, 0)

        self.max_sigma = self.max_sigma
        self.min_sigma = self.min_sigma

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
                 config
                 ):
        super(Critic, self).__init__()
        input_critic = config.embedding_dim
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

        # for GLEET, input_dim = 32
        self.input_dim = input_critic
        # for GLEET, hidden_dim1 = 32, hidden_dim2 = 16
        self.hidden_dim1 = config.hidden_dim1_critic
        self.hidden_dim2 = config.hidden_dim2_critic

        self.value_head = MLP(input_dim=self.input_dim, mid_dim1=self.hidden_dim1, mid_dim2=self.hidden_dim2, output_dim=1)

    def forward(self, h_features):
        # since it's joint actions, the input should be meaned at population-dimention
        h_features = torch.mean(h_features, dim=-2)
        # pass through value_head to get baseline_value
        baseline_value = self.value_head(h_features)

        return baseline_value.detach().squeeze(), baseline_value.squeeze()



# load model from load_path
def load_model(load_path, agent):
    assert load_path is not None
    load_data = torch_load_cpu(load_path)

    # load data for actor
    model_actor = get_inner_model(agent.actor)
    model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

    if not agent.config.test:
        # load data for critic
        model_critic = get_inner_model(agent.critic)
        model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})

        # load data for torch and cuda
        torch.set_rng_state(load_data['rng_state'])
        if agent.config.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
    # done
    print(' [*] Load model from {}'.format(load_path))

# save model to save_path
def save_model(save_path, agent):
    assert save_path is not None
    save_data = {
        'actor': get_inner_model(agent.actor).state_dict(),
        'critic': get_inner_model(agent.critic).state_dict(),
        'rng_state': torch.get_rng_state(),
    }
    if agent.config.use_cuda:
        save_data['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    torch.save(save_data, save_path)
    print(' [*] Save model to {}'.format(save_path))



class ppo(basic_Agent.learnable_Agent):
    # init the network
    def __init__(self,config):
        self.config = config
        # memory store some needed information
        # agent network
        self.nets = [Actor(config),Critic(config)]
        # optimizer
        self.optimizer = torch.optim.Adam(
            [{'params': self.actor.parameters(), 'lr': config.lr_model}] +
            [{'params': self.critic.parameters(), 'lr': config.lr_model}])
        # figure out the lr schedule
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay, last_epoch=-1, )

    def get_feature(self,env):

        pass


    def inference(self,state,need_gd):
        # get aciton/fitness

        # check if need gradient to change mode
        if need_gd:
            torch.set_grad_enabled(True)  ##
            self.nets[0].train()
            if not self.config.test: self.nets[1].train()
        else:
            torch.set_grad_enabled(False)  ##
            self.nets[0].eval()
            if not self.config.test: self.nets[1].eval()

        self.memory.states.append(state.clone())
        action, log_lh, _to_critic, entro_p = self.nets[0](state,
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
        total_cost = 0
        entropy = []
        bl_val_detached = []
        bl_val = []

        baseline_loss = 0
        reinforce_loss = 0
        loss = 0

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
                    _, log_p, _to_critic, entro_p = self.nets[0](old_states[tt],
                                                                fixed_action=old_actions[tt],
                                                                require_entropy=True,  # take same action
                                                                to_critic=True
                                                                )

                    logprobs.append(log_p)
                    entropy.append(entro_p.detach().cpu())

                    baseline_val_detached, baseline_val = self.nets[1](_to_critic)

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
            R = self.nets[1](self.nets[0](env.state, only_critic=True))[0]

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

        return baseline_loss,reinforce_loss,loss,


    def learning(self,K_epochs=4,env=None,t_time=10):
        # begin update

        _,_,loss = self.cal_loss(K_epochs,env,t_time)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # end update


