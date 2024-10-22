import numpy as np
import torch
from optimizer.learnable_optimizer import Learnable_Optimizer
import torch.nn as nn

def vector2nn(x,net):
    assert len(x) == sum([param.nelement() for param in net.parameters()]), 'dim of x and net not match!'
    params = net.parameters()
    ptr = 0
    for v in params:
        num_of_params = v.nelement()
        temp = torch.FloatTensor(x[ptr: ptr+num_of_params])
        v.data = temp.reshape(v.shape)
        ptr += num_of_params
    return net


class SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(3,8)
        self.Wk = nn.Linear(3,8)
        self.Wv = nn.Linear(3,1)
    
    def forward(self, X):
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        attn_score = torch.softmax(torch.matmul(Q, K.T)/np.sqrt(8), dim=-1)
        return torch.softmax(torch.matmul(attn_score, V), dim=0).squeeze()

class LrNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(19,8)
        self.ln2 = nn.Linear(8,2)
        self.sm = nn.Sigmoid()
    def forward(self, X):
        X = self.ln1(X)
        return self.sm(self.ln2(X))

class LES_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.max_fes = config.maxFEs

        self.attn = SelfAttn()
        self.mlp = LrNet()
        self.alpha = [0.1,0.5,0.9] # alpha time-scale
        self.timestamp = np.array([1,3,10,30,50,100,250,500,750,1000,1250,1500,2000])
        self.save_time=0
        self.NP=16
        self.sigma_ratio=0.2

        self.FEs = None

        # for record
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval
    
    def init_population(self, problem):
        self.ub = problem.ub
        self.lb = problem.lb
        self.problem = problem

        mu = problem.lb + (problem.ub-problem.lb) * np.random.rand(problem.dim)
        sigma = np.ones(problem.dim)*self.ub*self.sigma_ratio
        population = np.clip(np.random.normal(mu,sigma,(self.NP,problem.dim)), self.lb, self.ub) # is it correct?
        costs = problem.eval(population)
        self.evolution_info = {'parents': population,
                'parents_cost':costs,
                'generation_counter': 0, 
                'gbest':np.min(costs),
                'Pc':np.zeros((3,problem.dim)),
                'Ps':np.zeros((3,problem.dim)),
                'mu':mu,
                'sigma':sigma}
        self.FEs = self.NP

        self.cost = [np.min(costs)]
        self.log_index = 1
        return None

    def cal_attn_feature(self):
        # z-score of population costs
        population_costs = self.evolution_info['parents_cost']
        z_score = (population_costs-np.mean(population_costs))/(np.std(population_costs)+1e-8) # avoid nan
        # shifted normalized ranking
        shifted_rank = np.argsort(population_costs)/self.NP - 0.5
        # improvement indicator
        improved = population_costs < self.evolution_info['gbest']
        # concat above three feature to N * 3 array
        return torch.from_numpy(np.vstack([z_score,shifted_rank,improved]).T).to(torch.float32)
    
    def cal_mlp_feature(self, W):
        # P_c_t P_sigma_t
        Pc = []
        Ps = []
        for i,alpha in enumerate(self.alpha):
            temp1 = (1-alpha) * self.evolution_info['Pc'][i] + \
                    alpha * (np.sum((self.evolution_info['parents'] - self.evolution_info['mu'])*W[:,None],axis=0) - self.evolution_info['Pc'][i]) # need to be checked!
            temp2 = (1-alpha) * self.evolution_info['Ps'][i] + \
                    alpha * (np.sum((self.evolution_info['parents'] - self.evolution_info['mu'])/self.evolution_info['sigma']*W[:,None],axis=0) - self.evolution_info['Ps'][i]) # need to be checked!
            Pc.append(temp1)
            Ps.append(temp2)
        
        # timestamp embedding
        rho = np.tanh(self.evolution_info['generation_counter'] / self.timestamp  - 1)[None,:].repeat(self.problem.dim,axis=0) #  dim * 13
        c = np.vstack(Pc) # dim * 3
        s = np.vstack(Ps) # dim * 3
        # concat to 19dim feature
        return torch.from_numpy(np.hstack([c.T,s.T,rho])).to(torch.float32), c, s
    
    def update(self,action, problem):

        # get new model parameters 
        self.attn=vector2nn(action['attn'],self.attn)
        self.mlp=vector2nn(action['mlp'],self.mlp)
        skip_step = None
        if action.get('skip_step') is not None:
            skip_step = action['skip_step']
        
        step = 0
        is_end = False
        init_y = None
        while not is_end:
            # get features of present population
            fitness_feature = self.cal_attn_feature()
            # get w_{i} for each individual
            W = self.attn(fitness_feature).detach().numpy() 
            # get features for mlp
            alpha_feature, Pc, Ps = self.cal_mlp_feature(W)
            # get learning rates
            alpha = self.mlp(alpha_feature).detach().numpy() # self.dim * 2
            alpha_mu = alpha[:,0]
            alpha_sigma = alpha[:,1]
            # update mu and sigma for next generation
            mu = (1 - alpha_mu) * self.evolution_info['mu'] + \
                alpha_mu * np.sum((self.evolution_info['parents'] - self.evolution_info['mu'])*W[:,None],axis=0)
            sigma = (1 - alpha_sigma) * self.evolution_info['sigma'] + \
                alpha_sigma * np.sqrt(np.sum((self.evolution_info['parents'] - self.evolution_info['mu']) ** 2 *W[:,None],axis=0)) # need to be checked!
            # sample childs according new mu and sigma
            population = np.clip(np.random.normal(mu,sigma,(self.NP,self.problem.dim)), self.lb, self.ub)
            # evaluate the childs
            costs = self.problem.eval(population)
            self.FEs += self.NP
            gbest = np.min([np.min(costs),self.evolution_info['gbest']])
            if step == 0:
                init_y = gbest
            t = self.evolution_info['generation_counter'] + 1
            # update evolution information
            self.evolution_info = {'parents': population,
                    'parents_cost':costs,
                    'generation_counter': t, 
                    'gbest':gbest,
                    'Pc':Pc,
                    'Ps':Ps,
                    'mu':mu,
                    'sigma':sigma}
            if problem.optimum is None:
                is_end = (self.FEs >= self.max_fes)
            else:
                is_end = (self.FEs >= self.max_fes or gbest <= 1e-8)
            step += 1
            if skip_step is not None:
                is_end = (step >= skip_step)

            if self.FEs >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(gbest)
            
            if is_end:
                if len(self.cost) >= self.__config.n_logpoint + 1:
                    self.cost[-1] = gbest
                else:
                    self.cost.append(gbest)
        
        return self.evolution_info['gbest'],(init_y - gbest) / init_y,is_end,{}
    


    