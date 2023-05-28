import copy
from utils import construct_problem_set
import numpy as np
import pickle
from environment.basic_environment import PBO_Env
import time
from tqdm import tqdm
import os


from agent import (
    DE_DDQN_Agent,
    DEDQN_Agent,
    RL_HPSDE_Agent,
    LDE_Agent,
    QLPSO_Agent,
    RLEPSO_Agent,
    RL_PSO_Agent
)
from optimizer import (
    DE_DDQN_Optimizer,
    DEDQN_Optimizer,
    RL_HPSDE_Optimizer,
    LDE_Optimizer,
    QLPSO_Optimizer,
    RLEPSO_Optimizer,
    RL_PSO_Optimizer,

    DEAP_DE,
    JDE21,
    MadDE,
    NL_SHADE_LBC,

    DEAP_PSO,
    GL_PSO,
    sDMS_PSO,
    SAHLPSO,

    DEAP_CMAES,
    Random_search,
    BayesianOptimizer
)

def cal_t0(dim,fes):
    start = time.perf_counter()
    for _ in range(fes):
        x = np.random.rand(dim)
        x + x
        x / (x+2)
        x * x
        np.sqrt(x)
        np.log(x)
        np.exp(x)
    end = time.perf_counter()
    # ms
    return (end - start) * 1000

def cal_t1(problem, dim, fes):
    x = np.random.rand(fes, dim)
    start = time.perf_counter()
    # for i in range(fes):
    #     problem.eval(x[i])
    problem.eval(x)
    end = time.perf_counter()
    # ms
    return (end - start) * 1000

class Tester(object):
    def __init__(self, config):
        # ! 改成self.agent
        agent_name = config.agent
        agent_load_dir = config.agent_load_dir
        self.agent = None
        if agent_name is not None:  # learnable optimizer
            file_path = agent_load_dir + agent_name + '.pkl'
            with open(file_path, 'rb') as f:
                self.agent = pickle.load(f)
            # self.agent = pickle.load(agent_load_dir + agent_name + '.pkl')
        if config.optimizer is not None:
            self.optimizer_name = config.optimizer
            self.optimizer = eval(config.optimizer)(copy.deepcopy(config))
        self.log_dir = config.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.config = config
        _, self.test_set = construct_problem_set(self.config)
        self.seed = range(51)
        # initialize the dataframe for logging
        self.test_results = {'cost': {},
                             'fes': {},
                             'T0': 0.,
                             'T1': 0.,
                             'T2': {}}

        # prepare experimental optimizers and agents
        self.agent_for_cp = []
        # ! 改了pickle load文件时的报错
        for agent in config.agent_for_cp:
            file_path = agent_load_dir + agent + '.pkl'
            with open(file_path, 'rb') as f:
                self.agent_for_cp.append(pickle.load(f))
            # self.agent_for_cp.append(pickle.load(agent_load_dir + agent + '.pkl'))
        self.l_optimizer_for_cp = []
        for optimizer in config.l_optimizer_for_cp:
            self.l_optimizer_for_cp.append(eval(optimizer)(copy.deepcopy(config)))
        self.t_optimizer_for_cp = []
        for optimizer in config.t_optimizer_for_cp:
            self.t_optimizer_for_cp.append(eval(optimizer)(copy.deepcopy(config)))
        # ! 改成了self.agent
        if self.agent is not None:
            self.agent_for_cp.append(self.agent)
            self.l_optimizer_for_cp.append(self.optimizer)
        elif config.optimizer is not None:
            self.t_optimizer_for_cp.append(self.optimizer)


        # logging
        if len(self.agent_for_cp) == 0:
            print('None of learnable agent')
        else:
            print(f'there are {len(self.agent_for_cp)} agent')
            for a, l_optimizer in zip(self.agent_for_cp, self.l_optimizer_for_cp):
                print(f'learnable_agent:{type(a).__name__},l_optimizer:{type(l_optimizer).__name__}')

        if len(self.t_optimizer_for_cp) == 0:
            print('None of traditional optimizer')
        else:
            print(f'there are {len(self.t_optimizer_for_cp)} traditional optimizer')
            for t_optmizer in self.t_optimizer_for_cp:
                print(f't_optmizer:{type(t_optmizer).__name__}')

        for agent in self.agent_for_cp:
            self.test_results['T2'][type(agent).__name__] = 0.
        for optimizer in self.t_optimizer_for_cp:
            self.test_results['T2'][type(optimizer).__name__] = 0.

        for problem in self.test_set:
            self.test_results['cost'][problem.__str__()] = {}
            self.test_results['fes'][problem.__str__()] = {}
            for agent in self.agent_for_cp:
                self.test_results['cost'][problem.__str__()][type(agent).__name__] = []  # 51个np.array
                self.test_results['fes'][problem.__str__()][type(agent).__name__] = []  # 51个数字
            for optimizer in self.t_optimizer_for_cp:
                self.test_results['cost'][problem.__str__()][type(optimizer).__name__] = []  # 51个np.array
                self.test_results['fes'][problem.__str__()][type(optimizer).__name__] = []  # 51个数字

    def test(self):
        print(f'start testing: {self.config.run_time}')
        # calculate T0
        T0 = cal_t0(self.config.dim, self.config.maxFEs)
        self.test_results['T0'] = T0
        # calculate T1
        T1 = cal_t1(self.test_set[0], self.config.dim, self.config.maxFEs)
        self.test_results['T1'] = T1
        pbar_len = (len(self.t_optimizer_for_cp) + len(self.agent_for_cp)) * self.test_set.N * 51
        with tqdm(range(pbar_len), desc='Testing') as pbar:
            for i,problem in enumerate(self.test_set):
                # run learnable optimizer
                for agent,optimizer in zip(self.agent_for_cp,self.l_optimizer_for_cp):
                    T2 = 0
                    for run in range(51):
                        start = time.perf_counter()
                        np.random.seed(self.seed[run])
                        # construct an ENV for (problem,optimizer)
                        env = PBO_Env(problem,optimizer)
                        info = agent.rollout_episode(env)
                        cost = info['cost']
                        while len(cost) < 51:
                            cost.append(cost[-1])
                        fes = info['fes']
                        end = time.perf_counter()
                        if i == 0:
                            T2 += (end - start) * 1000  # ms
                        self.test_results['cost'][problem.__str__()][type(agent).__name__].append(cost)
                        self.test_results['fes'][problem.__str__()][type(agent).__name__].append(fes)
                        pbar_info = {'problem': problem.__str__(),
                                     'optimizer': type(agent).__name__,
                                     'run': run,
                                     'cost': cost[-1],
                                     'fes': fes}
                        pbar.set_postfix(pbar_info)
                        pbar.update(1)
                    if i == 0:
                        self.test_results['T2'][type(agent).__name__] = T2/51
                # run traditional optimizer
                for optimizer in self.t_optimizer_for_cp:
                    T2 = 0
                    for run in range(51):
                        start = time.perf_counter()
                        np.random.seed(self.seed[run])
                        info = optimizer.run_episode(problem)
                        cost = info['cost']
                        while len(cost) < 51:
                            cost.append(cost[-1])
                        fes = info['fes']
                        end = time.perf_counter()
                        if i == 0:
                            T2 += (end - start) * 1000  # ms
                        self.test_results['cost'][problem.__str__()][type(optimizer).__name__].append(cost)
                        self.test_results['fes'][problem.__str__()][type(optimizer).__name__].append(fes)
                        pbar_info = {'problem': problem.__str__(),
                                     'optimizer': type(optimizer).__name__,
                                     'run': run,
                                     'cost': cost[-1],
                                     'fes': fes, }
                        pbar.set_postfix(pbar_info)
                        pbar.update(1)
                    if i == 0:
                        self.test_results['T2'][type(optimizer).__name__] = T2/51
        with open(self.log_dir + 'test.pkl', 'wb') as f:
            pickle.dump(self.test_results, f, -1)


def rollout(config):
    print(f'start rollout: {config.run_time}')

    train_set,_=construct_problem_set(config)
    agent_load_dir=config.agent_load_dir
    n_checkpoint=config.n_checkpoint

    train_rollout_results = {'cost': {},
                             'fes': {},
                             'return':{}}

    agent_for_rollout=config.agent_for_rollout

    load_agents={}
    for agent_name in agent_for_rollout:
        load_agents[agent_name]=[]
        for checkpoint in range(0,n_checkpoint+1):
            file_path = agent_load_dir+ agent_name + '/' + 'checkpoint'+str(checkpoint) + '.pkl'
            with open(file_path, 'rb') as f:
                load_agents[agent_name].append(pickle.load(f))

    optimizer_for_rollout=[]
    for optimizer_name in config.optimizer_for_rollout:
        optimizer_for_rollout.append(eval(optimizer_name)(copy.deepcopy(config)))
    for problem in train_set:
        train_rollout_results['cost'][problem.__str__()] = {}
        train_rollout_results['fes'][problem.__str__()] = {}
        train_rollout_results['return'][problem.__str__()] = {}
        for agent_name in agent_for_rollout:
            train_rollout_results['cost'][problem.__str__()][agent_name] = []  # n_checkpoint 个np.array
            train_rollout_results['fes'][problem.__str__()][agent_name] = []  # n_checkpoint 个数字
            train_rollout_results['return'][problem.__str__()][agent_name]=[]

    pbar_len = (len(agent_for_rollout)) * train_set.N * (n_checkpoint+1)
    with tqdm(range(pbar_len), desc='Rollouting') as pbar:
        for agent_name,optimizer in zip(agent_for_rollout,optimizer_for_rollout):
            return_list=[]  # n_checkpoint + 1
            agent=None
            for checkpoint in range(0,n_checkpoint+1):
                agent=load_agents[agent_name][checkpoint]
                # return_sum=0
                for problem in train_set:
                    env = PBO_Env(problem,optimizer)
                    info = agent.rollout_episode(env)
                    cost=info['cost']
                    while len(cost)<51:
                        cost.append(cost[-1])
                    fes=info['fes']
                    R=info['return']

                    train_rollout_results['cost'][problem.__str__()][agent_name].append(cost)
                    train_rollout_results['fes'][problem.__str__()][agent_name].append(fes)
                    train_rollout_results['return'][problem.__str__()][agent_name].append(R)

                    pbar_info = {'problem': problem.__str__(),
                                'agent': type(agent).__name__,
                                'checkpoint': checkpoint,
                                'cost': cost[-1],
                                'fes': fes, }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
            
    log_dir=config.rollout_log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + 'rollout.pkl', 'wb') as f:
        pickle.dump(train_rollout_results, f, -1)