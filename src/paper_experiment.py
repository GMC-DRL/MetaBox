import os.path
import pickle
import copy
import time
import numpy as np
from tqdm import tqdm
from utils import construct_problem_set
from tester import cal_t0
from environment import PBO_Env
from logger import Logger
from agent import (
    DE_DDQN_Agent,
    DEDQN_Agent,
    RL_HPSDE_Agent,
    LDE_Agent,
    QLPSO_Agent,
    RLEPSO_Agent,
    RL_PSO_Agent,
    L2L_Agent
)
from optimizer import (
    DE_DDQN_Optimizer,
    DEDQN_Optimizer,
    RL_HPSDE_Optimizer,
    LDE_Optimizer,
    QLPSO_Optimizer,
    RLEPSO_Optimizer,
    RL_PSO_Optimizer,
    L2L_Optimizer,
    Random_search
)


def aei_metric(self, data: dict, random: dict, maxFEs=20000):
    baseline = get_random_baseline(random, maxFEs)
    if 'complexity' not in data.keys():
        data['complexity'] = {}
    avg = baseline['complexity_avg']
    std = baseline['complexity_std']
    results_complex = {}
    problems = data['fes'].keys()
    agents = data['fes'][list(problems)[0]].keys()

    for key in agents:
        if key not in data['complexity'].keys():
            t0 = data['T0']
            t1 = data['T1']
            t2 = data['T2'][key]
            data['complexity'][key] = ((t2 - t1) / t0)
        results_complex[key] = np.exp((np.log10(1/data['complexity'][key]) - avg)/std/1000 * 1)

    fes_data = data['fes']

    avg = baseline['fes_avg']
    std = baseline['fes_std']
    results_fes = {}
    log_fes = {}
    for agent in agents:
        fes_problem = []
        for problem in problems:
            if agent in ['L2L_Agent','BayesianOptimizer']:
                    fes_ =np.log10(100/np.array(fes_data[problem][agent]))
            else:
                    fes_ =np.log10(maxFEs/np.array(fes_data[problem][agent]))
            fes_problem.append(fes_.mean())
        log_fes[agent] = np.mean(fes_problem)
        results_fes[agent] = np.exp((log_fes[agent] - avg) * 1)

    cost_data = data['cost']
    avg = baseline['cost_avg']
    std = baseline['cost_std']
    results_cost = {}
    log_cost = {}
    for agent in agents:
        costs_problem = []
        for problem in problems:
            cost_ = np.log10(1/(np.array(cost_data[problem][agent])[:, -1]+1))
            costs_problem.append(cost_.mean())
        log_cost[agent] = np.mean(costs_problem)
        results_cost[agent] = np.exp((log_cost[agent] - avg) * 1)

    results = {}
    for agent in agents:
        key = agent
        if agent == 'Random_search':
            continue
        results[key] = results_complex[agent] * results_cost[agent] * results_fes[agent]
    return results


def mgd_test(config):
    print(f'start MGD_test: {config.run_time}')
    # get test set
    config.problem = config.problem_to
    config.difficulty = config.difficulty_to
    _, test_set = construct_problem_set(config)
    # get agents
    with open(config.model_from, 'rb') as f:
        agent_from = pickle.load(f)
    with open(config.model_to, 'rb') as f:
        agent_to = pickle.load(f)
    # get optimizer
    l_optimizer = eval(config.optimizer)(copy.deepcopy(config))
    # initialize the dataframe for logging
    test_results = {'cost': {},
                    'fes': {},
                    'T0': 0.,
                    'T1': {},
                    'T2': {}}
    agent_name_list = [f'{config.agent}_from', f'{config.agent}_to']
    for agent_name in agent_name_list:
        test_results['T1'][agent_name] = 0.
        test_results['T2'][agent_name] = 0.
    for problem in test_set:
        test_results['cost'][problem.__str__()] = {}
        test_results['fes'][problem.__str__()] = {}
        for agent_name in agent_name_list:
            test_results['cost'][problem.__str__()][agent_name] = []  # 51 np.arrays
            test_results['fes'][problem.__str__()][agent_name] = []  # 51 scalars
    # calculate T0
    test_results['T0'] = cal_t0(config.dim, config.maxFEs)
    # begin mgd_test
    seed = range(51)
    pbar_len = len(agent_name_list) * len(test_set) * 51
    with tqdm(range(pbar_len), desc='MGD_Test') as pbar:
        for i, problem in enumerate(test_set):
            # run model_from and model_to
            for agent_id, agent in enumerate([agent_from, agent_to]):
                T1 = 0
                T2 = 0
                for run in range(51):
                    start = time.perf_counter()
                    np.random.seed(seed[run])
                    # construct an ENV for (problem,optimizer)
                    env = PBO_Env(problem, l_optimizer)
                    info = agent.rollout_episode(env)
                    cost = info['cost']
                    while len(cost) < 51:
                        cost.append(cost[-1])
                    fes = info['fes']
                    end = time.perf_counter()
                    if i == 0:
                        T1 += env.problem.T1
                        T2 += (end - start) * 1000  # ms
                    test_results['cost'][problem.__str__()][agent_name_list[agent_id]].append(cost)
                    test_results['fes'][problem.__str__()][agent_name_list[agent_id]].append(fes)
                    pbar_info = {'problem': problem.__str__(),
                                 'optimizer': agent_name_list[agent_id],
                                 'run': run,
                                 'cost': cost[-1],
                                 'fes': fes}
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                if i == 0:
                    test_results['T1'][agent_name_list[agent_id]] = T1 / 51
                    test_results['T2'][agent_name_list[agent_id]] = T2 / 51
    if not os.path.exists(config.mgd_test_log_dir):
        os.makedirs(config.mgd_test_log_dir)
    with open(config.mgd_test_log_dir + 'test.pkl', 'wb') as f:
        pickle.dump(test_results, f, -1)
    # logger = Logger(config)
    # aei = logger.aei_metric(test_results)
    # print(aei)
