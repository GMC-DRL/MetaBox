import os.path
import pickle
import copy
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Optional, Union
from utils import construct_problem_set
from tester import cal_t0, test_for_random_search
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
    L2L_Optimizer
)


def get_random_baseline(random: dict, fes: Optional[Union[int, float]]):
    baseline = {}
    if isinstance(random['T1'], dict):
        baseline['complexity_avg'] = np.log10(1 / (random['T2']['Random_search'] - random['T1']['Random_search']) / random['T0'])
    else:
        baseline['complexity_avg'] = np.log10(1 / (random['T2']['Random_search'] - random['T1']) / random['T0'])
    baseline['complexity_std'] = 0.005

    problems = random['cost'].keys()
    avg = []
    std = []
    for problem in problems:
        g = np.log10(fes/np.array(random['fes'][problem]['Random_search']))
        avg.append(g.mean())
        std.append(g.std())
    baseline['fes_avg'] = np.mean(avg)
    baseline['fes_std'] = np.mean(std)

    avg = []
    std = []
    for problem in problems:
        g = np.log10(1 / (np.array(random['cost'][problem]['Random_search'])[:, -1] + 1))
        avg.append(g.mean())
        std.append(g.std())
    baseline['cost_avg'] = np.mean(avg)
    baseline['cost_std'] = np.mean(std)
    return baseline


def aei_metric(data: dict, random: dict, maxFEs=20000):
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
            if isinstance(data['T1'], dict):
                t1 = data['T1'][key]
            else:
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
    random_results = test_for_random_search(config)
    aei = aei_metric(test_results, random_results, config.maxFEs)
    print(f'AEI: {aei}')
    print(f'MGD({config.problem_from}_{config.difficulty_from}, {config.problem_to}_{config.difficulty_to}) of {config.agent}: '
          f'{100 * (1 - aei[config.agent+"_from"] / aei[config.agent+"_to"])}%')


def mte_test(config):
    print(f'start MTE_test: {config.run_time}')
    pre_train_file = config.pre_train_rollout
    scratch_file = config.scratch_rollout
    agent = config.agent
    # pre_train_file = 'rlepso_transfer_to_bbob/20230609T133345_bbob_easy_10D(noisy_to_bbob)/rollout.pkl'
    # scratch_file = 'rlepso_transfer_to_bbob/20230604T190807_bbob_easy_10D(只在bbob上训练)/rollout.pkl'
    # agent = 'RLEPSO_Agent'
    min_max = False
    other_pre_train_file = None

    # preprocess data for agent
    def preprocess(file, agent):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        # aggregate all problem's data together
        returns = data['return']
        results = None
        i = 0
        for problem in returns.keys():
            if i == 0:
                results = np.array(returns[problem][agent])
            else:
                results = np.concatenate([results, np.array(returns[problem][agent])], axis=1)
            i += 1
        return np.array(results)

    bbob_data = preprocess(pre_train_file, agent)
    noisy_data = preprocess(scratch_file, agent)
    # calculate min_max avg
    temp = np.concatenate([bbob_data, noisy_data], axis=1)
    if min_max:
        temp_ = (temp - temp.min(-1)[:, None]) / (temp.max(-1)[:, None] - temp.min(-1)[:, None])
    else:
        temp_ = temp
    bd, nd = temp_[:, :90], temp_[:, 90:]
    checkpoints = np.hsplit(bd, 18)
    g = []
    for i in range(18):
        g.append(checkpoints[i].tolist())
    checkpoints = np.array(g)
    avg = bd.mean(-1)
    avg = savgol_filter(avg, 13, 5)
    std = np.mean(np.std(checkpoints, -1), 0) / np.sqrt(5)
    checkpoints = np.hsplit(nd, 18)
    g = []
    for i in range(18):
        g.append(checkpoints[i].tolist())
    checkpoints = np.array(g)
    std_ = np.mean(np.std(checkpoints, -1), 0) / np.sqrt(5)
    avg_ = nd.mean(-1)
    avg_ = savgol_filter(avg_, 13, 5)
    plt.figure(figsize=(40, 15))
    plt.subplot(1, 3, (2, 3))
    x = np.arange(21)
    x = (1.5e6 / x[-1]) * x
    idx = np.argmax(avg_) + 1
    idx = 21
    smooth = 1
    s = np.zeros(21)
    a = s[0] = avg[0]
    norm = smooth + 1
    for i in range(1, 21):
        a = a * smooth + avg[i]
        s[i] = a / norm if norm > 0 else a
        norm *= smooth
        norm += 1

    s_ = np.zeros(21)
    a = s_[0] = avg_[0]
    norm = smooth + 1
    for i in range(1, 21):
        a = a * smooth + avg_[i]
        s_[i] = a / norm if norm > 0 else a
        norm *= smooth
        norm += 1
    #     s = savgol_filter(s,13,5)
    #     s_ = savgol_filter(s_,13,5)

    #     plt.plot(x[:idx],avg[:idx], label='pre-train', marker='*', markersize=15, markevery=1, c='blue')
    #     plt.fill_between(x[:idx], avg[:idx] - std[:idx], avg[:idx]+std[:idx], alpha=0.1, facecolor='blue')
    #     plt.plot(x[:idx],avg_[:idx], label='scratch', marker='*', markersize=15, markevery=1, c='red')
    #     plt.fill_between(x[:idx], avg_[:idx] - std_[:idx], avg_[:idx]+std_[:idx], alpha=0.1, facecolor='red')
    plt.plot(x[:idx], s[:idx], label='pre-train', marker='*', markersize=30, markevery=1, c='blue', linewidth=5)
    plt.fill_between(x[:idx], s[:idx] - std[:idx], s[:idx] + std[:idx], alpha=0.2, facecolor='blue')
    plt.plot(x[:idx], s_[:idx], label='scratch', marker='*', markersize=30, markevery=1, c='red', linewidth=5)
    plt.fill_between(x[:idx], s_[:idx] - std_[:idx], s_[:idx] + std_[:idx], alpha=0.2, facecolor='red')
    # Search MTE
    scratch = s_[:idx]
    pretrain = s[:idx]
    topx = np.argmax(scratch)
    topy = scratch[topx]
    T = topx / 21
    t = 0
    if pretrain[0] < topy:
        for i in range(1, 21):
            if pretrain[i - 1] < topy <= pretrain[i]:
                t = ((topy - pretrain[i - 1]) / (pretrain[i] - pretrain[i - 1]) + i - 1) / 21
                break
    if np.max(pretrain[-1]) < topy:
        t = 1
    MTE = 1 - t / T

    def name_translate(problem):
        if problem in ['bbob', 'bbob-torch']:
            return 'Synthetic'
        elif problem in ['bbob-noisy', 'bbob-noisy-torch']:
            return 'Noisy-Synthetic'
        elif problem in ['protein', 'protein-torch']:
            return 'Protein-Docking'
        else:
            raise ValueError(problem + ' is not defined!')

    print(f'MTE({name_translate(config.problem_from)}_{config.difficulty_from}, {name_translate(config.problem_to)}_{config.difficulty_to}) of {config.agent}: '
          f'{MTE}')

    # plt.plot([x[4], x[4]], [s_[0]-0.2, s_[4]], lw=4, ls='--', c='r')
    # d = 0.015
    # plt.plot([x[3]+d, x[3]+d], [s_[0]-0.2, s_[3]], lw=4, ls='--', c='b')

    # plt.text(x[3] - 0.12 * 1e6, 5.3, 't = 0.24', fontsize=50)
    # plt.text(x[4] - 0.13 * 1e6, 5.1, 'T = 0.30', fontsize=50)
    # plt.text(0.45 * 1e6, 5.7, 'MTE = (1 - t/T) = 0.2', fontsize=50)

    # plt.plot([x[3]+0.01 * 1e6, 0.45 * 1e6], [5.3+0.05, 5.7], lw=4, c='b')
    # plt.plot([x[4], 0.45 * 1e6], [5.1+0.05, 5.7], lw=4, c='r')
    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontsize(45)
    plt.xticks(fontsize=45, )
    plt.yticks(fontsize=45)
    #     plt.xlim(0,1.5)
    #     plt.ylim(3.2,4.5)
    # plt.grid()
    plt.legend(loc=0, fontsize=60)
    plt.xlabel('Learning Steps', fontsize=55)
    plt.ylabel('Avg Return', fontsize=55)
    # plt.ylim(s_[0] - 0.2, 7.7)

    plt.title(f'Fine-tuning ({name_translate(config.problem_from)} $\\rightarrow$ {name_translate(config.problem_to)})',
              fontsize=60)
    plt.tight_layout()
    plt.grid()
    plt.subplots_adjust(wspace=0.2)
    if not os.path.exists(config.mte_test_log_dir):
        os.makedirs(config.mte_test_log_dir)
    plt.savefig(f'{config.mte_test_log_dir}/MTE_{agent}.png', bbox_inches='tight')
