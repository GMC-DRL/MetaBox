from typing import Any

import numpy as np


class Info:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get(self):
        return self.kwargs

    def add(self, key, value):
        self.kwargs[key] = value


# random walk sampler
def rw_sampling(group):
    gs,dim = group.shape
    Pmax = np.max(group, axis=0)
    Pmin = np.min(group, axis=0)
    size = Pmax - Pmin
    walk = []
    walk.append(np.random.rand(dim))
    for i in range(1,gs):
        step = np.random.rand(dim) * size
        tmp_res = walk[i-1] + step
        walk.append(tmp_res-np.floor(tmp_res))
    return np.array(walk)


# compare diff between 2 characters
def compare_diff(diff, epsilon):
    S_epsilon = []
    label_counts = np.zeros(6) + 1
    for i in range(len(diff)):
        if diff[i] < -1 * epsilon:
            S_epsilon.append(-1)
        elif diff[i] > epsilon:
            S_epsilon.append(1)
        else:
            S_epsilon.append(0)
    for i in range(len(S_epsilon)-1):
        if S_epsilon[i] == -1 and S_epsilon[i+1] == 0:
            label_counts[0] += 1
        if S_epsilon[i] == -1 and S_epsilon[i+1] == 1:
            label_counts[1] += 1
        if S_epsilon[i] == 1 and S_epsilon[i+1] == 0:
            label_counts[2] += 1
        if S_epsilon[i] == 1 and S_epsilon[i+1] == -1:
            label_counts[3] += 1
        if S_epsilon[i] == 0 and S_epsilon[i+1] == -1:
            label_counts[4] += 1
        if S_epsilon[i] == 0 and S_epsilon[i+1] == 1:
            label_counts[5] += 1
    probs = label_counts / np.sum(label_counts)
    entropy = -1 * np.sum(probs * (np.log(probs)/np.log(6)))
    return entropy


"""
following 4 functions is built for extracting problem features
which is originated from paper "https://doi.org/10.1016/j.asoc.2021.107678"
these well designed features will further participate in total feature of encodings 
"""


# calculate FDC of group
def cal_fdc(group, costs):
    opt_x = sorted(zip(group, costs), key=lambda x: x[1])[0][0]
    ds = np.sum((group - opt_x) ** 2, axis=1)
    fs = 1/(costs+(1e-8))
    C_fd = ((fs - fs.mean()) * (ds - ds.mean())).mean()
    delta_f = ((fs - fs.mean()) ** 2).mean()
    delta_d = ((ds - ds.mean()) ** 2).mean()
    return C_fd/((delta_d * delta_f) + (1e-8))


# calculate RIE(ruggedness of information entropy) of group
def cal_rf(costs):
    diff = costs[1:] - costs[:len(costs) - 1]
    epsilon_max = np.max(diff)
    entropy_list = []
    factor = 128
    while factor >= 1:
        entropy_list.append(compare_diff(diff,epsilon_max/factor))
        factor /= 2
    entropy_list.append(compare_diff(diff,0))
    return np.max(entropy_list)


# calculate Auto-correlation of group fitness
def cal_acf(costs):
    temp = costs[:-1]
    temp_shift = costs[1:]
    fmean = np.mean(costs)
    cov = np.sum((temp - fmean)*(temp_shift - fmean))
    v = np.sum((costs - fmean)**2)
    return cov/(v+(1e-8))


# calculate local fitness landscape metric
def cal_nopt(group, costs):
    opt_x = sorted(zip(group, costs), key=lambda x: x[1])[0][0]
    ds = np.sum((group - opt_x) ** 2, axis=1)
    costs_sorted, _ = zip(*sorted(zip(costs,ds),key=lambda x:x[1]))
    counts = 0
    for i in range(len(costs) - 1):
        if costs_sorted[i+1] <= costs_sorted[i]:
            counts += 1
    return counts/len(costs)


# dispersion metric and ratio
def dispersion(group, costs):  # [a] + [f]
    gs, dim = group.shape
    group_sorted = group[np.argsort(costs)]
    group_sorted = (group_sorted / 200) + 0.5
    diam = np.sqrt(dim)
    # calculate max and avg distances
    max_dis = 0
    disp = 0
    for i in range(1, gs):
        shift_group = np.concatenate((group_sorted[i:], group_sorted[:i]), 0)
        distances = np.sqrt(np.sum((group_sorted - shift_group) ** 2, -1))
        disp += np.sum(distances)
        max_dis = np.maximum(max_dis, np.max(distances))
    disp /= gs ** 2
    # calculate avg distance of 10% individuals
    disp10 = 0
    gs10 = gs * 10 // 100
    group_sorted = group_sorted[:gs10]
    for i in range(1, gs10):
        shift_group = np.concatenate((group_sorted[i:], group_sorted[:i]), 0)
        disp10 += np.sum(np.sqrt(np.sum((group_sorted - shift_group) ** 2, -1)))
    disp10 /= gs10 ** 2
    return disp10 - disp, max_dis / diam


def population_evolvability(group_cost, sample_costs):  # [i]
    fbs = np.min(sample_costs, -1)
    n_plus = np.sum(fbs < np.min(group_cost))
    gs = group_cost.shape[0]
    if n_plus == 0:
        return 0
    evp = np.sum(np.fabs(fbs - np.min(group_cost)) / gs / (np.std(group_cost) + 1e-8)) / sample_costs.shape[0]
    return evp


def negative_slope_coefficient(group_cost, sample_cost):  # [j]
    gs = sample_cost.shape[0]
    m = 10
    gs -= gs % m  # to be divisible
    if gs < m:  # not enough costs for m dividing
        return 0
    sorted_cost = np.array(sorted(list(zip(group_cost[:gs], sample_cost[:gs]))))
    sorted_group = sorted_cost[:, 0].reshape(m, -1)
    sorted_sample = sorted_cost[:, 1].reshape(m, -1)
    Ms = np.mean(sorted_group, -1)
    Ns = np.mean(sorted_sample, -1)
    nsc = np.minimum((Ns[1:] - Ns[:-1]) / (Ms[1:] - Ms[:-1] + 1e-8), 0)
    return np.sum(nsc)


def average_neutral_ratio(group_cost, sample_costs, eps=1):
    gs = sample_costs.shape[1]
    dcost = np.fabs(sample_costs - group_cost[:gs])
    return np.mean(np.sum(dcost < eps, 0) / sample_costs.shape[0])


def non_improvable_worsenable(group_cost, sample_costs):
    gs = sample_costs.shape[1]
    NI = 1 - np.count_nonzero(np.sum(group_cost[:gs] > sample_costs, -1)) / sample_costs.shape[0]
    NW = 1 - np.count_nonzero(np.sum(group_cost[:gs] < sample_costs, -1)) / sample_costs.shape[0]
    return NI, NW


def average_delta_fitness(group_cost, sample_costs):
    gs = sample_costs.shape[1]
    return np.sum(sample_costs - group_cost[:gs]) / sample_costs.shape[0] / gs / np.max(group_cost[:gs])


# Online score judge, get performance sequences from running algorithms and calculate scores
def score_judge(results):
    alg_num = len(results)
    n = 30
    score = np.zeros(alg_num)
    for problem in list(results[0].keys()):
        for config in list(results[0][problem].keys()):
            Fevs = np.array([])
            FEs = np.array([])
            for alg in range(alg_num):
                Fevs = np.append(Fevs, results[alg][problem][config]['Fevs'][:, -1])
                FEs = np.append(FEs, results[alg][problem][config]['success_fes'])
            nm = n*alg_num
            order = sorted(list(zip(FEs, Fevs, np.arange(nm))))
            for i in range(nm):
                score[order[i][2] // n] += nm - i
            score -= n * (n + 1) / 2
    return score


# score judge, get performance sequences from result files
def score_judge_from_file(result_paths, num_problem):
    alg_num = len(result_paths)
    n = 30
    nm = n * alg_num
    score = np.zeros(alg_num)
    fpts = []
    for i in range(alg_num):
        fpts.append(open(result_paths[i], 'r'))
    for p in range(num_problem):
        Fevs = np.array([])
        FEs = np.array([])
        for alg in range(alg_num):
            fpt = fpts[alg]
            text = fpt.readline()
            while text != 'Function error values:\n':
                text = fpt.readline()
            for i in range(n):
                text = fpt.readline().split()
                success_fes = float(text[-1])
                error_value = float(text[-2])
                Fevs = np.append(Fevs, error_value)
                FEs = np.append(FEs, success_fes)
        order = sorted(list(zip(FEs, Fevs, np.arange(nm))))
        for i in range(nm):
            score[order[i][2] // n] += nm - i
        score -= n * (n + 1) / 2
    return score



