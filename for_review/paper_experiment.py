import numpy as np
import pickle
import os
from typing import Optional, Union
import matplotlib.pyplot as plt
from logger import *
from matplotlib import cm
from scipy.signal import savgol_filter


plt.rc('font', family='Times New Roman')
plt.rcParams['figure.dpi'] = 300


logger = Logger(None)


def label_replace(agent, grid=False):
    key = agent
    if agent == 'L2L_Agent':
        key = 'RNN-OI'
    if agent == 'DE_DDQN_Agent':
        key = 'DEDDQN'
    if agent == 'DEAP_CMAES':
        key = 'CMA-ES'    
    if agent == 'DEAP_DE':
        key = 'DE'
    if agent == 'DEAP_PSO':
        key = 'PSO'
    if agent == 'GL_PSO':
        key = 'GLPSO'
    if agent == 'sDMS_PSO':
        key = 'sDMSPSO'
    if agent == 'BayesianOptimizer':
        key = 'BO'
    if agent == 'NL_SHADE_LBC':
        key = 'NL-SHADE-LBC'
    if agent == 'RL_HPSDE_Agent':
        key = 'RLHPSDE'
    if agent == 'RL_PSO_Agent':
        key = 'RLPSO'
    if agent == 'LDE_Agent' and grid:
        key = 'RE.-50'
    if agent == 'LDE_Agent_reinforce_30':
        key = 'RE.-30'
    if agent == 'LDE_Agent_ppo_50':
        key = 'PPO-50'
    if agent == 'LDE_Agent_ppo_30':
        key = 'PPO-30'
    return key


def preprocess(file,agent):
    with open(file,'rb') as f:
        data = pickle.load(f)
    # aggregate all problem's data toghether
    returns = data['return']
    results = None
    i = 0
    for problem in returns.keys():
        if i == 0:
            results = np.array(returns[problem][agent])
        else:
            results = np.concatenate([results, np.array(returns[problem][agent])], axis=1)
        i+=1
    return np.array(results)


def draw_metric_hist():
    plt.figure(figsize=(70, 40))

    # Load data
    with open('AEI_data/bbob-random.pkl', 'rb') as f:
        random_bbob = pickle.load(f)

    with open('AEI_data/bbob_easy.pkl', 'rb') as f:
        results0 = pickle.load(f)

    with open('AEI_data/bbob_difficult.pkl', 'rb') as f:
        results1 = pickle.load(f)
    aei = logger.aei_metric(results0, random_bbob)
    X0, Y0 = aei.keys(), aei.values()
    aei = logger.aei_metric(results1, random_bbob)
    X1, Y1 = aei.keys(), aei.values()

    # Adjust agent order
    X0 = list(X0)
    X0[5], X0[6] = X0[6], X0[5]
    Y0 = list(Y0)
    Y0[5], Y0[6] = Y0[6], Y0[5]
    Y1 = list(Y1)
    Y1[5], Y1[6] = Y1[6], Y1[5]

    X = np.arange(len(list(X0))) *2.5
    RC = 16.25
    CS = 41.25
    width = 0.4

    # Plot Synthetic
    plt.subplot(2, 1, 1)
    plt.bar(X - width, Y0, label='Synthetic-easy', color='dodgerblue')
    for a,b in zip(X, Y0):
        plt.text(a - width - 0.15, b+0.1, '%.2f' % b, ha='center', fontsize=55)
        
    plt.bar(X + width, Y1, label='Synthetic-difficult', color='darkorange')
    for a,b in zip(X, Y1):
        plt.text(a + width + 0.15, b+0.1, '%.2f' % b, ha='center', fontsize=55)

    # for i in range(len(X0)):
    #     X0[i] = to_label(label_replace(X0[i]))

    # plt.xticks(X, labels=X0)
    plt.yticks(fontsize=60)
    plt.ylim(0, 16)
    plt.ylabel('AEI', fontsize=60)
    plt.xticks([])
    plt.legend(fontsize=60,)
    plt.plot([RC, RC], [0, 16], linewidth=8, c='black', linestyle='--')
    plt.plot([CS, CS], [0, 16], linewidth=8, c='black', linestyle='--')
    plt.text(0, 17, 'MetaBBO-RL', fontsize=70, ha='center', va='center')
    plt.text(RC+3, 17, 'Classic Optimizer', fontsize=70, ha='center', va='center')
    plt.text(CS+1.5, 17, 'MetaBBO-SL', fontsize=70, ha='center', va='center')


    # Plot Protein-Docking
    plt.subplot(2, 1, 2)
    with open('AEI_data/protein-random.pkl', 'rb') as f:
        random_protein = pickle.load(f)

    with open('AEI_data/protein_easy.pkl', 'rb') as f:
        results0 = pickle.load(f)

    with open('AEI_data/protein_difficult.pkl', 'rb') as f:
        results1 = pickle.load(f)
        
    aei = logger.aei_metric(results0, random_protein)
    X0, Y0 = aei.keys(), aei.values()
    aei = logger.aei_metric(results1, random_protein)
    X1, Y1 = aei.keys(), aei.values()

    # Adjust agent order
    X0 = list(X0)
    X0[5], X0[6] = X0[6], X0[5]
    Y0 = list(Y0)
    Y0[5], Y0[6] = Y0[6], Y0[5]
    Y1 = list(Y1)
    Y1[5], Y1[6] = Y1[6], Y1[5]

    plt.bar(X - width, Y0, label='Protein-easy', color='dodgerblue')
    for a,b in zip(X, Y0):
        plt.text(a - width - 0.2, b+0.05, '%.2f' % b, ha='center', fontsize=55)
        
    plt.bar(X + width, Y1, label='Protein-difficult', color='darkorange')
    for a,b in zip(X, Y1):
        plt.text(a + width + 0.2, b+0.05, '%.2f' % b, ha='center', fontsize=55)

    for i in range(len(X0)):
        X0[i] = to_label(label_replace(X0[i]))

    plt.xticks(X, labels=X0)
    plt.xticks(rotation=45, fontsize=60)
    plt.yticks(fontsize=60)
    plt.ylim(0, 5)
    plt.ylabel('AEI', fontsize=60)
    plt.legend(fontsize=60,)
    plt.plot([RC, RC], [0, 5], linewidth=8, c='black', linestyle='--')
    plt.plot([CS, CS], [0, 3.9], linewidth=8, c='black', linestyle='--')

    plt.subplots_adjust(hspace=0.05)
    plt.savefig('pics/std_metric_hist_all.pdf', bbox_inches='tight')
    plt.savefig('pics/std_metric_hist_all.png', bbox_inches='tight')


def draw_grid_search(agent_name, labels=None):
    with open(f'Grid_Search_data/{agent_name}/test/test.pkl', 'rb') as f:
        rptest = pickle.load(f)
        
    with open(f'Grid_Search_data/{agent_name}/rollout/rollout.pkl', 'rb') as f:
        rproll = pickle.load(f)

    grids = []
    Y = {}
    for key in rptest['cost'][list(rptest['cost'].keys())[0]].keys():
        Y[key] = {'mean': [], 'std': []}
        grids.append(key)
    if labels is None:
        labels = grids
    print(grids)

    fig = plt.figure(figsize=(70, 15))
    title_font = 80
    xy_font = 70
    tick_font = 55
    legend_font = 50

    # Plot Return
    plt.subplot(1, 5, (1, 2))
    plt.title('The average return', fontsize=title_font)
    returns, stds = get_average_returns(rproll['return'], norm=False)
    problems = []
    for p in rproll['cost'].keys():
        problems.append(p)
    for id, agent in enumerate(list(Y.keys())):
        x = np.arange(len(returns[agent]), dtype=np.float64)
        x = (1.5e6 / x[-1]) * x
        y = returns[agent]
        # Smoothing
        s = np.zeros(y.shape[0])
        a = s[0] = y[0]
        norm = 0.8 + 1
        for i in range(1, y.shape[0]):
            a = a * 0.8 + y[i]
            s[i] = a / norm if norm > 0 else a
            norm *= 0.8
            norm += 1
        plt.plot(x, s, label=labels[id], marker='*', markersize=30, markevery=2, c=colors[id], lw=5)
        plt.fill_between(x, (s - stds[agent]), (s + stds[agent]), alpha=0.2, facecolor=colors[id])
    plt.legend(fontsize=legend_font)
    plt.xlabel('Learning Steps',fontsize=xy_font)
    plt.ylabel('Avg Return',fontsize=xy_font)
    plt.grid()
    # plt.ylim(4, 11)
    plt.tick_params(axis='both',which='major',labelsize=tick_font)
    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontsize(40)

    # Plot cost curve
    plt.subplot(1, 5, (3, 4))
    plt.title('The normalized cost curve', fontsize=title_font)

    problems = []
    for p in rptest['cost'].keys():
        problems.append(p)
    for id, agent in enumerate(list(Y.keys())):
        for p in problems:
            values = np.array(rptest['cost'][p][agent])
            values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
            values = np.log(np.maximum(values, 1e-8))
            std = np.std(values, 0)
            mean = np.mean(values, 0)
            Y[agent]['mean'].append(mean)
            Y[agent]['std'].append(std)
    for id, agent in enumerate(list(Y.keys())):
        mean = np.mean(Y[agent]['mean'], 0)
        std = np.mean(Y[agent]['std'], 0)

        X = np.arange(mean.shape[-1])
        X = np.array(X, dtype=np.float64)
        X *= (20000 / X[-1])
        plt.plot(X, mean, label=labels[id], marker='*', markevery=8, markersize=30, c=colors[id], lw=5)
        plt.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=colors[id])
    plt.grid()
    plt.xlabel('FEs',fontsize=xy_font)
    plt.ylabel('Logged Normalized Costs',fontsize=xy_font)
    plt.legend(fontsize=legend_font)
    plt.tick_params(axis='both',which='major',labelsize=tick_font)

    plt.subplot(1, 5, 5)
    with open('Grid_Search_data/bbob-random.pkl', 'rb') as f:
        random = pickle.load(f)
    with open('Grid_Search_data/bbob_easy.pkl', 'rb') as f:
        result0 = pickle.load(f)
        bbob = logger.aei_metric(result0, random)
    
    grid = logger.aei_metric(rptest, random)
    if agent_name == 'LDE_Agent':
        grid['RE.-50'] = bbob['LDE_Agent']
        
        grid['RE.-30'] = grid['LDE_Agent_reinforce_30']
        grid['PPO-50'] = grid['LDE_Agent_ppo_50']
        grid['PPO-30'] = grid['LDE_Agent_ppo_30']
        del grid['LDE_Agent'], grid['LDE_Agent_reinforce_30'], grid['LDE_Agent_ppo_50'], grid['LDE_Agent_ppo_30']
    elif agent_name == 'RLEPSO_Agent':
        grid['PPO-100'] = bbob['RLEPSO_Agent']
        
        grid['PPO-50'] = grid['RLEPSO_ppo_50']
        grid['RE.-100'] = grid['RLEPSO_reinforce_100']
        grid['RE.-50'] = grid['RLEPSO_reinforce_50']
        del grid['RLEPSO_ppo_100'], grid['RLEPSO_ppo_50'], grid['RLEPSO_reinforce_100'], grid['RLEPSO_reinforce_50']

    plt.bar(grid.keys(), grid.values(), color='dodgerblue', width=0.6)
    plt.xticks(rotation=30, fontsize=xy_font)
    for a,b in zip(grid.keys(), grid.values()):
        plt.text(a, b+0.05, '%.2f' % b, ha='center', fontsize=55)
    plt.title('The AEI scores', fontsize=title_font)
    plt.ylabel('AEI', fontsize=xy_font)
    # plt.ylim(0, 12.5)
    plt.tick_params(axis='y',which='major',labelsize=tick_font)
    plt.tick_params(axis='x',which='major',labelsize=tick_font+5)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(f'pics/grid_search_{agent_name}.pdf', bbox_inches='tight')
    plt.savefig(f'pics/grid_search_{agent_name}.png', bbox_inches='tight')


def draw_MGD_MTE(agent_name):
    values = np.zeros((3, 3))
    with open(f'MGD_data/{agent_name}/noisy-random.pkl', 'rb') as f:
        random_n = pickle.load(f)

    with open(f'MGD_data/{agent_name}/bbob-random.pkl', 'rb') as f:
        random = pickle.load(f)

    with open(f'MGD_data/{agent_name}/protein-random.pkl', 'rb') as f:
        random_p = pickle.load(f)

    with open(f'MGD_data/{agent_name}/Synthetic_to_Synthetic/test.pkl', 'rb') as f:
        b2b = pickle.load(f)
        values[0, 0] = logger.aei_metric(b2b, random, 20000)[agent_name]
        
    with open(f'MGD_data/{agent_name}/Noisy_Synthetic_to_Noisy_Synthetic/test.pkl', 'rb') as f:
        n2n = pickle.load(f)
        values[1, 1] = logger.aei_metric(n2n, random_n, 20000)[agent_name]
        
    with open(f'MGD_data/{agent_name}/Protein_to_Protein/test.pkl', 'rb') as f:
        p2p = pickle.load(f)
        values[2, 2] = logger.aei_metric(p2p, random_p, 1000)[agent_name]

    with open(f'MGD_data/{agent_name}/Synthetic_to_Protein/test.pkl', 'rb') as f:
        b2p = pickle.load(f)
        values[0, 2] = 100 * (1 - logger.aei_metric(b2p, random_p, 1000)[agent_name] / values[2, 2])

    with open(f'MGD_data/{agent_name}/Synthetic_to_Noisy_Synthetic/test.pkl', 'rb') as f:
        b2n = pickle.load(f)
        values[0, 1] = 100 * (1 - logger.aei_metric(b2n, random_n, 20000)[agent_name] / values[1, 1])
        
    with open(f'MGD_data/{agent_name}/Noisy_Synthetic_to_Synthetic/test.pkl', 'rb') as f:
        n2b = pickle.load(f)
        values[1, 0] =  100 * (1 - logger.aei_metric(n2b, random, 20000)[agent_name] / values[0, 0])
        
    with open(f'MGD_data/{agent_name}/Noisy_Synthetic_to_Protein/test.pkl', 'rb') as f:
        n2p = pickle.load(f)
        values[1, 2] = 100 * (1 - logger.aei_metric(n2p, random_p, 1000)[agent_name] / values[2, 2])
        
    with open(f'MGD_data/{agent_name}/Protein_to_Synthetic/test.pkl', 'rb') as f:
        p2b = pickle.load(f)
        values[2, 0] =  100 * (1 - logger.aei_metric(p2b, random, 20000)[agent_name] / values[0, 0])
        
    with open(f'MGD_data/{agent_name}/Protein_to_Noisy_Synthetic/test.pkl', 'rb') as f:
        p2n = pickle.load(f)
        values[2, 1] =  100 * (1 - logger.aei_metric(p2n, random_n, 20000)[agent_name] / values[1, 1])
    values *= 1 - np.eye(3)
    plt.figure(figsize=(50, 15))

    plt.subplot(1, 3, 1)
    labels = ['Synthetic', 'Noisy\nSynthetic', 'Protein\nDocking']
    plt.imshow(values, origin='lower',cmap=cm.get_cmap('RdYlGn_r'), vmin=-20, vmax=20)
    plt.xticks(np.arange(3), labels=labels, fontsize=50)
    plt.yticks(np.arange(3), labels=labels, rotation=270, va='center', ma='center', fontsize=50)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=45)
    plt.title('Zero-shot Generalization', fontsize=60, pad=50)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, '%.3f' % values[i, j] + '%', ha='center', va='center', color = 'black', fontsize=55)

    pre_train_file = f'MTE_data/{agent_name}/pretrain_rollout.pkl'
    scratch_file = f'MTE_data/{agent_name}/scratch_rollout.pkl'
    min_max = False
    bbob_data = preprocess(pre_train_file,agent_name)

    noisy_data = preprocess(scratch_file,agent_name)
    # calculate min_max avg
    temp = np.concatenate([bbob_data,noisy_data],axis=1)
    if min_max:
        temp_ = (temp - temp.min(-1)[:,None]) / (temp.max(-1)[:,None] - temp.min(-1)[:,None])
    else:
        temp_ = temp
    bd,nd = temp_[:,:90], temp_[:,90:]
    checkpoints = np.hsplit(bd,18)
    g = []
    for i in range(18):
        g.append(checkpoints[i].tolist())
    checkpoints = np.array(g)
    avg = bd.mean(-1)
    avg = savgol_filter(avg,13,5)
    std = np.mean(np.std(checkpoints,-1),0)/np.sqrt(5)
    checkpoints = np.hsplit(nd,18)
    g = []
    for i in range(18):
        g.append(checkpoints[i].tolist())
    checkpoints = np.array(g)
    std_ = np.mean(np.std(checkpoints,-1),0)/np.sqrt(5)
    avg_ = nd.mean(-1)
    avg_ = savgol_filter(avg_,13,5)
    plt.subplot(1, 3, (2,3))
    x = np.arange(21)
    x = (1.5e6 / x[-1]) * x
    idx = np.argmax(avg_)+1
    idx=21
    smooth = 1
    s = np.zeros(21)
    a = s[0] = avg[0]
    norm = smooth + 1
    for i in range(1,21):
        a = a * smooth + avg[i]
        s[i] = a / norm if norm > 0 else a
        norm *= smooth
        norm += 1

    s_ = np.zeros(21)
    a = s_[0] = avg_[0]
    norm = smooth + 1
    for i in range(1,21):
        a = a * smooth + avg_[i]
        s_[i] = a / norm if norm > 0 else a
        norm *= smooth
        norm += 1
    plt.plot(x[:idx],s[:idx], label='pre-train', marker='*', markersize=30, markevery=1, c='blue',linewidth=5)
    plt.fill_between(x[:idx], s[:idx] - std[:idx], s[:idx]+std[:idx], alpha=0.2, facecolor='blue')
    plt.plot(x[:idx],s_[:idx], label='scratch', marker='*', markersize=30, markevery=1, c='red',linewidth=5)
    plt.fill_between(x[:idx], s_[:idx] - std_[:idx], s_[:idx]+std_[:idx], alpha=0.2, facecolor='red')

    # Search MTE
    scratch = s_[:idx]
    pretrain = s[:idx]
    topx = np.argmax(scratch)
    topy = scratch[topx]
    T = topx / 21
    t = 0
    if pretrain[0] < topy:
        for i in range(1, 21):
            if pretrain[i-1] < topy <= pretrain[i]:
                t = ((topy - pretrain[i-1]) / (pretrain[i] - pretrain[i-1]) + i - 1) / 21
                break
    if np.max(pretrain[-1]) < topy:
        t = 1
    MTE = 1 - t / T
    # print('MTE =', MTE)
    # plt.plot([x[4], x[4]], [s_[0]-0.2, s_[4]], lw=4, ls='--', c='r')
    # d = 0.015
    # plt.plot([x[3]+d, x[3]+d], [s_[0]-0.2, s_[3]], lw=4, ls='--', c='b')

    # plt.text(x[3] - 0.12 * 1e6, 5.3, 't = 0.24', fontsize=50)
    # plt.text(x[4] - 0.13 * 1e6, 5.1, 'T = 0.30', fontsize=50)
    # plt.text(0.45 * 1e6, 5.7, 'MTE = (1 - t/T) = 0.2', fontsize=50)

    # plt.plot([x[3]+0.01 * 1e6, 0.45 * 1e6], [5.3+0.05, 5.7], lw=4, c='b')``
    # plt.plot([x[4], 0.45 * 1e6], [5.1+0.05, 5.7], lw=4, c='r')
    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontsize(45)
    plt.xticks(fontsize =45, )
    plt.yticks(fontsize =45)
    plt.legend(loc='lower right', fontsize=60)
    plt.xlabel('Learning Steps',fontsize=55)
    plt.ylabel('Avg Return',fontsize=55)
    # plt.ylim(s_[0] - 0.2, 7.7)
    plt.title(r'Fine-tuning (Noisy-Synthetic $\rightarrow$ Synthetic)', fontsize=60)
    plt.tight_layout()
    plt.grid()
    plt.subplots_adjust(wspace=0.2)

    plt.savefig(f'pics/MGD_MTE_{agent_name}.pdf', bbox_inches='tight') 
    plt.savefig(f'pics/MGD_MTE_{agent_name}.png', bbox_inches='tight') 


if __name__ == '__main__':
    draw_metric_hist()
    draw_grid_search('LDE_Agent', labels=['REINFORCE-50', 'REINFORCE-30', 'PPO-50', 'PPO-30'])
    draw_grid_search('RLEPSO_Agent', labels=['PPO-100', 'PPO-50', 'REINFORCE-100', 'REINFORCE-50'])
    draw_MGD_MTE('LDE_Agent')
    draw_MGD_MTE('RLEPSO_Agent')