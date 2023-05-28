import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Union
import argparse
params = {
    'axes.labelsize': '25',
    'xtick.labelsize': '25',
    'ytick.labelsize': '25',
    'lines.linewidth': '3',
    'legend.fontsize': '24',
    'figure.figsize': '20,11',
}
plt.rcParams.update(params)

markers = ['o', '^', '*', 'O', 'v', 'x', 'X', 'd', 'D', '.', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H']
colors = ['b', 'g', 'orange', 'r', 'purple', 'brown', 'grey', 'limegreen', 'turquoise', 'olivedrab', 'royalblue', 'darkviolet', 
          'chocolate', 'crimson', 'teal','seagreen', 'navy', 'deeppink', 'maroon', 'goldnrod', 
          ]


def get_average_returns(results: dict):
    problems=[]
    agents=[]

    for problem in results.keys():
        problems.append(problem)
    for agent in results[problems[0]].keys():
        agents.append(agent)
    avg_return={}
    # n_checkpoint=len(results[problems[0]][agents[0]])
    for agent in agents:
        avg_return[agent]=[]
        for problem in problems:
            values = results[problem][agent]
            values = (values - np.min((values))) / (np.max(values) - np.min(values))
            avg_return[agent].append(values)
        avg_return[agent] = np.mean(avg_return[agent], 0)
        # for checkpoint in range(n_checkpoint):
        #     return_sum=0
        #     for problem in problems:
        #         return_sum+=results[problem][agent][checkpoint]
        #     avg_return[agent].append(return_sum/len(problems))
    return avg_return  # {'agent':[] len = n_checkpoints}


def get_average_costs(results: dict):
    problems=[]
    agents=[]
    for problem in results.keys():
        problems.append(problem)
    for agent in results[problems[0]].keys():
        agents.append(agent)
    avg_cost={}
    # n_checkpoint=len(results[problems[0]][agents[0]])
    for agent in agents:
        avg_cost[agent]=[]
        for problem in problems:
            values = np.array(results[problem][agent])[:, -1]
            values = (values - np.min((values))) / (np.max(values) - np.min(values))
            avg_cost[agent].append(values)
        avg_cost[agent] = np.mean(avg_cost[agent], 0)
        # for checkpoint in range(n_checkpoint):
        #     return_sum=0
        #     for problem in problems:
        #         return_sum+=results[problem][agent][checkpoint]
        #     avg_return[agent].append(return_sum/len(problems))
    return avg_cost  # {'agent':[] len = n_checkpoints}


def cal_scores1(D: dict, maxf: float):
    SNE = []
    for agent in D.keys():
        values = D[agent]
        sne = 0.5 * np.sum(np.min(values, -1) / maxf)
        SNE.append(sne)
    SNE = np.array(SNE)
    score1 = (1 - (SNE - np.min(SNE)) / SNE) * 50
    return score1


def gen_algorithm_complexity_table(results: dict, out_dir: str) -> None:
    save_list=[]
    t0=results['T0']
    t1=results['T1']
    t2s=results['T2']
    ratios=[]
    t2_list=[]
    indexs=[]
    columns=['T0','T1','T2','(T2-T1)/T0']
    for key,value in t2s.items():
        indexs.append(key)
        t2_list.append(value)
        ratios.append((value-t1)/t0)
    n=len(t2_list)
    data=np.zeros((n,4))
    data[:,0]=t0
    data[:,1]=t1
    data[:,2]=t2_list
    data[:,3]=ratios
    table=pd.DataFrame(data=np.round(data,2),index=indexs,columns=columns)
    # table["number_str"] = table["number_str"].astype(long).astype(str)
    #(table)
    table.to_excel(os.path.join(out_dir,'algorithm_complexity.xlsx'))


def gen_agent_performance_table(results: dict, out_dir: str) -> None:
    total_cost=results['cost']
    table_data={}
    indexs=[]
    columns=['Worst','Best','Median','Mean','Std']
    for problem,value in total_cost.items():
        indexs.append(problem)
        problem_cost=value
        for alg,alg_cost in problem_cost.items():
            n_cost=[]
            for run in alg_cost:
                n_cost.append(run[-1])
            # if alg == 'MadDE' and problem == 'F5':
            #     for run in alg_cost:
            #         print(len(run))
            #     print(len(n_cost))
            best=np.min(n_cost)
            best=np.format_float_scientific(best,precision=3,exp_digits=3)
            worst=np.max(n_cost)
            worst=np.format_float_scientific(worst,precision=3,exp_digits=3)
            median=np.median(n_cost)
            median=np.format_float_scientific(median,precision=3,exp_digits=3)
            mean=np.mean(n_cost)
            mean=np.format_float_scientific(mean,precision=3,exp_digits=3)
            std=np.std(n_cost)
            std=np.format_float_scientific(std,precision=3,exp_digits=3)

            if not alg in table_data:
                table_data[alg]=[]
            table_data[alg].append([worst,best,median,mean,std])
    for alg,data in table_data.items():
        dataframe=pd.DataFrame(data=data,index=indexs,columns=columns)
        #print(dataframe)
        dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_table.xlsx'))


def gen_overall_tab(results: dict, out_dir: str) -> None:
    # get multi-indexes first
    problems = []
    statics = ['Obj','Gap','FEs']
    optimizers = []
    for problem in results['cost'].keys():
        problems.append(problem)
    for optimizer in results['T2'].keys():
        optimizers.append(optimizer)
    multi_columns = pd.MultiIndex.from_product(
        [problems,statics], names=('Problem', 'metric')
    )
    df_results = pd.DataFrame(np.ones(shape=(len(optimizers),len(problems)*len(statics))),
                              index=optimizers,
                              columns=multi_columns)

    # calculate baseline1 cmaes
    cmaes_obj = {}
    for problem in problems:
        blobj_problem = results['cost'][problem]['DEAP_CMAES']  # 51 * record_length
        objs = []
        for run in range(51):
            objs.append(blobj_problem[run][-1])
        cmaes_obj[problem] = sum(objs) / 51

    # calculate baseline2 random_search
    rs_obj = {}
    for problem in problems:
        blobj_problem = results['cost'][problem]['Random_search']  # 51 * record_length
        objs = []
        for run in range(51):
            objs.append(blobj_problem[run][-1])
        rs_obj[problem] = sum(objs) / 51

    # calculate each Obj
    for problem in problems:
        for optimizer in optimizers:
            obj_problem_optimizer = results['cost'][problem][optimizer]
            objs_ = []
            for run in range(51):
                objs_.append(obj_problem_optimizer[run][-1])
            avg_obj = sum(objs_)/51
            df_results.loc[optimizer,(problem,'Obj')] = np.format_float_scientific(avg_obj, precision=3, exp_digits=1)
            # calculate each Gap
            # df_results.loc[optimizer, (problem, 'Gap')] = "%.2f%%" % ((avg_obj - cmaes_obj[problem]) / cmaes_obj[problem] * 100)
            df_results.loc[optimizer, (problem, 'Gap')] = "%.3f" % (1-(rs_obj[problem]-avg_obj) / (rs_obj[problem]-cmaes_obj[problem]+1e-10) )
            # df_results.loc[optimizer, (problem, 'Gap')] = "%.3f" % (1-cmaes_obj[problem] / (avg_obj+1e-10) )
            # if avg_obj > rs_obj[problem]:
            #     print(f'optimizr:{optimizer},problem:{problem}')
            fes_problem_optimizer = np.array(results['fes'][problem][optimizer])
            df_results.loc[optimizer, (problem, 'FEs')] = np.format_float_scientific(fes_problem_optimizer.mean(), precision=3, exp_digits=1)

    df_results.to_excel(out_dir+'overall_table.xlsx')


class Logger:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0

    def draw_test_cost(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, logged: bool=False, categorized: bool=False) -> None:
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            # if logged:
            #     plt.title('log cost curve ' + name)
            # else:
            #     plt.title('cost curve ' + name)
            if not categorized:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(values)
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=agent, marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                if logged:
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'{name}_log_cost_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'{name}_cost_curve.png', bbox_inches='tight')
                plt.close()
            else:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.agent_for_cp:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(values)
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=agent, marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                if logged:
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'learnable_{name}_log_cost_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'learnable_{name}_cost_curve.png', bbox_inches='tight')
                plt.close()

                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.t_optimizer_for_cp:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(values)
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=agent, marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                
                plt.legend()
                if logged:
                    plt.ylabel('log Costs')
                    plt.savefig(output_dir + f'traditional_{name}_log_cost_curve.png', bbox_inches='tight')
                else:
                    plt.ylabel('Costs')
                    plt.savefig(output_dir + f'traditional_{name}_cost_curve.png', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_costs(self, data: dict, output_dir: str, named_agents: dict, logged: bool=False) -> None:
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            X = np.arange(51)
            X = np.array(X, dtype=np.float64)
            X *= (self.config.maxFEs / X[-1])
            X = np.log(X)/np.log(10)
            X[0] = 2
            Y = {}
            for problem in list(data.keys()):
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent])
                    values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
                    if logged:
                        values = np.log(values)
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    Y[agent]['mean'].append(mean)
                    Y[agent]['std'].append(std)

            for id, agent in enumerate(list(Y.keys())):
                mean = np.mean(Y[agent]['mean'], 0)
                std = np.mean(Y[agent]['std'], 0)
                

                ax.plot(X, mean, label=agent, marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            plt.xlabel('log10 FEs')
            plt.ylabel('Normalized Costs')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_cost_curve_logX.png', bbox_inches='tight')
        plt.close()

    def draw_concrete_performance_hist(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None) -> None:
        D = {}
        X = []
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D.keys():
                    D[agent] = []
                values = np.array(data[name][agent])
                D[agent].append(values[:, -1] / values[:, 0])

        for agent in D.keys():
            plt.figure()
            # plt.title(f'{agent} performance histgram')
            X = list(data.keys())
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])
            for a,b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)
            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            plt.ylabel('Normalized Costs')
            plt.savefig(output_dir + f'{agent}_concrete_performance_hist.png', bbox_inches='tight')

    def draw_train_return(self, data: dict, output_dir: str) -> None:
        returns = get_average_returns(data['return'])
        plt.figure()
        for agent in returns.keys():
            x = np.arange(len(returns[agent]), dtype=np.float64)
            x = (self.config.max_learning_step / x[-1]) * x
            y = returns[agent]
            s = np.zeros(y.shape[0])
            a = s[0] = y[0]
            norm = self.config.plot_smooth + 1
            for i in range(1, y.shape[0]):
                a = a * self.config.plot_smooth + y[i]
                s[i] = a / norm if norm > 0 else a
                norm *= self.config.plot_smooth
                norm += 1
            if agent not in self.color_arrangement.keys():
                self.color_arrangement[agent] = colors[self.arrange_index]
                self.arrange_index += 1
            plt.plot(x, s, label=agent, marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
            # plt.plot(x, returns[agent], label=agent)
        plt.legend()
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Normalized Return')
        plt.grid()
        plt.savefig(output_dir + f'avg_return_curve.png', bbox_inches='tight')
        plt.close()

    def draw_train_avg_cost(self, data: dict, output_dir: str) -> None:
        costs = get_average_costs(data['cost'])
        plt.figure()
        for agent in costs.keys():
            x = np.arange(len(costs[agent]), dtype=np.float64)
            x = (self.config.max_learning_step / x[-1]) * x
            y = costs[agent]
            s = np.zeros(y.shape[0])
            a = s[0] = y[0]
            norm = self.config.plot_smooth + 1
            for i in range(1, y.shape[0]):
                a = a * self.config.plot_smooth + y[i]
                s[i] = a / norm if norm > 0 else a
                norm *= self.config.plot_smooth
                norm += 1
            if agent not in self.color_arrangement.keys():
                self.color_arrangement[agent] = colors[self.arrange_index]
                self.arrange_index += 1
            plt.plot(x, s, label=agent, marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
            # plt.plot(x, returns[agent], label=agent)
        plt.legend()
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Normalized Cost')
        plt.grid()
        plt.savefig(output_dir + f'avg_cost_curve.png', bbox_inches='tight')
        plt.close()

    def draw_boxplot(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None) -> None:
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                values = np.array(data[name][agent])
                Y.append(values[:, -1])
            Y = np.transpose(Y)
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            plt.ylabel(f'{name} Cost Boxplots')
            plt.savefig(output_dir + f'{name}_boxplot.png', bbox_inches='tight')
            plt.close()

    def draw_overall_boxplot(self, data: dict, output_dir: str, ignore: Optional[list]=None) -> None:
        problems=[]
        agents=[]
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        run = len(data[problems[0]][agents[0]])
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        for ip, problem in enumerate(problems):
            for ia, agent in enumerate(agents):
                values[ia][ip] = np.array(data[problem][agent])[:, -1]
            values[:, ip, :] = (values[:, ip, :] - np.min(values[:, ip, :])) / (np.max(values[:, ip, :]) - np.min(values[:, ip, :]))
        values = values.reshape(len(agents), -1).transpose()
        
        plt.boxplot(values, labels=agents, showmeans=True, patch_artist=True, showfliers=False,
                    medianprops={'color': 'green', 'linewidth': 3}, 
                    meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                    boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                    capprops={'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    )
        plt.xticks(rotation=30, fontsize=18)
        plt.xlabel('Agents')
        plt.ylabel('Cost Boxplots')
        plt.savefig(output_dir + f'overall_boxplot.png', bbox_inches='tight')
        plt.close()

    def draw_rank_hist(self, data: dict, output_dir: str, ignore: Optional[list]=None) -> None:
        plt.figure(figsize=(30,15))
        # plt.title('rank histgram')
        D = {}
        M = []
        X = []
        Y = []
        R = []
        data, fes = data['cost'], data['fes']
        for problem in list(data.keys()):
            maxf = 0
            avg_cost = []
            avg_fes = []
            for agent in list(data[problem].keys()):
                if ignore is not None and agent in ignore:
                    continue
                if agent not in D.keys():
                    D[agent] = []
                values = np.array(data[problem][agent])[:, -1]
                D[agent].append(values)
                maxf = max(maxf, np.max(values))
                avg_cost.append(np.mean(values))
                avg_fes.append(np.mean(fes[problem][agent]))

            M.append(maxf)
            order = np.lexsort((avg_fes, avg_cost))
            rank = np.zeros(len(avg_cost))
            rank[order] = np.arange(len(avg_cost)) + 1
            R.append(rank)
        sr = 0.5 * np.sum(R, 0)
        score2 = (1 - (sr - np.min(sr)) / sr) * 50
        score1 = cal_scores1(D, M)
        score = score1 + score2
        for i, agent in enumerate(D.keys()):
            X.append(agent)
            Y.append(score[i])
        plt.bar(X, Y)
        for a,b in zip(X, Y):
            plt.text(a, b+0.5, '%.2f' % b, ha='center', fontsize=16)
        plt.xticks(rotation=30, fontsize=15)
        plt.xlabel('Agents')
        plt.ylabel('Scores')
        plt.savefig(output_dir + f'rank_hist.png', bbox_inches='tight')


def post_processing_test_statics(log_dir: str, logger: Logger) -> None:
    with open(log_dir+'test.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Genetate excel tables
    if not os.path.exists(log_dir + 'tables/'):
        os.makedirs(log_dir + 'tables/')
    gen_overall_tab(results, log_dir+'tables/')
    gen_algorithm_complexity_table(results, log_dir+'tables/')
    gen_agent_performance_table(results, log_dir+'tables/')

    # Genetate figures
    if not os.path.exists(log_dir + 'pics/'):
        os.makedirs(log_dir + 'pics/')
    logger.draw_test_cost(results['cost'],log_dir + 'pics/', logged=True, categorized=True)
    logger.draw_named_average_test_costs(results['cost'], log_dir + 'pics/', 
                                        {'RLs': ['DE_DDQN_Agent', 'RL_HPSDE_Agent', 'LDE_Agent', 'QLPSO_Agent', 'RLEPSO_Agent', 'RL_PSO_Agent', 'DEDQN_Agent'], 
                                         'RL+Tra': ['RL_HPSDE_Agent',  'LDE_Agent', 'RLEPSO_Agent', 'RL_PSO_Agent', 'DEAP_DE', 'DEAP_CMAES', 'DEAP_PSO']},
                                        logged=False)
    logger.draw_rank_hist(results,log_dir + 'pics/', ignore=['L2:_Agent', 'BayesianOptimizer'])
    logger.draw_boxplot(results['cost'],log_dir + 'pics/', ignore=['L2:_Agent', 'BayesianOptimizer'])
    logger.draw_overall_boxplot(results['cost'],log_dir + 'pics/', ignore=['L2:_Agent', 'BayesianOptimizer'])
    logger.draw_concrete_performance_hist(results['cost'],log_dir + 'pics/')


def post_processing_rollout_statics(log_dir: str, logger: Logger) -> None:
    with open(log_dir+'rollout.pkl', 'rb') as f:
        results = pickle.load(f)
    if not os.path.exists(log_dir + 'pics/'):
        os.makedirs(log_dir + 'pics/')
    logger.draw_train_return(results, log_dir + 'pics/', )
    logger.draw_train_avg_cost(results, log_dir + 'pics/', )

