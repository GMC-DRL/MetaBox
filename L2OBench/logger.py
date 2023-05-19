import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os


def gen_algorithm_complexity_table(results,out_dir):
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


def gen_agent_performance_table(results,out_dir):
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


def gen_overall_tab(results,output_dir):
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
    #calculate baseline
    baseline_obj = {}
    for problem in problems:
        blobj_problem = results['cost'][problem]['DEAP_CMAES'] # 51 * record_length
        objs = []
        for run in range(51):
            objs.append(blobj_problem[run][-1])
        baseline_obj[problem] = sum(objs)/51
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
            df_results.loc[optimizer, (problem, 'Gap')] = "%.2f%%" % ((avg_obj - baseline_obj[problem]) / baseline_obj[problem] * 100)
            fes_problem_optimizer = np.array(results['fes'][problem][optimizer])
            df_results.loc[optimizer, (problem, 'FEs')] = np.format_float_scientific(fes_problem_optimizer.mean(), precision=3, exp_digits=1)

    df_results.to_excel(output_dir+'overall_table.xlsx')


def draw_test_cost(data, output_dir, Name=None, logged=None):
    for problem in list(data.keys()):
        if Name is None:
            name = problem
        elif (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
            continue
        else:
            name = Name
        plt.figure()
        if logged:
            plt.title('log cost curve ' + name)
        else:
            plt.title('cost curve ' + name)
        for agent in list(data[name].keys()):
            values = np.array(data[name][agent])
            x = np.arange(values.shape[-1])
            if logged:
                values = np.log(values)
            std = np.std(values, 0)
            mean = np.mean(values, 0)
            plt.plot(x, mean)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        if logged:
            plt.savefig(output_dir + f'{name}_log_cost_curve.png')
        else:
            plt.savefig(output_dir + f'{name}_cost_curve.png')


def draw_average_test_cost(data, output_dir):
    plt.figure()
    plt.title('all problem cost curve')
    X = np.arange(51)
    Y = {}
    for problem in list(data.keys()):
        for agent in list(data[problem].keys()):
            if agent not in Y.keys():
                Y[agent] = {'mean': [], 'std': []}
            values = np.array(data[problem][agent])
            values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
            std = np.std(values, 0)
            mean = np.mean(values, 0)
            Y[agent]['mean'].append(mean)
            Y[agent]['std'].append(std)

    for agent in list(Y.keys()):
        mean = np.mean(Y[agent]['mean'], 0)
        std = np.mean(Y[agent]['std'], 0)
        plt.plot(X, mean)
        plt.fill_between(X, mean - std, mean + std, alpha=0.2)
    plt.savefig(output_dir + f'all_problem_cost_curve.png')


def draw_average_performance_hist(data, output_dir, baseline='DEAP_CMAES'):
    plt.figure()
    plt.title('average performance histgram')
    D = {}
    X = []
    Y = []
    for problem in list(data.keys()):
        for agent in list(data[problem].keys()):
            if agent not in D.keys():
                D[agent] = []
            D[agent].append(np.array(data[problem][agent])[:, -1])

    for agent in D.keys():
        if agent == baseline:
            continue
        D[agent] = np.mean(np.array(D[agent]) / np.array(D[baseline]))
        X.append(agent)
        Y.append(D[agent])
    plt.bar(X, Y)
    plt.savefig(output_dir + f'average_performance_hist.png')


def post_processing_test_statics(log_dir):
    with open(log_dir+'test.pkl', 'rb') as f:
        results = pickle.load(f)
    if not os.path.exists(log_dir + 'tables/'):
        os.makedirs(log_dir + 'tables/')
    gen_overall_tab(results, log_dir+'tables/')
    gen_algorithm_complexity_table(results, log_dir+'tables/')
    gen_agent_performance_table(results, log_dir+'tables/')

    if not os.path.exists(log_dir + 'pics/'):
        os.makedirs(log_dir + 'pics/')
    draw_test_cost(results['cost'],log_dir + 'pics/')
    draw_average_test_cost(results['cost'],log_dir + 'pics/')
    draw_average_performance_hist(results['cost'],log_dir + 'pics/')
    draw_concrete_performance_hist(results['cost'],log_dir + 'pics/')


def draw_concrete_performance_hist(data, output_dir, Name=None, baseline='DEAP_CMAES'):
    D = {}
    X = []
    for problem in list(data.keys()):
        if Name is None:
            name = problem
        elif (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
            continue
        else:
            name = Name
        X.append(name)
        for agent in list(data[name].keys()):
            if agent not in D.keys():
                D[agent] = []
            D[agent].append(np.array(data[name][agent])[:, -1])

    for agent in D.keys():
        if agent == baseline:
            continue
        plt.figure()
        plt.title(f'{agent} performance histgram')
        X = []
        D[agent] = np.mean(np.array(D[agent]) / np.array(D[baseline]), -1)
        X.append(agent)
        plt.bar(X, D[agent])
        plt.savefig(output_dir + f'{agent}_concrete_performance_hist.png')

params = {
    'axes.labelsize': '25',
    'xtick.labelsize': '25',
    'ytick.labelsize': '25',
    'lines.linewidth': '4',
    'legend.fontsize': '24',
    'figure.figsize': '20,10',
}
plt.rcParams.update(params)


# markers = ['o', '^', '*', 'O', 'v', 'x', 'X', 'd', 'D', '.', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H']

class Logger:
    def __init__(self, config):
        self.config = config

    def draw_train_cost(self, train_set):
        log_dir = self.config.log_dir + f'/train/all_agent/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for problem in train_set:
            name = problem.__str__()
            plt.figure()
            plt.title('all agent ' + problem.__str__() + ' cost')
            for agent in self.config.agent_for_cp:
                load_dir = self.config.log_dir + f'/train/{agent}/{self.config.run_time}/log/'
                values = np.load(load_dir + name + '_cost.npy')
                x, y, n = values
                y /= n
                plt.plot(x, y)
            plt.savefig(log_dir + f'pic/all_agent_{name}_cost.png')
            plt.close()

    def draw_train_average_cost(self, train_set):
        log_dir = self.config.log_dir + f'/train/all_agent/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        plt.title('all agent all problem cost train')
        for agent in self.config.agent_for_cp:
            load_dir = self.config.log_dir + f'/train/{agent}/{self.config.run_time}/log/'
            X = []
            Y = []
            for problem in train_set:
                name = problem.__str__()
                values = np.load(load_dir + name + '_cost.npy')
                x, y, n = values
                y /= n
                X.append(x)
                Y.append(y)
            X = np.mean(X, 0)
            Y = np.mean(Y, 0)
            plt.plot(X, Y)
        plt.savefig(log_dir + f'pic/all_agent_all_problem_cost.png')
        plt.close()

    def draw_train_return(self):
        log_dir = self.config.log_dir + f'/train/all_agent/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        plt.title('all agent return')
        for agent in self.config.agent_for_cp:
            load_dir = self.config.log_dir + f'/train/{agent}/{self.config.run_time}/log/'
            values = np.load(load_dir + 'return.npy')
            plt.plot(values[0], values[1])
        plt.savefig(log_dir + f'pic/all_agent_return.png')
        plt.close()

    # {cost: {problem1: {optimizer1: [[51]*51], agent1: [[]*51]}}, fes: {optimizer1: [51], agent1: [51]}, Time: ......}

    def draw_test_cost(self, data=None, Name=None, logged=False):
        log_dir = self.config.log_dir + f'/test/{self.config.run_time}/pic/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if data is None:
            with open(self.config.log_dir + f'/test/{self.config.run_time}/test.pkl', 'rb') as f:
                data = pickle.load(f)['cost']
        for problem in list(data.keys()):
            if Name is None:
                name = problem
            elif (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = Name
            plt.figure()
            if logged:
                plt.title('log cost curve ' + name)
            else:
                plt.title('cost curve ' + name)
            print(len(data[name]['DEAP_CMAES'][0]))
            print(name)
            for agent in list(data[name].keys()):
                values = np.array(data[name][agent])
                x = np.arange(values.shape[-1])
                if logged:
                    values = np.log(values)
                print(values.shape)
                std = np.std(values, 0)
                mean = np.mean(values, 0)
                plt.plot(x, mean)
                plt.fill_between(x, mean - std, mean + std, alpha=0.2)
            if logged:
                plt.savefig(log_dir + f'{name}_log_cost_curve.png')
            else:
                plt.savefig(log_dir + f'{name}_cost_curve.png')

    def draw_average_test_cost(self, data=None):
        log_dir = self.config.log_dir + f'/test/{self.config.run_time}/pic/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if data is None:
            with open(self.config.log_dir + f'/test/{self.config.run_time}/test.pkl', 'rb') as f:
                data = pickle.load(f)['cost']
        plt.figure()
        plt.title('all problem cost curve')
        X = np.arange(51)
        Y = {}
        for problem in list(data.keys()):
            for agent in list(data[problem].keys()):
                if agent not in Y.keys():
                    Y[agent] = {'mean': [], 'std': []}
                values = np.array(data[problem][agent])
                values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
                std = np.std(values, 0)
                mean = np.mean(values, 0)
                Y[agent]['mean'].append(mean)
                Y[agent]['std'].append(std)

        for agent in list(data[problem].keys()):
            mean = np.mean(Y[agent]['mean'], 0)
            std = np.mean(Y[agent]['std'], 0)
            plt.plot(X, mean)
            plt.fill_between(X, mean - std, mean + std, alpha=0.2)
        plt.savefig(log_dir + f'all_problem_cost_curve.png')

    def draw_average_performance_hist(self, data=None, baseline='DEAP_CMAES'):
        log_dir = self.config.log_dir + f'/test/{self.config.run_time}/pic/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if data is None:
            with open(self.config.log_dir + f'/test/{self.config.run_time}/test.pkl', 'rb') as f:
                data = pickle.load(f)['cost']
        plt.figure()
        plt.title('average performance histgram')
        D = {}
        X = []
        Y = []
        for problem in list(data.keys()):
            for agent in list(data[problem].keys()):
                if agent not in D.keys():
                    D[agent] = []
                D[agent].append(np.array(data[problem][agent])[:, -1])

        for agent in D.keys():
            if agent == baseline:
                continue
            D[agent] = np.mean(np.array(D[agent]) / np.array(D[baseline]))
            X.append(agent)
            Y.append(D[agent])
        plt.bar(X, Y)
        plt.savefig(log_dir + f'average_performance_hist.png')

    def draw_concrete_performance_hist(self, data=None, Name=None, baseline='DEAP_CMAES'):
        log_dir = self.config.log_dir + f'/test/{self.config.run_time}/pic/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if data is None:
            with open(self.config.log_dir + f'/test/{self.config.run_time}/test.pkl', 'rb') as f:
                data = pickle.load(f)['cost']
        D = {}
        X = []
        for problem in list(data.keys()):
            if Name is None:
                name = problem
            elif (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = Name
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D.keys():
                    D[agent] = []
                D[agent].append(np.array(data[name][agent])[:, -1])

        for agent in D.keys():
            if agent == baseline:
                continue
            plt.figure()
            plt.title(f'{agent} performance histgram')
            X = []
            D[agent] = np.mean(np.array(D[agent]) / np.array(D[baseline]), -1)
            X.append(agent)
            plt.bar(X, D[agent])
            plt.savefig(log_dir + f'{agent}_concrete_performance_hist.png')
