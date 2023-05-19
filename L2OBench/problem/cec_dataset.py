from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset
from problem.cec_test_func import *
import pickle
import time

# make a dictionary contains all supported problems
problem_types = copy.deepcopy(functions)
problem_types['Composition'] = Composition
problem_types['Hybrid'] = Hybrid

# provide an uniform interface from problem data to a problem instance object
def get_instance(problem_data):
    name = problem_data[0]
    return problem_types[name].get_instance(problem_data)


'''
    The following two dictionaries are examples of using benchmark problems. The shift vectors, rotation matrix can be 
    read from files. However, other messages such as bias or sub problems can not be obtain in the same way. Thus we 
    introduce a multi-layer dictionary to obtain and index these messages. User can extend it once construct a benchmark
    dictionary in the form of cec2021 and cec2022
'''
cec2021 = {'Bent_cigar': [1, {'bias': 100}], 'Schwefel': [2, {'bias': 1100}], 'bi_Rastrigin': [3, {'bias': 700}], 'Grie_rosen': [4, {'bias': 1900}],
           'Hybrid1': [5, {'length': [0.3, 0.3, 0.4], 'bias': 1700, 'subproblems': ['Schwefel', 'Rastrigin', 'Ellipsoidal']}],
           'Hybrid2': [6, {'length': [0.2, 0.2, 0.3, 0.3], 'bias': 1600, 'subproblems': ['Escaffer6', 'Hgbat', 'Rosenbrock', 'Schwefel', ]}],
           'Hybrid3': [7, {'length': [0.1, 0.2, 0.2, 0.2, 0.3], 'bias': 2100, 'subproblems': ['Escaffer6', 'Hgbat', 'Rosenbrock', 'Schwefel', 'Ellipsoidal']}],
           'Composition1': [8, {'lamda': [1, 10, 1], 'sigma': [10, 20, 30], 'bias': [0, 100, 200], 'F': 2200, 'subproblems': ['Rastrigin', 'Griewank', 'Schwefel']}],
           'Composition2': [9, {'lamda': [10, 1e-6, 10, 1], 'sigma': [10, 20, 30, 40], 'bias': [0, 100, 200, 300], 'F': 2400, 'subproblems': ['Ackley', 'Ellipsoidal', 'Griewank', 'Rastrigin']}],
           'Composition3': [10, {'lamda': [10, 1, 10, 1e-6, 1], 'sigma': [10, 20, 30, 40, 50], 'bias': [0, 100, 200, 300, 400], 'F': 2500, 'subproblems': ['Rastrigin', 'Happycat', 'Ackley', 'Discus', 'Rosenbrock']}], }

cec2022 = {'Zakharov': [1, {'bias': 300}], 'Rosenbrock': [2, {'bias': 400}], 'Escaffer6': [3, {'bias': 600}], 'Step_Rastrigin': [4, {'bias': 800}], 'Levy': [5, {'bias': 900}],
           'Hybrid1': [6, {'length': [0.4, 0.4, 0.2], 'bias': 1800, 'subproblems': ['Bent_cigar', 'Hgbat', 'Rastrigin']}],
           'Hybrid2': [7, {'length':  [0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2], 'bias': 2000, 'subproblems': ['Hgbat', 'Katsuura', 'Ackley', 'Rastrigin', 'Schwefel', 'Scaffer_F7']}],
           'Hybrid3': [8, {'length': [0.3, 0.2, 0.2, 0.1, 0.2], 'bias': 2200, 'subproblems': ['Katsuura', 'Happycat', 'Grie_rosen', 'Schwefel', 'Ackley']}],
           'Composition1': [9, {'lamda': [1, 1e-6, 1e-6, 1e-6, 1e-6], 'sigma': [10, 20, 30, 40, 50], 'bias': [0, 200, 300, 100, 400], 'F': 2300, 'subproblems': ['Rosenbrock', 'Ellipsoidal', 'Bent_cigar', 'Discus', 'Ellipsoidal']}],
           'Composition2': [10, {'lamda': [1, 1, 1], 'sigma': [20, 10, 10], 'bias': [0, 200, 100], 'F': 2400, 'subproblems': ['Schwefel', 'Rastrigin', 'Hgbat']}],
           'Composition3': [11, {'lamda': [1e-6, 10, 1e-6, 10, 5e-6], 'sigma': [20, 20, 30, 30, 20], 'bias': [0, 200, 300, 400, 200], 'F': 2600, 'subproblems': ['Escaffer6', 'Schwefel', 'Griewank', 'Rosenbrock', 'Rastrigin']}],
           'Composition4': [12, {'lamda': [10, 10, 2.5, 1e-6, 1e-6, 5e-4], 'sigma': [10, 20, 30, 40, 50, 60], 'bias': [0, 300, 500, 100, 400, 200], 'F': 2700, 'subproblems': ['Hgbat', 'Rastrigin', 'Schwefel', 'Bent_cigar', 'Ellipsoidal', 'Escaffer6']}], }

cec2005 = {'Composition2005F18': Composition2005F18, 'Composition2005F23': Composition2005F23}


# a dataset class for training and testing, provides a Dataloader-like iterable object
class Training_Dataset(Dataset):
    def __init__(self,
                 dim,                           # the dimension of problems
                 num_samples,                   # the size of dataset
                 batch_size=1,                  # the size of batches
                 problem_list = None,           # the list of sub problems of Hybrid or Composition problems, ignored when there are only basic problems
                 problem_length=None,           # a list specifies the length assignment for sub problems
                 offset=0,                      # the offset of file reading if the dataset is generated from file reading
                 problems='all',                # the types of problems in the dataset, default('all') includes all problems obtained in problem_types dictionary, user can specify a list of candidate problems or a single problem type in string
                 shifted=True,                  # the three following parameters are the config of problems, whether they are shifted, rotated or biased
                 rotated=True,
                 biased=True,
                 training_seed=0,               # specify a random seed for reappearance or variable controlling
                 filename=None,                 # if want to read dataset data from a file, determine the file name
                 indicated_specific=False,      # if true, all sub problems in problem_names will join in order, or randomly selected when false, ignored when there are only basic problems
                 indicated_dataset=None,
                 save_generated_data=False      # if want to save the generated random problems, set True, else False
                 ):
        super(Training_Dataset, self).__init__()
        self.dim = dim
        self.batch_size = batch_size
        # initialize pointer for iteratively getting data batch
        self.ptr = [i for i in range(0, num_samples, batch_size)]
        # initialize the order data being selected, preparation for shuffling
        self.index = np.arange(num_samples)
        self.data = []
        if training_seed > 0:
            np.random.seed(training_seed)
        if filename is not None and filename != '':
            self.data = Training_Dataset.dataset_read(filename,num_samples, offset)
        else:
            if indicated_dataset is None:
                self.data = Training_Dataset.dataset_gen(num_samples,
                                                         dim,
                                                         problem_list=problem_list,
                                                         problem_length=problem_length,
                                                         problems=problems,
                                                         shifted=shifted,
                                                         rotated=rotated,
                                                         biased=biased,
                                                         indicated_specific=indicated_specific)
            else:
                self.data = Training_Dataset.rand_indicated_dataset(indicated_dataset,
                                                                    num_samples,
                                                                    dim,
                                                                    shifted=shifted,
                                                                    rotated=rotated,
                                                                    biased=biased,)
            if save_generated_data:
                run_name = time.strftime("%Y%m%dT%H%M%S")
                problem_name = '-'.join(name for name in problems)
                path = 'problem_dataset/' + problem_name + '-' + run_name + '.pickle'
                with open(path,'wb') as f:
                    pickle.dump(self.data,f)
        self.N = len(self.data)

    def __getitem__(self, item):
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def shuffle(self):
        self.index = np.random.permutation(self.N)

    # generate a set of problem data
    @staticmethod
    def dataset_gen(size,                       # the size of the set
                    dim,                        # the dimension of problems
                    problems='all',             # types of problems in the set
                    cf_num=0,                   # number of sub problems in Hybrid and Composition problems
                    problem_list=None,          # candidate sub problems for Hybrid and Composition problems
                    problem_length=None,        # a list specifies the length assignment for sub problems
                    shifted=True,               # following three parameters for config of problems
                    rotated=True,
                    biased=True,
                    indicated_specific=False    # determine sub problems are randomly selected or all selected
                    ):
        data = []
        for i in range(size):
            # select the type of current problem
            if problems == 'all':
                problem = np.random.choice(list(problem_types.keys()), 1)[0]
            elif isinstance(problems, list):
                problem = np.random.choice(problems, 1)[0]
            else:
                problem = problems
            # Composition and Hybrid problems has different generator interface from basic problems
            if problem == 'Composition':
                problem_data = problem_types[problem].generator(filename=None,
                                                 dim=dim,
                                                 cf_num=cf_num,
                                                 problem_names=problem_list,
                                                 shifted=shifted,
                                                 rotated=rotated,
                                                 biased=biased,
                                                 indicated_specific=indicated_specific)
            elif problem == 'Hybrid':
                problem_data = problem_types[problem].generator(filename=None,
                                                             dim=dim,
                                                             cf_num=cf_num,
                                                             problem_names=problem_list,
                                                             problem_length=problem_length,
                                                             shifted=shifted,
                                                             rotated=rotated,
                                                             biased=biased,
                                                             indicated_specific=indicated_specific)
            else:
                problem_data = problem_types[problem].generator(dim, shifted=shifted, rotated=rotated, biased=biased)
            data.append(get_instance(problem_data))
        return data

    @staticmethod
    def dataset_read(path, size, offset=0):  # Load a dataset from file
        with open(path,mode='rb') as f:
            data = pickle.load(f)
        return data[offset:offset+size]

    # an interface for benchmark testing, generate a problem from benchmark data
    @staticmethod
    def read_for_test(problem,              # the name of problem, it should be obtained in benchmark dictionary
                      dim,                  # dimension of problem, should follow the rule in technical report
                      directory,            # a file dictionary where benchmark problem data files locate
                      config,               # the configuration of problem, integer in [0, 7](000 to 111 in binary)
                      benchmark=cec2021     # specify the benchmark, in this file it could be cec2021 or cec2022, users can extend it once construct a benchmark dictionary in the form of cec2021 and cec2022
                      ):
        # resolve config code into problem configuration, follow the resolving rule in benchmark technical report
        biased, shifted, rotated = config & 4, config & 2, config & 1
        # get and resolve data from benchmark dictionary
        data = benchmark[problem]
        func_no = data[0]
        data = data[1]
        rotate_path = directory + 'M_{}_D{}.txt'.format(func_no, dim)
        shift_path = directory + 'shift_data_{}.txt'.format(func_no)
        # Hybrid problems
        if problem[:6] == 'Hybrid':
            rotate = np.eye(dim)
            if rotated:
                with open(rotate_path, 'r') as fpt:
                    for i in range(dim):
                        text = fpt.readline().split()
                        for j in range(dim):
                            rotate[i][j] = float(text[j])

            shift = np.zeros(dim)
            if shifted:
                with open(shift_path, 'r') as fpt:
                    text = fpt.readline().split()
                    for i in range(dim):
                        shift[i] = float(text[i])

            shuffle = np.zeros(dim, dtype=int)
            shuffle_path = directory + 'shuffle_data_{}_D{}.txt'.format(func_no, dim)
            with open(shuffle_path, 'r') as fpt:
                text = fpt.readline().split()
                for i in range(dim):
                    shuffle[i] = int(text[i]) - 1

            cf_num = len(data['length'])
            problem_ = []
            length = np.array(data['length']) * dim
            for i in range(cf_num):
                name = data['subproblems'][i]
                d = int(length[i])
                tmp = [name, d, np.zeros(d), np.eye(d), 0]
                problem_.append(Problem.get_instance(tmp))

            return eval('Hybrid')(np.array(dim), np.array(cf_num), np.array(shift), np.array(rotate),
                                  np.array(length, dtype=int), np.array(shuffle), np.array(data['bias']) if biased else 0, problem_)
        # Composition problems
        elif problem[:11] == 'Composition':
            cf_num = len(data['subproblems'])
            rotates = []
            with open(rotate_path, 'r') as fpt:
                for i in range(cf_num):
                    rotate = np.eye(dim)
                    if rotated:
                        for j in range(dim):
                            text = fpt.readline().split()
                            for k in range(dim):
                                rotate[j][k] = float(text[k])
                    rotates.append(rotate)

            shifts = []
            with open(shift_path, 'r') as fpt:
                for i in range(cf_num):
                    shift = np.zeros(dim)
                    if shifted:
                        text = fpt.readline().split()
                        for j in range(dim):
                            shift[j] = float(text[j])
                    shifts.append(shift)

            problem_ = []
            for i in range(cf_num):
                name = data['subproblems'][i]
                tmp = [name, dim, shifts[i], rotates[i], 0]
                problem_.append(Problem.get_instance(tmp))
            return eval('Composition')(np.array(dim), np.array(cf_num), np.array(data['lamda']), np.array(data['sigma']),
                                       np.array(data['bias']), np.array(data['F']) if biased else 0, problem_)
        # basic problems
        else:
            rotate = np.eye(dim)
            if rotated:
                with open(rotate_path, 'r') as fpt:
                    for i in range(dim):
                        text = fpt.readline().split()
                        for j in range(dim):
                            rotate[i][j] = float(text[j])

            shift = np.zeros(dim)
            if shifted:
                with open(shift_path, 'r') as fpt:
                    text = fpt.readline().split()
                    for i in range(dim):
                        shift[i] = float(text[i])
            return eval(problem)(np.array(dim), np.array(shift), np.array(rotate), np.array(data['bias']) if biased else 0)

    @staticmethod
    def rand_indicated_dataset(dataset,
                               size,          # the size of the set
                               dim,
                               shifted=True,  # following three parameters for config of problems
                               rotated=True,
                               biased=True,
                               ):
        data = []
        for i in range(size):
            # select the type of current problem
            problem = np.random.choice(list(dataset.keys()))
            if problem[:11] == 'Composition':
                problem_list = dataset[problem][1]['subproblems']
                problem_data = problem_types['Composition'].generator(filename=None,
                                                                     dim=dim,
                                                                     cf_num=len(problem_list),
                                                                     problem_names=problem_list,
                                                                     shifted=shifted,
                                                                     rotated=rotated,
                                                                     biased=biased,
                                                                     indicated_specific=True)
            elif problem[:6] == 'Hybrid':
                problem_list = dataset[problem][1]['subproblems']
                problem_data = problem_types['Hybrid'].generator(filename=None,
                                                                 dim=dim,
                                                                 cf_num=len(problem_list),
                                                                 problem_names=problem_list,
                                                                 problem_length=dataset[problem][1]['length'],
                                                                 shifted=shifted,
                                                                 rotated=rotated,
                                                                 biased=biased,
                                                                 indicated_specific=True)
            else:
                problem_data = problem_types[problem].generator(dim, shifted=shifted, rotated=rotated, biased=biased)
            data.append(get_instance(problem_data))
        return data


'''
2 Choice for Test:

ALL: 
    testing agent on all func_suites

Specific One
    testing a func in func_suites for your agent
    NOTED: param "indicated_problem" should be the KEY in func_suites
'''


# a method for benchmark testing
def Test(agent,                     # the optimizer object, all objects here should obtain a uniform interface test_run(see optimizer.JDE21)
         func_suites,               # the benchmark to test
         dim,                       # the dimension of problems
         MaxFEs,                    # the max number of evaluations
         data_path='test_data/',    # the directory of test data files
         result_path=None,          # the file path for result storing, will be generated with time stamp when it's None
         indicated_problem=None,    # specify a problem in benchmark to test
         indicated_config=None      # specify a configuration
         ):
    # initialize total record dictionary, it will also be multi-layer
    records = {}
    # get random seed pointer to assign seed to each testing
    seed_fpt = open(data_path + 'Rand_Seeds.txt', 'r')
    # generate result_path with time stamp if it is not specified, the path will contain information of optimizer, problem and configuration
    str_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    if result_path is None:
        result_path = 'test_result/' + agent.__class__.__name__ + '_'
        if indicated_problem is None:
            result_path += 'result_all'
        else:
            result_path += 'result_' + (indicated_problem
                                        if isinstance(indicated_problem, str) else '+'.join(p for p in indicated_problem))
        if indicated_config is not None:
            result_path += '_config' + (str(indicated_config)
                                         if isinstance(indicated_config, int) else ''.join(str(c) for c in indicated_config))
        result_path += '_' + str_time + '.txt'
    result_fpt = open(result_path, 'w')
    # iterate through every problem and pass those unspecified so that all random seeds for every testing are constant
    for problem in func_suites.keys():
        seeds = []
        # there are 1000 seeds in the Seeds.txt, every 30 seeds for each problem according to their Problem No.
        for i in range(30):
            seeds.append(int(float(seed_fpt.readline())))
        # pass unspecified problem
        if indicated_problem is not None:
            if (isinstance(indicated_problem, list) and problem not in indicated_problem) or \
                    (isinstance(indicated_problem, str) and problem != indicated_problem):
                continue
        record = {}
        # iterate through and pass unspecified configuration
        for config in range(8):
            if indicated_config is not None:
                if (isinstance(indicated_config, list) and config not in indicated_config) or \
                        (isinstance(indicated_config, int) and config != indicated_config):
                    continue
            # initialize records
            result_fpt.write(problem + ' with config {}:\n'.format(config))
            tmp = {}
            Fevs = []
            computation_time = []
            succ = []
            # initialize benchmark problem
            p = Training_Dataset.read_for_test(problem, dim, data_path, config, func_suites)
            # initialize progress bar
            pbar = tqdm(total=30, desc=problem + ' with config {}'.format(config),
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

            for run_id in range(30):
                # run test_run and get cost record, computation time and the number of evaluations in the end
                Fev, c_time, fes = agent.test_run(p, seeds[run_id], MaxFEs)
                # record results
                Fevs.append(Fev)
                computation_time.append(c_time)
                succ.append(fes)
                pbar.update(1)
            pbar.close()
            # record result in file
            result_fpt.write('\nFunction error values:\n')
            for i in range(30):
                result_fpt.write(' '.join(str(fev) for fev in Fevs[i]))
                result_fpt.write(' ' + str(succ[i]))
                result_fpt.write('\n')
            result_fpt.write('\ncomputation time:\n')
            result_fpt.write(' '.join(str(t) for t in computation_time))
            result_fpt.write('\n\n\n')
            result_fpt.flush()
            # record result in record dictionary
            tmp['Fevs'] = np.stack(Fevs)
            tmp['computation_time'] = np.stack(computation_time)
            tmp['success_fes'] = np.stack(succ)
            record[config] = tmp
        records[problem] = record
    return records
    # the returned records is a multi-layer dictionary:
    # {problem name: {configuration: {error value sequence, computation time, ending evaluation number } } }


# a method testing optimizer of a set of random problems, the main process is mostly the same as Test method above, but the problems are generated randomly
def random_Test(agent,                  # the optimizer object
                dim,                    # the dimension of problems
                MaxFEs,                 # the max number of evaluations
                data_path='test_data/', # the directory of test data files
                result_path=None,       # the file path for result storing
                indicated_problem=None, # specify a problem in benchmark to test
                indicated_config=None,  # specify a configuration
                seed=0                  # the random seed
                ):
    # initialize records, random seed and result file path
    records = {}
    np.random.seed(seed)
    seed_fpt = open(data_path + 'Rand_Seeds.txt', 'r')
    str_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    if result_path is None:
        result_path = 'test_result/' + agent.__class__.__name__ + '_'
        if indicated_problem is None:
            result_path += 'result_random'
        else:
            result_path += 'result_random_' + (indicated_problem
                                        if isinstance(indicated_problem, str) else '+'.join(p for p in indicated_problem))
        if indicated_config is not None:
            result_path += '_config' + (str(indicated_config)
                                         if isinstance(indicated_config, int) else ''.join(str(c) for c in indicated_config))
        result_path += '_' + str_time + '.txt'
    result_fpt = open(result_path, 'w')
    # initialize candidate problems and configurations
    if indicated_problem is not None:
        if isinstance(indicated_problem, list):
            problems = list(np.random.choice(indicated_problem, 30))
        else:
            problems = [indicated_problem for _ in range(30)]
    else:
        problems = list(np.random.choice(list(problem_types.keys()), 30))
    if indicated_config is not None:
        if isinstance(indicated_config, list):
            configs = list(np.random.choice(indicated_config, 30))
        else:
            configs = [indicated_config for _ in range(30)]
    else:
        configs = list(np.random.randint(0, 8, size=30))
    seeds = []
    # there are 1000 seeds in the Seeds.txt, every 30 seeds for each problem according to their Problem No.
    for i in range(30):
        seeds.append(int(float(seed_fpt.readline())))

    # run 30 random problem tests
    for i in range(30):
        problem = problems[i]
        config = configs[i]
        # different from Test method above, problem here is not read from benchmark data but randomly generated
        problems[i] = problem_types[problem].get_instance(problem_types[problem].generator(dim=dim, shifted=config & 2, rotated=config & 1, biased=config & 4))

    pbar = tqdm(total=30, desc='random test',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    record = {}
    Fevs = []
    computation_time = []
    succ = []
    result_fpt.write('random test\n')
    for i in range(30):
        # run test_run and get cost record, computation time and the number of evaluations in the end
        p = problems[i]
        Fev, c_time, fes = agent.test_run(p, seeds[i], MaxFEs)
        Fevs.append(Fev)
        computation_time.append(c_time)
        succ.append(fes)
        pbar.update(1)
    pbar.close()
    # result record
    result_fpt.write('\nFunction error values:\n')
    for i in range(30):
        result_fpt.write(' '.join(str(fev) for fev in Fevs[i]))
        result_fpt.write(' ' + str(succ[i]))
        result_fpt.write('\n')
    result_fpt.write('\ncomputation time:\n')
    result_fpt.write(' '.join(str(t) for t in computation_time))
    result_fpt.write('\n\n\n')
    result_fpt.flush()
    tmp = {}
    tmp['Fevs'] = np.stack(Fevs)
    tmp['computation_time'] = np.stack(computation_time)
    tmp['success_fes'] = np.stack(succ)
    record['rand'] = tmp
    records['rand'] = record

    return records





