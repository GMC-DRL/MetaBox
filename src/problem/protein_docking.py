from os import path
import torch
import numpy as np
from torch.utils.data import Dataset
from problem.basic_problem import Basic_Problem


class Protein_Docking(Basic_Problem):
    n_atoms = 100  # number of interface atoms considered for computational concern
    dim = 12
    lb = -1.5
    ub = 1.5

    def __init__(self, coor_init, q, e, r, basis, eigval, problem_id):
        self.coor_init = coor_init  # [n_atoms, 3]
        self.q = q                  # [n_atoms, n_atoms]
        self.e = e                  # [n_atoms, n_atoms]
        self.r = r                  # [n_atoms, n_atoms]
        self.basis = basis          # [dim, 3*n_atoms]
        self.eigval = eigval        # [dim]
        self.problem_id = problem_id
        self.optimum = None      # unknown, set to None

    def __str__(self):
        return self.problem_id

    def func(self, x):
        eigval = 1.0 / np.sqrt(self.eigval)
        product = np.matmul(x * eigval, self.basis)  # [NP, 3*n_atoms]
        new_coor = product.reshape((-1, self.n_atoms, 3)) + self.coor_init  # [NP, n_atoms, 3]

        p2 = np.expand_dims(np.sum(new_coor * new_coor, axis=-1), axis=-1)  # sum of squares along last dim.  [NP, n_atoms, 1]
        p3 = np.matmul(new_coor, np.transpose(new_coor, (0, 2, 1)))  # inner products among row vectors. [NP, n_atoms, n_atoms]
        pair_dis = p2 - 2 * p3 + np.transpose(p2, (0, 2, 1))
        pair_dis = np.sqrt(pair_dis + 0.01)  # [NP, n_atoms, n_atoms]

        gt0_lt7 = (pair_dis > 0.11) & (pair_dis < 7.0)
        gt7_lt9 = (pair_dis > 7.0) & (pair_dis < 9.0)

        pair_dis += np.eye(self.n_atoms)  # [NP, n_atoms, n_atoms]
        coeff = self.q / (4. * pair_dis) + np.sqrt(self.e) * ((self.r / pair_dis) ** 12 - (self.r / pair_dis) ** 6)  # [NP, n_atoms, n_atoms]

        energy = np.mean(
            np.sum(10 * gt0_lt7 * coeff + 10 * gt7_lt9 * coeff * ((9 - pair_dis) ** 2 * (-12 + 2 * pair_dis) / 8),
                   axis=1), axis=-1)  # [NP]

        return energy


class Protein_Docking_torch(Basic_Problem):
    n_atoms = 100  # number of interface atoms considered for computational concern
    dim = 12
    lb = -1.5
    ub = 1.5

    def __init__(self, coor_init, q, e, r, basis, eigval, problem_id):
        self.coor_init = torch.as_tensor(coor_init, dtype=torch.float64)  # [n_atoms, 3]
        self.q = torch.as_tensor(q, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.e = torch.as_tensor(e, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.r = torch.as_tensor(r, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.basis = torch.as_tensor(basis, dtype=torch.float64)    # [dim, 3*n_atoms]
        self.eigval = torch.as_tensor(eigval, dtype=torch.float64)  # [dim]
        self.problem_id = problem_id
        self.optimum = None  # unknown, set to None

    def __str__(self):
        return self.problem_id

    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dtype != torch.float64:
            x = x.type(torch.float64)
        if x.ndim == 1:  # x is a single individual
            return self.func(x.reshape(1, -1))[0]
        elif x.ndim == 2:  # x is a whole population
            return self.func(x)
        else:
            return self.func(x.reshape(-1, x.shape[-1]))

    def func(self, x):
        eigval = 1.0 / torch.sqrt(self.eigval)
        product = torch.matmul(x * eigval, self.basis)  # [NP, 3*n_atoms]
        new_coor = product.reshape((-1, self.n_atoms, 3)) + self.coor_init  # [NP, n_atoms, 3]

        p2 = torch.sum(new_coor * new_coor, dim=-1, dtype=torch.float64)[:, :,
             None]  # sum of squares along last dim.  [NP, n_atoms, 1]
        p3 = torch.matmul(new_coor,
                          new_coor.permute(0, 2, 1))  # inner products among row vectors. [NP, n_atoms, n_atoms]
        pair_dis = p2 - 2 * p3 + p2.permute(0, 2, 1)
        pair_dis = torch.sqrt(pair_dis + 0.01)  # [NP, n_atoms, n_atoms]

        gt0_lt7 = (pair_dis > 0.11) & (pair_dis < 7.0)
        gt7_lt9 = (pair_dis > 7.0) & (pair_dis < 9.0)

        pair_dis = pair_dis + torch.eye(self.n_atoms, dtype=torch.float64)  # [NP, n_atoms, n_atoms]
        coeff = self.q / (4. * pair_dis) + torch.sqrt(self.e) * (
                    (self.r / pair_dis) ** 12 - (self.r / pair_dis) ** 6)  # [NP, n_atoms, n_atoms]

        energy = torch.mean(
            torch.sum(10 * gt0_lt7 * coeff + 10 * gt7_lt9 * coeff * ((9 - pair_dis) ** 2 * (-12 + 2 * pair_dis) / 8),
                      dim=1, dtype=torch.float64), dim=-1)  # [NP]

        return energy


class Protein_Docking_Dataset(Dataset):
    proteins_set = {'rigid': ['1AVX', '1BJ1', '1BVN', '1CGI', '1DFJ', '1EAW', '1EWY', '1EZU', '1IQD', '1JPS',
                              '1KXQ', '1MAH', '1N8O', '1PPE', '1R0R', '2B42', '2I25', '2JEL', '7CEI', '1AY7'],
                    'medium': ['1GRN', '1IJK', '1M10', '1XQS', '2HRK'],
                    'difficult': ['1ATN', '1IBR', '2C0L']
                    }
    n_start_points = 10  # top models from ZDOCK

    def __init__(self,
                 data,
                 batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(version,
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy',
                     dataset_seed=1035):
        # apart train set and test set
        if difficulty == 'easy':
            train_set_ratio = 0.75
        elif difficulty == 'difficult':
            train_set_ratio = 0.25
        else:
            raise ValueError
        if dataset_seed > 0:
            np.random.seed(dataset_seed)
        train_proteins_set = []
        test_proteins_set = []
        for key in Protein_Docking_Dataset.proteins_set.keys():
            permutated = np.random.permutation(Protein_Docking_Dataset.proteins_set[key])
            n_train_proteins = max(1, min(int(len(permutated) * train_set_ratio), len(permutated) - 1))
            train_proteins_set.extend(permutated[:n_train_proteins])
            test_proteins_set.extend(permutated[n_train_proteins:])
        # construct problem instances
        data = []
        data_folder = path.join(path.dirname(__file__), 'protein_docking_data')
        for i in train_proteins_set + test_proteins_set:
            for j in range(Protein_Docking_Dataset.n_start_points):
                problem_id = i + '_' + str(j + 1)
                data_dir = path.join(data_folder, problem_id)
                coor_init = np.loadtxt(data_dir + '/coor_init')
                q = np.loadtxt(data_dir + '/q')
                e = np.loadtxt(data_dir + '/e')
                r = np.loadtxt(data_dir + '/r')
                basis = np.loadtxt(data_dir + '/basis')
                eigval = np.loadtxt(data_dir + '/eigval')

                q = np.tile(q, (1, 1))
                e = np.tile(e, (1, 1))
                r = np.tile(r, (len(r), 1))

                q = np.matmul(q.T, q)
                e = np.sqrt(np.matmul(e.T, e))
                r = (r + r.T) / 2
                if version == 'protein':
                    data.append(Protein_Docking(coor_init, q, e, r, basis, eigval, problem_id))
                elif version == 'protein-torch':
                    data.append(Protein_Docking_torch(coor_init, q, e, r, basis, eigval, problem_id))
                else:
                    raise ValueError(f'{version} version is invalid or is not supported yet.')
        n_train_instances = len(train_proteins_set) * Protein_Docking_Dataset.n_start_points
        return Protein_Docking_Dataset(data[:n_train_instances], train_batch_size), Protein_Docking_Dataset(data[n_train_instances:], test_batch_size)

    def __getitem__(self, item):
        if self.batch_size < 2:
            return self.data[self.index[item]]
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Protein_Docking_Dataset'):
        return Protein_Docking_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
