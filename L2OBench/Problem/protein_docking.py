from os import path
import numpy as np
from torch.utils.data import Dataset

from L2OBench.Problem import Basic_Problem


class Protein_Docking(Basic_Problem):
    n_atoms = 100  # number of interface atoms considered for computational concern
    dim = 12

    def __init__(self, coor_init, q, e, r, basis, eigval):
        self.coor_init = coor_init  # [n_atoms, 3]
        self.q = q                  # [n_atoms, n_atoms]
        self.e = e                  # [n_atoms, n_atoms]
        self.r = r                  # [n_atoms, n_atoms]
        self.basis = basis          # [dim, 3*n_atoms]
        self.eigval = eigval        # [dim]

        self.optimum = 0.           # unknown, set to zero

    def func(self, x):
        eigval = 1.0 / np.sqrt(self.eigval)
        product = np.matmul(x * eigval, self.basis)  # [NP, 3*n_atoms]
        new_coor = product.reshape((-1, self.n_atoms, 3)) + self.coor_init  # [NP, n_atoms, 3]

        p2 = np.expand_dims(np.sum(new_coor * new_coor, axis=-1), axis=-1)  # sum of squares along last dim.  [NP, n_atoms, 1]
        p3 = np.matmul(new_coor, np.transpose(new_coor, (0, 2, 1)))  # inner products among row vectors. [NP, n_atoms, n_atoms]
        pair_dis = p2 - 2 * p3 + np.transpose(p2, (0, 2, 1))
        pair_dis[np.arange(pair_dis.shape[0]).repeat(self.n_atoms, 0),
                 np.arange(self.n_atoms)[None, :].repeat(pair_dis.shape[0], 0).reshape(-1),
                 np.arange(self.n_atoms)[None, :].repeat(pair_dis.shape[0], 0).reshape(-1)] = 0.  # set diagonal to zeros
        pair_dis = np.sqrt(pair_dis + 0.01)  # [NP, n_atoms, n_atoms]

        gt0_lt7 = (pair_dis > 0.1) & (pair_dis < 7.0)
        gt7_lt9 = (pair_dis > 7.0) & (pair_dis < 9.0)

        pair_dis += np.eye(self.n_atoms)  # [NP, n_atoms, n_atoms]
        coeff = self.q / (4. * pair_dis) + np.sqrt(self.e) * ((self.r / pair_dis) ** 12 - (self.r / pair_dis) ** 6)  # [NP, n_atoms, n_atoms]

        energy = np.mean(
            np.sum(10 * gt0_lt7 * coeff + 10 * gt7_lt9 * coeff * ((9 - pair_dis) ** 2 * (-12 + 2 * pair_dis) / 8),
                   axis=1), axis=-1) - 7000  # [NP]

        return energy


class Protein_Docking_Dataset(Dataset):
    train_set = {'rigid': ['1AVX', '1BJ1', '1BVN', '1CGI', '1DFJ', '1EAW', '1EWY', '1EZU', '1IQD', '1JPS',
                           '1KXQ', '1MAH', '1N8O', '1PPE', '1R0R', '2B42', '2I25', '2JEL', '7CEI'],
                 'medium': ['1GRN', '1IJK', '1M10', '1XQS'],
                 'difficult': ['1ATN', '1IBR']
                 }

    test_set = {'rigid': ['1AY7'],
                'medium': ['2HRK'],
                'difficult': ['2C0L']
                }

    n_start_points = 10  # top models from ZDOCK

    def __init__(self,
                 num_samples,
                 batch_size=1,
                 difficulty=None,
                 sample_seed=0,
                 test=False):
        self.batch_size = batch_size
        # initialize pointer for iteratively getting data batch
        self.ptr = [i for i in range(0, num_samples, batch_size)]
        # initialize the order data being selected, preparation for shuffling
        self.index = np.arange(num_samples)
        self.data = []
        if sample_seed > 0:
            np.random.seed(sample_seed)
        self.data = Protein_Docking_Dataset.load_data(num_samples, difficulty, test)
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

    @classmethod
    def load_data(cls, num_samples, difficulty=None, test=False):
        set = cls.test_set if test else cls.train_set
        candidates = []
        if difficulty is None:
            for value in set.values():
                candidates.extend(value)
        else:
            try:
                candidates.extend(set[difficulty])
            except KeyError:
                raise ValueError(f"Difficulty should be None, 'rigid', 'medium' or 'difficult', but {difficulty} received.")

        complexes = np.random.choice(candidates, size=num_samples)
        start_points = np.random.choice(cls.n_start_points, size=num_samples) + 1

        data = []
        data_folder = path.join(path.dirname(__file__), 'protein_docking_data')

        for i in range(num_samples):
            data_dir = path.join(data_folder, complexes[i]+'_'+str(start_points[i]))
            coor_init = np.loadtxt(data_dir+'/coor_init')
            q = np.loadtxt(data_dir+'/q')
            e = np.loadtxt(data_dir+'/e')
            r = np.loadtxt(data_dir+'/r')
            basis = np.loadtxt(data_dir+'/basis')
            eigval = np.loadtxt(data_dir+'/eigval')

            q = np.tile(q, (1, 1))
            e = np.tile(e, (1, 1))
            r = np.tile(r, (len(r), 1))

            q = np.matmul(q.T, q)
            e = np.sqrt(np.matmul(e.T, e))
            r = (r + r.T) / 2

            data.append(Protein_Docking(coor_init, q, e, r, basis, eigval))
        return data
