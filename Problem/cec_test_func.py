import copy
import math
import time

import numpy as np
import random
import torch

eps = 1e-14
INF = 1e50


def rotatefunc(x, Mr):
    return np.matmul(Mr, x.transpose()).transpose()


def sr_func(x, Os, Mr, sh):  # shift and rotate
    y = (x[:, :Os.shape[-1]] - Os) * sh
    # x, Os belongs to [-100, 100]
    return rotatefunc(y, Mr)


def rotate_gen(dim):  # Generate a rotate matrix
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    mat = np.eye(dim)
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
    H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


# Superclass for all basis problems, for the clarity and simplicity of code, and in the convenience of calling
class Problem:
    def __init__(self, dim, shift, rotate, bias):
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.FES = -1
        self.ub = np.ones(dim) * 100
        self.lb = -np.ones(dim) * 100
        self.opt = self.shift
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:  # x is a single individual
            return self.func(x.reshape(1, -1))[0]
        elif x.ndim == 2:  # x is a whole population
            return self.func(x)
        else:
            raise ArithmeticError('The input should be an array of 1 or 2 dimensions.')

    def func(self, x):
        return 0

    def get_optimal(self):
        return self.opt

    @staticmethod
    def read(problem_path, problem_type, size=1):  # Read the problem data from file
        instances = []
        with open(problem_path, 'r') as fpt:
            if fpt is None:
                print("\n Error: Cannot open input file for reading \n")
                return
            for k in range(size):
                d = fpt.readline().split()
                if len(d) < 1:
                    print("\n Error: Not enough instances for reading \n")
                    return
                name = d[0]
                dim = int(d[1])
                bias = float(d[2])
                shift = np.zeros(dim)
                rotate = np.eye(dim)
                text = fpt.readline().split()
                for j in range(dim):
                    shift[j] = float(text[j])
                for i in range(dim):
                    text = fpt.readline().split()
                    for j in range(dim):
                        rotate[i][j] = float(text[j])
                instances.append([problem_type, dim, shift, rotate, bias])
            return instances

    @classmethod
    def generator(cls, dim, size=1, shifted=True, rotated=True, biased=True):  # Generate a specified number(size) of instance data of type-assigned problem
        instances = []
        # get current problem type(class name) so that the corresponding methods can be called
        problem_type = cls.__name__
        for i in range(size):
            if shifted:
                shift = np.random.random(dim) * 160 - 100
            else:
                shift = np.zeros(dim)
            if rotated:
                H = rotate_gen(dim)
            else:
                H = np.eye(dim)
            if biased:
                bias = np.random.randint(1, 26) * 100
            else:
                bias = 0
            # instances.append(eval(problem_type)(dim, shift, H))
            instances.append([problem_type, dim, shift, H, bias])
        return instances if size > 1 else instances[0]

    @staticmethod
    def store_instance(instances, filename):  # Store the problem instance data into a file
        size = len(instances)
        mode = 'w'
        for k in range(size):
            with open(filename, mode) as fpt:
                fpt.write(str(instances[k][0]) + " " + str(instances[k][1]) + " " + str(instances[k][4]) + '\n')
                fpt.write(' '.join(str(i) for i in instances[k][2]))
                fpt.write('\n')
                for i in range(instances[k][1]):
                    fpt.write(' '.join(str(j) for j in instances[k][3][i]))
                    fpt.write('\n')
            mode = 'a'

    @classmethod
    def get_instance(cls, problem_data):  # Transfer a batch of instance data to a batch of instance objects
        type, dim, shift, H, bias = problem_data
        return eval(type)(np.array(dim), np.array(shift), np.array(H), np.array(bias))


class Sphere(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.sum(z ** 2, -1) + self.bias


class Schwefel12(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        res = 0
        for i in range(self.dim):
            tmp = np.power(np.sum(z[:i]), 2)
            res += tmp
        return res + self.bias

class Ellipsoidal(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        i = np.arange(nx)
        return np.sum(np.power(10, 6 * i / (nx - 1)) * (z ** 2), -1) + self.bias


class Bent_cigar(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return z[:, 0] ** 2 + np.sum(np.power(10, 6) * (z[:, 1:] ** 2), -1) + self.bias


class Discus(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.power(10, 6) * (z[:, 0] ** 2) + np.sum(z[:, 1:] ** 2, -1) + self.bias


class Dif_powers(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        i = np.arange(self.dim)
        return np.power(np.sum(np.power(np.fabs(z), 2 + 4 * i / max(1, self.dim - 1)), -1), 0.5) + self.bias


class Rosenbrock(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 2.048 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z += 1
        z_ = z[:, 1:]
        z = z[:, :-1]
        tmp1 = z ** 2 - z_
        return np.sum(100 * tmp1 * tmp1 + (z - 1) ** 2, -1) + self.bias


class Ackley(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        sum1 = -0.2 * np.sqrt(np.sum(z ** 2, -1) / self.dim)
        sum2 = np.sum(np.cos(2 * np.pi * z), -1) / self.dim
        return np.round(np.e + 20 - 20 * np.exp(sum1) - np.exp(sum2), 15) + self.bias


class Weierstrass(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 0.5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        a, b, k_max = 0.5, 3.0, 20
        sum1, sum2 = 0, 0
        for k in range(k_max + 1):
            sum1 += np.sum(np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (z + 0.5)), -1)
            sum2 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
        return sum1 - self.dim * sum2 + self.bias


class Griewank(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 6
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        s = np.sum(z ** 2, -1)
        p = np.ones(x.shape[0])
        for i in range(self.dim):
            p *= np.cos(z[:, i] / np.sqrt(1 + i))
        return 1 + s / 4000 - p + self.bias


class Rastrigin(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5.12 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, -1) + self.bias


class Schwefel(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 10
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        a = 4.209687462275036e+002
        b = 4.189828872724338e+002
        z += a
        g1 = z * np.sin(np.sqrt(np.abs(z)))
        g2 = (500 - z % 500) * np.sin(np.sqrt(np.fabs(500 - z % 500))) - (z - 500) ** 2 / (10000 * self.dim)
        g3 = (-z % 500 - 500) * np.sin(np.sqrt(np.fabs(500 - -z % 500))) - (z + 500) ** 2 / (10000 * self.dim)
        g = np.where(np.fabs(z) <= 500, g1, 0) + np.where(z > 500, g2, 0) + np.where(z < -500, g3, 0)
        return b * self.dim - g.sum(-1)


class Katsuura(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        tmp3 = np.power(self.dim, 1.2)
        tmp1 = np.repeat(np.power(np.ones((1, 32)) * 2, np.arange(1, 33)), x.shape[0], 0)
        res = np.ones(x.shape[0])
        for i in range(self.dim):
            tmp2 = tmp1 * np.repeat(z[:, i, None], 32, 1)
            temp = np.sum(np.fabs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1, -1)
            res *= np.power(1 + (i + 1) * temp, 10 / tmp3)
        tmp = 10 / self.dim / self.dim
        return res * tmp - tmp + self.bias


class Grie_rosen(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z += 1
        z_ = np.concatenate((z[:, 1:], z[:, :1]), -1)
        _z = z
        tmp1 = _z ** 2 - z_
        temp = 100 * tmp1 * tmp1 + (_z - 1) ** 2
        res = np.sum(temp * temp / 4000 - np.cos(temp) + 1, -1)
        return res + self.bias


class Escaffer6(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z_ = np.concatenate((z[:, 1:], z[:, :1]), -1)
        return np.sum(0.5 + (np.sin(np.sqrt(z ** 2 + z_ ** 2)) ** 2 - 0.5) / ((1 + 0.001 * (z ** 2 + z_ ** 2)) ** 2), -1) + self.bias


class Happycat(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def get_optimal(self):
        return self.opt

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z -= 1
        sum_z = np.sum(z, -1)
        r2 = np.sum(z ** 2, -1)
        return np.power(np.fabs(r2 - self.dim), 1 / 4) + (0.5 * r2 + sum_z) / self.dim + 0.5 + self.bias


class Hgbat(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z -= 1
        sum_z = np.sum(z, -1)
        r2 = np.sum(z ** 2, -1)
        return np.sqrt(np.fabs(np.power(r2, 2) - np.power(sum_z, 2))) + (0.5 * r2 + sum_z) / self.dim + 0.5 + self.bias


class bi_Rastrigin(Problem):
    def __init__(self, dim, shift, rotate, bias):
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        d = 1
        s = 1 - 1 / (2 * np.sqrt(self.dim + 20) - 8.2)
        u0 = 2.5
        u1 = -np.sqrt((u0 * u0 - d) / s)
        y = 10 * (x - self.shift) / 100
        tmpx = 2 * y
        tmpx[:, self.shift < 0] *= -1
        z = rotatefunc(tmpx, self.rotate)
        tmpx += u0
        tmp1 = 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), -1))
        tmp2 = np.minimum(np.sum((tmpx - u0) ** 2, -1), d * self.dim + s * np.sum((tmpx - u1) ** 2, -1))
        return tmp1 + tmp2 + self.bias


class Zakharov(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1.0
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return np.sum(np.power(z, 2), -1) + np.power(np.sum(0.5 * z, -1), 2) + np.power(np.sum(0.5 * z, -1), 4)


class Levy(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1.0
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        w = 1 + (z - 1) / 4
        _w_ = w[:, :-1]
        return np.power(np.sin(np.pi * w[:, 0]), 2) + \
               np.sum(np.power(_w_ - 1, 2) * (1 + 10 * np.power(np.sin(np.pi * _w_ - 1), 2))) + \
               np.power(w[:, -1] - 1, 2) * (1 + np.power(np.sin(2 * np.pi * w[:, -1]), 2))


class Scaffer_F7(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 1
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        s = np.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)
        return np.power(1 / (self.dim - 1) * np.sum(np.sqrt(s) * (np.sin(50 * np.power(s, 0.2)) + 1), -1), 2)


class Step_Rastrigin(Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5.12 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        y = np.zeros((x.shape[0], self.dim))
        for i in range(x.shape[0]):
            for j in range(self.dim):
                if np.fabs(x[i][j] - self.shift[j]) > 0.5:
                    y[i][j] = self.shift[j] + np.floor(2 * (x[i][j] - self.shift[j]) + 0.5) / 2
        return Rastrigin(self.dim, self.shift, self.rotate, self.bias).func(y)


# Dictionary of all supported basic problems, provide a map from problem names to classes
functions = {'Sphere': Sphere, 'Ellipsoidal': Ellipsoidal, 'Bent_cigar': Bent_cigar, 'Discus': Discus,
             'Dif_powers': Dif_powers, 'Rosenbrock': Rosenbrock, 'Ackley': Ackley, 'Weierstrass': Weierstrass,
             'Griewank': Griewank,'Rastrigin': Rastrigin, 'Schwefel': Schwefel, 'Katsuura': Katsuura,
             'Grie_rosen': Grie_rosen, 'Escaffer6': Escaffer6, 'Happycat': Happycat, 'Hgbat': Hgbat,
             'bi_Rastrigin': bi_Rastrigin, 'Zakharov': Zakharov, 'Levy': Levy, 'Scaffer_F7': Scaffer_F7,
             'Step_Rastrigin': Step_Rastrigin}


class Hybrid:
    def __init__(self, dim, cf_num, shift, rotate, length, shuffle, bias, sub_problems):
        self.dim = dim
        self.cf_num = cf_num
        self.shift = shift
        self.rotate = rotate
        self.ub = np.ones(dim) * 100
        self.lb = -np.ones(dim) * 100
        self.length = length
        self.shuffle = shuffle
        self.sub_problems = sub_problems
        self.bias = bias
        self.opt = self.shift
        self.FES = 0
        # calculate the optimal value
        self.optimum = self.func(self.opt.reshape(1, -1))[0]
        self.FES = 0

    def get_optimal(self):
        return self.opt

    @staticmethod
    def read(problems_path, size=1):  # Read a specified number of problem data from file
        with open(problems_path, 'r') as fpt:
            if fpt is None:
                print("\n Error: Cannot open input file for reading \n")
                return
            instances = []
            for i in range(size):
                tmp = fpt.readline().split()
                if len(tmp) < 1:
                    print("\n Error: Not enough instances for reading \n")
                    return
                dim = int(tmp[0])
                cf_num = int(tmp[1])
                bias = float(tmp[2])

                shift = np.zeros(dim)
                text = fpt.readline().split()
                for j in range(dim):
                    shift[j] = float(text[j])

                rotate = np.eye(dim)
                for i in range(dim):
                    text = fpt.readline().split()
                    for j in range(dim):
                        rotate[i][j] = float(text[j])

                length_str = fpt.readline().split()
                length = np.zeros(cf_num, dtype=int)
                for i in range(cf_num):
                    length[i] = int(length_str[i])

                shuffle_str = fpt.readline().split()
                shuffle = np.zeros(dim, dtype=int)
                for i in range(dim):
                    shuffle[i] = int(shuffle_str[i])

                problems = []
                text = fpt.readline().split()
                for i in range(cf_num):
                    name = text[i]
                    problems.append(name)
                instances.append(['Hybrid', dim, cf_num, shift, rotate, length, shuffle, bias, problems])
        return instances

    def func(self, x):  # evaluate a group of solutions
        self.FES += x.shape[0]
        y = sr_func(x, self.shift, self.rotate, 1,)
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            z[i] = y[i][self.shuffle]
        index = 0
        res = 0
        for i in range(self.cf_num):
            zi = z[:, index:index + self.length[i]]
            index += self.length[i]
            res += self.sub_problems[i].func(zi)
        return res + self.bias

    # Generate a specified number of hybrid problems and store in a file (or not)
    @staticmethod
    def generator(dim=0,                    # problem dimensions, randomly selected if the value <= 0
                  cf_num=0,                 # the number of sub problems, also randomly selected if the value <= 0
                  problem_names=None,       # a list provides candidate sub problem names(string), these problems can be randomly selected and join or add all in order
                  problem_length=None,      # a list specifies the length assignment for sub problems
                  size=1,                   # the size of generated dataset
                  shifted=True,             # the three following parameters are the config of problems, whether they are shifted, rotated or biased
                  rotated=True,
                  biased=True,
                  store=False,              # whether to store these problem data into a file
                  filename=None,            # if store, determine the file name
                  indicated_specific=False  # if true, all sub problems in problem_names will join in order, or randomly selected when false
                  ):
        # if dim parameter <= 0, randomly select one
        if dim <= 0:
            dim = np.random.randint(3, 6) * 10
        # if cf_num parameter <= 0, randomly select one
        if cf_num <= 0:
            cf_num = np.random.randint(3, 6)
        # if all sub problems in problem_names join in order, the number of sub problems should be the same as length of problem_names
        if indicated_specific and problem_names is not None:
            cf_num = len(problem_names)
        instances = []
        mode = 'w'
        for i in range(size):
            # generate data for each problem
            if shifted:
                shift = np.random.random(dim) * 160 - 100
            else:
                shift = np.zeros(dim)
            if rotated:
                H = rotate_gen(dim)
            else:
                H = np.eye(dim)
            if biased:
                bias = np.random.randint(1, 26) * 100
            else:
                bias = 0
            seg = np.random.uniform(0.1, 1, cf_num)
            seg /= np.sum(seg)
            if problem_length is not None:
                seg = np.array(problem_length)
            length = np.array(np.round(dim * seg), dtype=int)
            # length[length < 3] = 3
            length[-1] = dim - np.sum(length[:-1])
            shuffle = np.random.permutation(dim)
            # generate data for sub problems
            problems = []
            for i in range(cf_num):
                if problem_names is None:  # User doesn't assign the problems in the composition
                    name = random.sample(list(functions.keys()), 1)[0]
                elif indicated_specific:
                    name = problem_names[i]
                else:
                    name = random.sample(problem_names, 1)[0]
                # the least length of solutions for bi_Rastrigin problem has limitation
                while not indicated_specific and name == 'bi_Rastrigin' and length[i] < 5:
                    if problem_names is None or len(problem_names) == 0:
                        name = random.sample(list(functions.keys()), 1)[0]
                    else:
                        name = random.sample(problem_names, 1)[0]
                problems.append(name)
            # if user chooses to store problem set into a file
            if store:
                with open(filename, mode) as fpt:
                    fpt.write(str(dim) + ' ' + str(cf_num) + ' ' + str(bias) + '\n')
                    fpt.write(' '.join(str(i) for i in shift))
                    fpt.write('\n')
                    for i in range(dim):
                        fpt.write(' '.join(str(j) for j in H[i]))
                        fpt.write('\n')
                    fpt.write(' '.join(str(int(i)) for i in length))
                    fpt.write('\n')
                    fpt.write(' '.join(str(int(i)) for i in shuffle))
                    fpt.write('\n')
                    for i in range(cf_num):
                        fpt.write(' '.join(name for name in problems))
            instances.append(['Hybrid', dim, cf_num, shift, H, length, shuffle, bias, problems])
            mode = 'a'
        return instances if size > 1 else instances[0]  # for convenience, return a single object if the size is 1

    @staticmethod
    # Transfer an instance data to an instance object
    def get_instance(problem_data):
        name, dim, cf_num, shift, rotate, length, shuffle, bias, problems = problem_data
        problem_ = []
        for j in range(cf_num):
            name = problems[j]
            d = length[j]
            tmp = [name, d, np.zeros(d), np.eye(d), 0]
            problem_.append(Problem.get_instance(tmp))
        return eval('Hybrid')(np.array(dim), np.array(cf_num), np.array(shift), np.array(rotate), np.array(length), np.array(shuffle), np.array(bias), problem_)


class Composition:
    def __init__(self, dim, cf_num, lamda, sigma, bias, F, sub_problems):
        self.dim = dim
        self.cf_num = cf_num
        self.lamda = lamda
        self.sigma = sigma
        self.bias = bias
        self.F = F
        self.sub_problems = sub_problems
        self.FES = 0
        # calculate optimal value and solution
        self.opt = self.sub_problems[0].opt
        self.optimum = self.func(self.sub_problems[0].opt.reshape(1, -1))[0]
        for i in range(cf_num - 1):
            value = self.func(self.sub_problems[i + 1].opt.reshape(1, -1))[0]
            if value < self.optimum:
                self.opt = self.sub_problems[i + 1].opt
                self.optimum = value
        self.FES = 0

    def get_optimal(self):
        return self.opt

    @staticmethod
    def read(problems_path, size=1):  # Read a specified number of problem data from file
        with open(problems_path, 'r') as fpt:
            if fpt is None:
                print("\n Error: Cannot open input file for reading \n")
                return
            instances = []
            for i in range(size):
                cf_str = fpt.readline().split()
                if len(cf_str) < 1:
                    print("\n Error: Not enough instances for reading \n")
                    return
                cf_num = int(cf_str[0])

                lamda_str = fpt.readline().split()
                lamda = np.zeros(cf_num)
                for i in range(cf_num):
                    lamda[i] = float(lamda_str[i])

                sigma_str = fpt.readline().split()
                sigma = np.zeros(cf_num)
                for i in range(cf_num):
                    sigma[i] = float(sigma_str[i])

                bias_str = fpt.readline().split()
                bias = np.zeros(cf_num)
                for i in range(cf_num):
                    bias[i] = float(bias_str[i])

                F = float(fpt.readline().split()[0])

                problems = []
                dim = 0
                for i in range(cf_num):
                    text = fpt.readline().split()
                    name = text[0]
                    d = int(text[1])
                    dim = max(dim, d)
                    shift = np.zeros(d)
                    rotate = np.eye(d)
                    text = fpt.readline().split()
                    for j in range(d):
                        shift[j] = float(text[j])
                    for i in range(d):
                        text = fpt.readline().split()
                        for j in range(d):
                            rotate[i][j] = float(text[j])
                    if functions.get(name) is None:
                        print("\n Error: No such problem function: {} \n".format(name))
                        return
                    problems.append([name, dim, shift, rotate])
                instances.append(['Composition', dim, cf_num, lamda, sigma, bias, F, problems])
        return instances

    def func(self, x):  # evaluate a group of solutions
        self.FES += x.shape[0]
        w = np.zeros((x.shape[0], self.cf_num))
        for j in range(x.shape[0]):
            for i in range(self.cf_num):
                a = np.sqrt(np.sum((x[j][:self.sub_problems[i].dim] - self.sub_problems[i].shift) ** 2))
                if a != 0:
                    w[j][i] = 1 / a * np.exp(-np.sum((x[j][:self.sub_problems[i].dim] - self.sub_problems[i].shift) ** 2) / (2 * self.sub_problems[i].dim * self.sigma[i] * self.sigma[i]))
                else:
                    w[j][i] = INF
            if np.max(w[j]) == 0:
                w[j] = np.ones(self.cf_num)
        res = np.zeros(x.shape[0])
        for i in range(self.cf_num):
            fit = self.lamda[i] * self.sub_problems[i].func(x) + self.bias[i]
            res += w[:, i] / np.sum(w, -1) * fit
        return np.round(res, 15) + self.F

    # Generate a specified number of composition problems and store in a file (or not)
    @staticmethod
    def generator(dim=0,                        # problem dimensions, randomly selected if the value <= 0
                  cf_num=0,                     # the number of sub problems, also randomly selected if the value <= 0
                  problem_names=None,           # a list provides candidate sub problem names(string), these problems can be randomly selected and join or add all in order
                  size=1,                       # the size of generated dataset
                  shifted=True,                 # the three following parameters are the config of problems, whether they are shifted, rotated or biased
                  rotated=True,
                  biased=True,
                  store=False,                  # whether to store these problem data into a file
                  filename=None,                # if store, determine the file name
                  indicated_specific=False      # if true, all sub problems in problem_names will join in order, or randomly selected when false
                  ):
        # if dim parameter <= 0, randomly select one
        if dim <= 0:
            dim = np.random.randint(3, 6) * 10
        # if cf_num parameter <= 0, randomly select one
        if cf_num <= 0:
            cf_num = np.random.randint(2, 5)
        # if all sub problems in problem_names join in order, the number of sub problems should be the same as length of problem_names
        if indicated_specific and problem_names is not None:
            cf_num = len(problem_names)
        instances = []
        mode = 'w'
        for i in range(size):
            # generate problem data
            lamda = np.random.random(cf_num)
            sigma = np.random.randint(1, cf_num, cf_num) * 10
            bias = np.random.permutation(cf_num) * 100
            if biased:
                F = np.random.randint(1, 16) * 100
            else:
                F = 0
            # generate data for sub problems
            problems = []
            names = []
            for i in range(cf_num):
                if problem_names is None:  # User doesn't assign the problems in the composition
                    name = random.sample(list(functions.keys()), 1)[0]
                elif indicated_specific:
                    name = problem_names[i]
                else:
                    name = random.sample(problem_names, 1)[0]
                names.append(name)
                problems.append(functions[name].generator(dim, shifted=shifted, rotated=rotated, biased=False))
            # if user chooses to store problem set into a file
            if store:
                with open(filename, mode) as fpt:
                    fpt.write(str(cf_num) + '\n')
                    fpt.write(' '.join(str(i) for i in lamda))
                    fpt.write('\n')
                    fpt.write(' '.join(str(i) for i in sigma))
                    fpt.write('\n')
                    fpt.write(' '.join(str(i) for i in bias))
                    fpt.write('\n')
                    fpt.write(str(F))
                    fpt.write('\n')
                    for i in range(cf_num):
                        fpt.write(names[i] + ' ' + str(problems[i][1]) + ' ' + str(problems[i][4]) + '\n')
                        fpt.write(' '.join(str(j) for j in problems[i][2]))
                        fpt.write('\n')
                        for k in range(problems[i][1]):
                            fpt.write(' '.join(str(j) for j in problems[i][3][k]))
                            fpt.write('\n')
            instances.append(['Composition', dim, cf_num, lamda, sigma, bias, F, problems])
            mode = 'a'
        return instances if size > 1 else instances[0]  # for convenience, return a single object if the size is 1

    # Transfer an instance data to an instance object
    @staticmethod
    def get_instance(problem_data):
        name, dim, cf_num, lamda, sigma, bias, F, problems = problem_data
        problem = []
        for j in range(len(problems)):
            tmp = []
            for k in range(len(problems[j])):
                tmp.append(problems[j][k])
            problem.append(Problem.get_instance(tmp))
        return eval('Composition')(np.array(dim), np.array(cf_num), np.array(lamda), np.array(sigma),
                                   np.array(bias), np.array(F), problem)


def Read2005(dim, func_data, M):
    with open(func_data, 'r') as fpt:
        shifts = np.zeros(dim)
        data = fpt.readline().split()[:dim]
        for j in range(dim):
            shifts[j] = float(data[j])
    with open(M, 'r') as fpt:
        rotates = np.zeros((dim, dim))
        for j in range(dim):
            data = fpt.readline().split()[:dim]
            for k in range(dim):
                rotates[j][k] = float(data[k])
    return shifts, rotates


class Composition2005:
    def __init__(self, dim, cf_num, sub_problems, lamda, sigma, bias, C, F, func_data, M):
        self.dim = dim
        self.cf_num = cf_num
        self.lamda = lamda
        self.sigma = sigma
        self.bias = bias
        self.C = C
        self.F = F
        self.sub_problems = []
        self.shifts, self.rotates = Composition2005.read(dim, cf_num, func_data, M)
        zero = np.zeros(self.dim)
        eye = np.eye(self.dim)
        for i in range(cf_num):
            self.sub_problems.append(eval(sub_problems[i])(dim, zero, eye, 0))
        for i in range(self.cf_num):
            self.sub_problems[i].shrink = 1
        self.fmax = np.zeros(self.cf_num)
        for i in range(self.cf_num):
            self.fmax[i] = np.fabs(self.sub_problems[i].func(rotatefunc(np.ones((1, self.dim)) * 5 / self.lamda[i], self.rotates[i]))[0])
        self.FES = 0
        # calculate optimal value and solution
        self.opt = self.shifts[0]
        self.optimum = self.func(self.opt.reshape(1, -1) * 20)[0]
        self.FES = 0

    def get_optimal(self):
        return self.opt

    @staticmethod
    def read(dim, cf_num, func_data, M):  # Read a specified number of problem data from file
        with open(func_data, 'r') as fpt:
            shifts = np.zeros((cf_num, dim))
            for i in range(cf_num):
                data = fpt.readline().split()[:dim]
                for j in range(dim):
                    shifts[i][j] = float(data[j])
        with open(M, 'r') as fpt:
            rotates = np.zeros((cf_num, dim, dim))
            for i in range(cf_num):
                for j in range(dim):
                    data = fpt.readline().split()[:dim]
                    for k in range(dim):
                        rotates[i][j][k] = float(data[k])
        return shifts, rotates

    def func(self, x):  # evaluate a group of solutions
        self.FES += x.shape[0]
        x = copy.deepcopy(x * 0.05)  # x in [-5, 5]
        w = np.zeros((x.shape[0], self.cf_num))
        for j in range(x.shape[0]):
            for i in range(self.cf_num):
                a = np.sum((x[j][:self.sub_problems[i].dim] - self.shifts[i]) ** 2)
                w[j][i] = np.exp(-np.sum(a / (2 * self.dim * self.sigma[i] * self.sigma[i])))
            mid = np.argmax(w[j])
            sum_w = np.sum(w[j])
            for i in range(self.cf_num):
                if i != mid:
                    w[j][i] *= (1 - np.power(w[j][mid], 10))
            w[j] /= sum_w
        res = np.zeros(x.shape[0])
        for i in range(self.cf_num):
            z = rotatefunc((x - self.shifts[i]) / self.lamda[i], self.rotates[i])
            fit = self.sub_problems[i].func(z) * self.C[i] / self.fmax[i] + self.bias[i]
            res += w[:, i] * fit
        return np.round(res, 15) + self.F


class Composition2005F18:
    def __init__(self, dim):
        self.dim = dim
        self.cf_num = 10
        self.lamda = [2 * 5 / 32, 5 / 32, 2, 1, 2 * 5 / 100, 5 / 100, 2 * 10, 10, 2 * 5 / 60, 5 / 60]
        self.sigma = [1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2]
        self.bias = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        self.C = [2, 3, 2, 3, 2, 3, 20, 30, 200, 300]
        self.F = 10
        self.sub_problems = []
        self.shifts, self.rotates = self.read()
        zero = np.zeros(self.dim)
        eye = np.eye(self.dim)
        self.sub_problems.append(Ackley(dim, zero, eye, 0));        self.sub_problems.append(Ackley(dim, zero, eye, 0))
        self.sub_problems.append(Rastrigin(dim, zero, eye, 0));     self.sub_problems.append(Rastrigin(dim, zero, eye, 0))
        self.sub_problems.append(Sphere(dim, zero, eye, 0));        self.sub_problems.append(Sphere(dim, zero, eye, 0))
        self.sub_problems.append(Weierstrass(dim, zero, eye, 0));   self.sub_problems.append(Weierstrass(dim, zero, eye, 0))
        self.sub_problems.append(Griewank(dim, zero, eye, 0));      self.sub_problems.append(Griewank(dim, zero, eye, 0))
        for i in range(self.cf_num):
            self.sub_problems[i].shrink = 1
        self.fmax = np.zeros(self.cf_num)
        for i in range(self.cf_num):
            self.fmax[i] = np.fabs(self.sub_problems[i].func(rotatefunc(np.ones((1, self.dim)) * 5 / self.lamda[i], self.rotates[i]))[0])
        self.FES = 0
        # calculate optimal value and solution
        self.opt = self.shifts[0]
        self.optimum = self.func(self.opt.reshape(1, -1) * 20)[0]
        self.FES = 0

    def get_optimal(self):
        return self.opt

    def read(self):  # Read a specified number of problem data from file
        with open('cec2005data/hybrid_func2_data.txt', 'r') as fpt:
            shifts = np.zeros((10, self.dim))
            for i in range(10):
                data = fpt.readline().split()[:self.dim]
                for j in range(self.dim):
                    shifts[i][j] = float(data[j])
        with open('cec2005data/hybrid_func2_M_D10.txt', 'r') as fpt:
            rotates = np.zeros((10, self.dim, self.dim))
            for i in range(10):
                for j in range(self.dim):
                    data = fpt.readline().split()[:self.dim]
                    for k in range(self.dim):
                        rotates[i][j][k] = float(data[k])
        return shifts, rotates

    def func(self, x):  # evaluate a group of solutions
        self.FES += x.shape[0]
        x = copy.deepcopy(x * 0.05)  # x in [-5, 5]
        w = np.zeros((x.shape[0], self.cf_num))
        for j in range(x.shape[0]):
            for i in range(self.cf_num):
                a = np.sum((x[j][:self.sub_problems[i].dim] - self.shifts[i]) ** 2)
                w[j][i] = np.exp(-np.sum(a / (2 * self.dim * self.sigma[i] * self.sigma[i])))
            mid = np.argmax(w[j])
            sum_w = np.sum(w[j])
            for i in range(self.cf_num):
                if i != mid:
                    w[j][i] *= (1 - np.power(w[j][mid], 10))
            w[j] /= sum_w
        res = np.zeros(x.shape[0])
        for i in range(self.cf_num):
            z = rotatefunc((x - self.shifts[i]) / self.lamda[i], self.rotates[i])
            fit = self.sub_problems[i].func(z) * self.C[i] / self.fmax[i] + self.bias[i]
            res += w[:, i] * fit
        return np.round(res, 15) + self.F


class Composition2005F23:
    def __init__(self, dim):
        self.dim = dim
        self.cf_num = 10
        self.lamda = [5*5/100, 5/100, 5*1, 1, 5*1, 1, 5*10, 10, 5*5/200, 5/200]
        self.sigma = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        self.bias = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        self.C = [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, ]
        self.F = 360
        self.sub_problems = []
        self.shifts, self.rotates = self.read()
        zero = np.zeros(self.dim)
        eye = np.eye(self.dim)
        self.sub_problems.append(Ackley(dim, zero, eye, 0));        self.sub_problems.append(Ackley(dim, zero, eye, 0))
        self.sub_problems.append(Rastrigin(dim, zero, eye, 0));     self.sub_problems.append(Rastrigin(dim, zero, eye, 0))
        self.sub_problems.append(Sphere(dim, zero, eye, 0));        self.sub_problems.append(Sphere(dim, zero, eye, 0))
        self.sub_problems.append(Weierstrass(dim, zero, eye, 0));   self.sub_problems.append(Weierstrass(dim, zero, eye, 0))
        self.sub_problems.append(Griewank(dim, zero, eye, 0));      self.sub_problems.append(Griewank(dim, zero, eye, 0))
        for i in range(self.cf_num):
            self.sub_problems[i].shrink = 1
        self.fmax = np.zeros(self.cf_num)
        for i in range(self.cf_num):
            self.fmax[i] = np.fabs(
                self.sub_problems[i].func(rotatefunc(np.ones((1, self.dim)) * 5 / self.lamda[i], self.rotates[i])))[0]
        self.FES = 0
        # calculate optimal value and solution
        self.opt = self.shifts[0]
        self.optimum = self.func(self.opt.reshape(1, -1) * 20)[0]
        self.FES = 0

    def get_optimal(self):
        return self.opt

    def read(self):  # Read a specified number of problem data from file
        with open('cec2005data/hybrid_func3_data.txt', 'r') as fpt:
            shifts = np.zeros((10, self.dim))
            for i in range(10):
                data = fpt.readline().split()[:self.dim]
                for j in range(self.dim):
                    shifts[i][j] = float(data[j])
        with open('cec2005data/hybrid_func3_M_D10.txt', 'r') as fpt:
            rotates = np.zeros((10, self.dim, self.dim))
            for i in range(10):
                for j in range(self.dim):
                    data = fpt.readline().split()[:self.dim]
                    for k in range(self.dim):
                        rotates[i][j][k] = float(data[k])
        return shifts, rotates

    def func(self, x):  # evaluate a group of solutions
        self.FES += x.shape[0]
        x = copy.deepcopy(x * 0.05)  # x in [-5, 5]
        flag = np.fabs(x - self.opt) >= 0.5
        x[flag] = np.round(x[flag] * 2) / 2
        w = np.zeros((x.shape[0], self.cf_num))
        for j in range(x.shape[0]):
            for i in range(self.cf_num):
                a = np.sum((x[j][:self.sub_problems[i].dim] - self.shifts[i]) ** 2)
                w[j][i] = np.exp(-np.sum(a / (2 * self.dim * self.sigma[i] * self.sigma[i])))
            mid = np.argmax(w[j])
            sum_w = np.sum(w[j])
            for i in range(self.cf_num):
                if i != mid:
                    w[j][i] *= (1 - np.power(w[j][mid], 10))
            w[j] /= sum_w
        res = np.zeros(x.shape[0])
        for i in range(self.cf_num):
            z = rotatefunc((x - self.shifts[i]) / self.lamda[i], self.rotates[i])
            fit = self.sub_problems[i].func(z) * self.C[i] / self.fmax[i] + self.bias[i]
            res += w[:, i] * fit
        return np.round(res, 15) + self.F





