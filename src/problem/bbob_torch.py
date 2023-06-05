import math
from problem.basic_problem import Basic_Problem
import numpy as np
import torch
from torch.utils.data import Dataset


def sr_func(x, Os, Mr):  # shift and rotate
    y = x[:, :Os.shape[-1]] - Os
    return torch.matmul(Mr, y.t()).t()


def rotate_gen(dim):  # Generate a rotate matrix
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
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
    return torch.tensor(H, dtype=torch.float64)


class BBOB_Basic_Problem_torch(Basic_Problem):
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.lb = lb
        self.ub = ub
        self.FES = 0
        self.opt = self.shift
        # self.optimum = self.eval(self.get_optimal())
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        return self.opt

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
        raise NotImplementedError


def osc_transform(x):
    """
    Implementing the oscillating transformation on objective values or/and decision values.

    :param x: If x represents objective values, x is a 1-D array in shape [NP] if problem is single objective,
              or a 2-D array in shape [NP, number_of_objectives] if multi-objective.
              If x represents decision values, x is a 2-D array in shape [NP, dim].
    :return: The array after transformation in the shape of x.
    """
    y = x.clone()
    idx = (x > 0.)
    y[idx] = torch.log(x[idx]) / 0.1
    y[idx] = torch.exp(y[idx] + 0.49 * (torch.sin(y[idx]) + torch.sin(0.79 * y[idx]))) ** 0.1
    idx = (x < 0.)
    y[idx] = torch.log(-x[idx]) / 0.1
    y[idx] = -torch.exp(y[idx] + 0.49 * (torch.sin(0.55 * y[idx]) + torch.sin(0.31 * y[idx]))) ** 0.1
    return y


def asy_transform(x, beta):
    """
    Implementing the asymmetric transformation on decision values.

    :param x: Decision values in shape [NP, dim].
    :param beta: beta factor.
    :return: The array after transformation in the shape of x.
    """
    NP, dim = x.shape
    idx = (x > 0.)
    y = x.clone()
    y[idx] = y[idx] ** (
                1. + beta * torch.linspace(0, 1, dim, dtype=torch.float64).reshape(1, -1).repeat(repeats=(NP, 1))[idx] * torch.sqrt(y[idx]))
    return y


def pen_func(x, ub):
    """
    Implementing the penalty function on decision values.

    :param x: Decision values in shape [NP, dim].
    :param ub: the upper-bound as a scalar.
    :return: Penalty values in shape [NP].
    """
    return torch.sum(torch.maximum(torch.tensor(0., dtype=torch.float64), torch.abs(x) - ub) ** 2, dim=-1)


class NoisyProblem:
    def noisy(self, ftrue):
        raise NotImplementedError

    def eval(self, x):
        ftrue = super().eval(x)
        return self.noisy(ftrue)

    def boundaryHandling(self, x):
        return 100. * pen_func(x, self.ub)


class GaussNoisyProblem(NoisyProblem):
    """
    Attribute 'gause_beta' need to be defined in subclass.
    """

    def noisy(self, ftrue):
        if not isinstance(ftrue, torch.Tensor):
            ftrue = torch.tensor(ftrue, dtype=torch.float64)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased * torch.exp(self.gauss_beta * torch.randn(ftrue_unbiased.shape, dtype=torch.float64))
        return torch.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)


class UniformNoisyProblem(NoisyProblem):
    """
    Attributes 'uniform_alpha' and 'uniform_beta' need to be defined in subclass.
    """

    def noisy(self, ftrue):
        if not isinstance(ftrue, torch.Tensor):
            ftrue = torch.tensor(ftrue, dtype=torch.float64)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased * (torch.rand(ftrue_unbiased.shape, dtype=torch.float64) ** self.uniform_beta) * \
                          torch.maximum(torch.tensor(1., dtype=torch.float64), (1e9 / (ftrue_unbiased + 1e-99)) ** (self.uniform_alpha * (0.49 + 1. / self.dim) * torch.rand(ftrue_unbiased.shape, dtype=torch.float64)))
        return torch.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)


class CauchyNoisyProblem(NoisyProblem):
    """
    Attributes 'cauchy_alpha' and 'cauchy_p' need to be defined in subclass.
    """

    def noisy(self, ftrue):
        if not isinstance(ftrue, torch.Tensor):
            ftrue = torch.tensor(ftrue, dtype=torch.float64)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased + self.cauchy_alpha * torch.maximum(torch.tensor(0., dtype=torch.float64), 1e3 + (torch.rand(ftrue_unbiased.shape, dtype=torch.float64) < self.cauchy_p) * torch.randn(ftrue_unbiased.shape, dtype=torch.float64) / (torch.abs(torch.randn(ftrue_unbiased.shape, dtype=torch.float64)) + 1e-199))
        return torch.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)


class _Sphere(BBOB_Basic_Problem_torch):
    """
    Abstract Sphere
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2, dim=-1) + self.bias + self.boundaryHandling(x)


class F1(_Sphere):
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Sphere'


class F101(GaussNoisyProblem, _Sphere):
    gauss_beta = 0.01
    def __str__(self):
        return 'Sphere_moderate_gauss'


class F102(UniformNoisyProblem, _Sphere):
    uniform_alpha = 0.01
    uniform_beta = 0.01
    def __str__(self):
        return 'Sphere_moderate_uniform'


class F103(CauchyNoisyProblem, _Sphere):
    cauchy_alpha = 0.01
    cauchy_p = 0.05
    def __str__(self):
        return 'Sphere_moderate_cauchy'


class F107(GaussNoisyProblem, _Sphere):
    gauss_beta = 1.
    def __str__(self):
        return 'Sphere_gauss'


class F108(UniformNoisyProblem, _Sphere):
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Sphere_uniform'


class F109(CauchyNoisyProblem, _Sphere):
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Sphere_cauchy'


class F2(BBOB_Basic_Problem_torch):
    """
    Ellipsoidal
    """

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Ellipsoidal'

    def func(self, x):
        self.FES += x.shape[0]
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate)
        z = osc_transform(z)
        i = torch.arange(nx, dtype=torch.float64)
        return torch.sum(torch.pow(10, 6 * i / (nx - 1)) * (z ** 2), -1) + self.bias


class F3(BBOB_Basic_Problem_torch):
    """
    Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        self.scales = (10. ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rastrigin'

    def func(self, x):
        self.FES += x.shape[0]
        z = self.scales * asy_transform(osc_transform(sr_func(x, self.shift, self.rotate)), beta=0.2)
        return 10. * (self.dim - torch.sum(torch.cos(2. * math.pi * z), dim=-1)) + torch.sum(z ** 2, dim=-1) + self.bias


class F4(BBOB_Basic_Problem_torch):
    """
    Bueche_Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift[::2] = torch.abs(shift[::2])
        self.scales = ((10. ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64))
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Buche_Rastrigin'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = osc_transform(z)
        even = z[:, ::2]
        even[even > 0.] *= 10.
        z *= self.scales
        return 10 * (self.dim - torch.sum(torch.cos(2 * math.pi * z), dim=-1)) + torch.sum(z ** 2, dim=-1) + 100 * pen_func(x, self.ub) + self.bias


class F5(BBOB_Basic_Problem_torch):
    """
    Linear_Slope
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift = np.sign(shift.numpy())
        shift[shift == 0.] = np.random.choice([-1., 1.], size=(shift == 0.).sum())
        shift = torch.tensor(shift, dtype=torch.float64) * ub
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Linear_Slope'

    def func(self, x):
        self.FES += x.shape[0]
        z = x.clone()
        exceed_bound = (x * self.shift) > (self.ub ** 2)
        z[exceed_bound] = torch.sign(z[exceed_bound]) * self.ub  # clamp back into the domain
        s = torch.sign(self.shift) * (10 ** torch.linspace(0, 1, self.dim, dtype=torch.float64))
        return torch.sum(self.ub * torch.abs(s) - z * s, dim=-1) + self.bias


class F6(BBOB_Basic_Problem_torch):
    """
    Attractive_Sector
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        rotate = torch.matmul(torch.matmul(rotate_gen(dim), torch.diag(scales)), rotate)
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Attractive_Sector'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        idx = (z * self.get_optimal()) > 0.
        z[idx] *= 100.
        return osc_transform(torch.sum(z ** 2, -1)) ** 0.9 + self.bias


class _Step_Ellipsoidal(BBOB_Basic_Problem_torch):
    """
    Abstract Step_Ellipsoidal
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        rotate = torch.matmul(torch.diag(scales), rotate)
        self.Q_rotate = rotate_gen(dim)
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z_hat = sr_func(x, self.shift, self.rotate)
        z = torch.matmul(torch.where(torch.abs(z_hat) > 0.5, torch.floor(0.5 + z_hat), torch.floor(0.5 + 10. * z_hat) / 10.),
                         self.Q_rotate.T)
        return 0.1 * torch.maximum(torch.abs(z_hat[:, 0]) / 1e4,
                                   torch.sum(100 ** torch.linspace(0, 1, self.dim, dtype=torch.float64) * (z ** 2), dim=-1)) + \
               self.boundaryHandling(x) + self.bias


class F7(_Step_Ellipsoidal):
    def boundaryHandling(self, x):
        return pen_func(x, self.ub)

    def __str__(self):
        return 'Step_Ellipsoidal'


class F113(GaussNoisyProblem, _Step_Ellipsoidal):
    gauss_beta = 1.
    def __str__(self):
        return 'Step_Ellipsoidal_gauss'


class F114(UniformNoisyProblem, _Step_Ellipsoidal):
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Step_Ellipsoidal_uniform'


class F115(CauchyNoisyProblem, _Step_Ellipsoidal):
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Step_Ellipsoidal_cauchy'


class _Rosenbrock(BBOB_Basic_Problem_torch):
    """
    Abstract Rosenbrock_original
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift *= 0.75  # range_of_shift=0.8*0.75*ub=0.6*ub
        rotate = torch.eye(dim, dtype=torch.float64)
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = max(1., self.dim ** 0.5 / 8.) * sr_func(x, self.shift, self.rotate) + 1
        return torch.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2,
                         dim=-1) + self.bias + self.boundaryHandling(x)


class F8(_Rosenbrock):
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Rosenbrock_original'


class F104(GaussNoisyProblem, _Rosenbrock):
    gauss_beta = 0.01
    def __str__(self):
        return 'Rosenbrock_moderate_gauss'


class F105(UniformNoisyProblem, _Rosenbrock):
    uniform_alpha = 0.01
    uniform_beta = 0.01
    def __str__(self):
        return 'Rosenbrock_moderate_uniform'


class F106(CauchyNoisyProblem, _Rosenbrock):
    cauchy_alpha = 0.01
    cauchy_p = 0.05
    def __str__(self):
        return 'Rosenbrock_moderate_cauchy'


class F110(GaussNoisyProblem, _Rosenbrock):
    gauss_beta = 1.
    def __str__(self):
        return 'Rosenbrock_gauss'


class F111(UniformNoisyProblem, _Rosenbrock):
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Rosenbrock_uniform'


class F112(CauchyNoisyProblem, _Rosenbrock):
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Rosenbrock_cauchy'


class F9(BBOB_Basic_Problem_torch):
    """
    Rosenbrock_rotated
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scale = max(1., dim ** 0.5 / 8.)
        self.linearTF = scale * rotate
        shift = torch.matmul(0.5 * torch.ones(dim, dtype=torch.float64), self.linearTF) / (scale ** 2)
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rosenbrock_rotated'

    def func(self, x):
        self.FES += x.shape[0]
        z = torch.matmul(x, self.linearTF.T) + 0.5
        return torch.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, dim=-1) + self.bias


class _Ellipsoidal(BBOB_Basic_Problem_torch):
    """
    Abstract Ellipsoidal
    """
    condition = None

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate)
        z = osc_transform(z)
        i = torch.arange(nx, dtype=torch.float64)
        return torch.sum((self.condition ** (i / (nx - 1))) * (z ** 2), -1, dtype=torch.float64) + self.bias + self.boundaryHandling(x)


class F10(_Ellipsoidal):
    condition = 1e6
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Ellipsoidal_high_cond'


class F116(GaussNoisyProblem, _Ellipsoidal):
    condition = 1e4
    gauss_beta = 1.
    def __str__(self):
        return 'Ellipsoidal_gauss'


class F117(UniformNoisyProblem, _Ellipsoidal):
    condition = 1e4
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Ellipsoidal_uniform'


class F118(CauchyNoisyProblem, _Ellipsoidal):
    condition = 1e4
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Ellipsoidal_cauchy'


class F11(BBOB_Basic_Problem_torch):
    """
    Discus
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Discus'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = osc_transform(z)
        return 1e6 * (z[:, 0] ** 2) + torch.sum(z[:, 1:] ** 2, -1) + self.bias


class F12(BBOB_Basic_Problem_torch):
    """
    Bent_cigar
    """
    beta = 0.5

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Bent_Cigar'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = asy_transform(z, beta=self.beta)
        z = torch.matmul(z, self.rotate.T)
        return z[:, 0] ** 2 + torch.sum(1e6 * (z[:, 1:] ** 2), -1, dtype=torch.float64) + self.bias


class F13(BBOB_Basic_Problem_torch):
    """
    Sharp Ridge
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10 ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        rotate = torch.matmul(torch.matmul(rotate_gen(dim), torch.diag(scales)), rotate)
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Sharp_Ridge'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        return z[:, 0] ** 2. + 100. * torch.sqrt(torch.sum(z[:, 1:] ** 2., dim=-1)) + self.bias


class _Dif_powers(BBOB_Basic_Problem_torch):
    """
    Abstract Different Powers
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        i = torch.arange(self.dim, dtype=torch.float64)
        return torch.pow(torch.sum(torch.pow(torch.abs(z), 2 + 4 * i / max(1, self.dim - 1)), -1),
                         0.5) + self.bias + self.boundaryHandling(x)


class F14(_Dif_powers):
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Different_Powers'


class F119(GaussNoisyProblem, _Dif_powers):
    gauss_beta = 1.
    def __str__(self):
        return 'Different_Powers_gauss'


class F120(UniformNoisyProblem, _Dif_powers):
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Different_Powers_uniform'


class F121(CauchyNoisyProblem, _Dif_powers):
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Different_Powers_cauchy'


class F15(BBOB_Basic_Problem_torch):
    """
    Rastrigin_F15
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        self.linearTF = torch.matmul(torch.matmul(rotate, torch.diag(scales)), rotate_gen(dim))
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rastrigin_F15'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = asy_transform(osc_transform(z), beta=0.2)
        z = torch.matmul(z, self.linearTF.T)
        return 10. * (self.dim - torch.sum(torch.cos(2. * math.pi * z), dim=-1)) + torch.sum(z ** 2, dim=-1) + self.bias


class F16(BBOB_Basic_Problem_torch):
    """
    Weierstrass
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (0.01 ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        self.linearTF = torch.matmul(torch.matmul(rotate, torch.diag(scales)), rotate_gen(dim))
        self.aK = 0.5 ** torch.arange(12, dtype=torch.float64)
        self.bK = 3.0 ** torch.arange(12, dtype=torch.float64)
        self.f0 = torch.sum(self.aK * torch.cos(math.pi * self.bK))
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Weierstrass'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = torch.matmul(osc_transform(z), self.linearTF.T)
        return 10 * torch.pow(
            torch.mean(torch.sum(self.aK * torch.cos(torch.matmul(2 * math.pi * (z[:, :, None] + 0.5), self.bK[None, :])), dim=-1),
                       dim=-1) - self.f0, 3) + \
               10 / self.dim * pen_func(x, self.ub) + self.bias


class _Scaffer(BBOB_Basic_Problem_torch):
    """
    Abstract Scaffers
    """
    condition = None  # need to be defined in subclass

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (self.condition ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        self.linearTF = torch.matmul(torch.diag(scales), rotate_gen(dim))
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = torch.matmul(asy_transform(z, beta=0.5), self.linearTF.T)
        s = torch.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)
        return torch.pow(
            1 / (self.dim - 1) * torch.sum(torch.sqrt(s) * (torch.pow(torch.sin(50 * torch.pow(s, 0.2)), 2) + 1), dim=-1), 2) + \
               self.boundaryHandling(x) + self.bias


class F17(_Scaffer):
    condition = 10.

    def boundaryHandling(self, x):
        return 10 * pen_func(x, self.ub)

    def __str__(self):
        return 'Schaffers'


class F18(_Scaffer):
    condition = 1000.

    def boundaryHandling(self, x):
        return 10 * pen_func(x, self.ub)

    def __str__(self):
        return 'Schaffers_high_cond'


class F122(GaussNoisyProblem, _Scaffer):
    condition = 10.
    gauss_beta = 1.
    def __str__(self):
        return 'Schaffers_gauss'


class F123(UniformNoisyProblem, _Scaffer):
    condition = 10.
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Schaffers_uniform'


class F124(CauchyNoisyProblem, _Scaffer):
    condition = 10.
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Schaffers_cauchy'


class _Composite_Grie_rosen(BBOB_Basic_Problem_torch):
    """
    Abstract Composite_Grie_rosen
    """
    factor = None

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scale = max(1., dim ** 0.5 / 8.)
        self.linearTF = scale * rotate
        shift = torch.matmul(0.5 * torch.ones(dim, dtype=torch.float64) / (scale ** 2.), self.linearTF)
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = torch.matmul(x, self.linearTF.T) + 0.5
        s = 100. * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (1. - z[:, :-1]) ** 2
        return self.factor + self.factor * torch.sum(s / 4000. - torch.cos(s), dim=-1) / (
                    self.dim - 1.) + self.bias + self.boundaryHandling(x)


class F19(_Composite_Grie_rosen):
    factor = 10.

    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Composite_Grie_rosen'


class F125(GaussNoisyProblem, _Composite_Grie_rosen):
    factor = 1.
    gauss_beta = 1.
    def __str__(self):
        return 'Composite_Grie_rosen_gauss'


class F126(UniformNoisyProblem, _Composite_Grie_rosen):
    factor = 1.
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Composite_Grie_rosen_uniform'


class F127(CauchyNoisyProblem, _Composite_Grie_rosen):
    factor = 1.
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Composite_Grie_rosen_cauchy'


class F20(BBOB_Basic_Problem_torch):
    """
    Schwefel
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift = 0.5 * 4.2096874633 * torch.tensor(np.random.choice([-1., 1.], size=dim), dtype=torch.float64)
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Schwefel'

    def func(self, x):
        self.FES += x.shape[0]
        tmp = 2 * torch.abs(self.shift)
        scales = (10 ** 0.5) ** torch.linspace(0, 1, self.dim, dtype=torch.float64)
        z = 2 * torch.sign(self.shift) * x
        z[:, 1:] += 0.25 * (z[:, :-1] - tmp[:-1])
        z = 100. * (scales * (z - tmp) + tmp)
        b = 4.189828872724339
        return b - 0.01 * torch.mean(z * torch.sin(torch.sqrt(torch.abs(z))), dim=-1) + 100 * pen_func(z / 100,
                                                                                            self.ub) + self.bias


class _Gallagher(BBOB_Basic_Problem_torch):
    """
    Abstract Gallagher
    """
    n_peaks = None

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        # problem param config
        if self.n_peaks == 101:   # F21
            opt_shrink = 1.       # shrink of global & local optima
            global_opt_alpha = 1e3
        elif self.n_peaks == 21:  # F22
            opt_shrink = 0.98     # shrink of global & local optima
            global_opt_alpha = 1e6
        else:
            raise ValueError(f'{self.n_peaks} peaks Gallagher is not supported yet.')

        # generate global & local optima y[i]
        self.y = opt_shrink * torch.tensor(np.random.rand(self.n_peaks, dim) * (ub - lb) + lb, dtype=torch.float64)  # [n_peaks, dim]
        self.y[0] = shift * opt_shrink  # the global optimum
        shift = self.y[0]

        # generate the matrix C[i]
        sqrt_alpha = 1000 ** np.random.permutation(np.linspace(0, 1, self.n_peaks - 1))
        sqrt_alpha = np.insert(sqrt_alpha, obj=0, values=np.sqrt(global_opt_alpha))
        self.C = [torch.tensor(np.random.permutation(sqrt_alpha[i] ** np.linspace(-0.5, 0.5, dim)), dtype=torch.float64) for i in range(self.n_peaks)]
        self.C = torch.vstack(self.C)  # [n_peaks, dim]

        # generate the weight w[i]
        self.w = torch.cat([torch.tensor([10.], dtype=torch.float64), torch.linspace(1.1, 9.1, self.n_peaks - 1, dtype=torch.float64)], dim=0)  # [n_peaks]

        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = torch.matmul(x[:, None, :].repeat((1, self.n_peaks, 1)) - self.y,
                         self.rotate.T)  # [NP, n_peaks, dim]
        z = torch.max(self.w * torch.exp((-0.5 / self.dim) * torch.sum(self.C * (z ** 2), dim=-1)), dim=-1)[0]  # [NP]
        return osc_transform(10 - z) ** 2 + self.bias + self.boundaryHandling(x)


class F21(_Gallagher):
    n_peaks = 101

    def boundaryHandling(self, x):
        return pen_func(x, self.ub)

    def __str__(self):
        return 'Gallagher_101Peaks'


class F22(_Gallagher):
    n_peaks = 21

    def boundaryHandling(self, x):
        return pen_func(x, self.ub)

    def __str__(self):
        return 'Gallagher_21Peaks'


class F128(GaussNoisyProblem, _Gallagher):
    n_peaks = 101
    gauss_beta = 1.
    def __str__(self):
        return 'Gallagher_101Peaks_gauss'


class F129(UniformNoisyProblem, _Gallagher):
    n_peaks = 101
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Gallagher_101Peaks_uniform'


class F130(CauchyNoisyProblem, _Gallagher):
    n_peaks = 101
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Gallagher_101Peaks_cauchy'


class F23(BBOB_Basic_Problem_torch):
    """
    Katsuura
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (100. ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        rotate = torch.matmul(torch.matmul(rotate_gen(dim), torch.diag(scales)), rotate)
        BBOB_Basic_Problem_torch.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Katsuura'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        tmp3 = self.dim ** 1.2
        tmp1 = torch.pow(torch.ones((1, 32), dtype=torch.float64) * 2, torch.arange(1, 33, dtype=torch.float64)).repeat((x.shape[0], 1))
        res = torch.ones(x.shape[0], dtype=torch.float64)
        for i in range(self.dim):
            tmp2 = tmp1 * z[:, i, None].repeat((1, 32))
            temp = torch.sum(torch.abs(tmp2 - torch.floor(tmp2 + 0.5)) / tmp1, -1)
            res *= torch.pow(1 + (i + 1) * temp, 10 / tmp3)
        tmp = 10 / self.dim / self.dim
        return res * tmp - tmp + pen_func(x, self.ub) + self.bias


class F24(BBOB_Basic_Problem_torch):
    """
    Lunacek_bi_Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        self.mu0 = 2.5 / 5 * ub
        shift = torch.tensor(np.random.choice([-1., 1.], size=dim) * self.mu0 / 2, dtype=torch.float64)
        scales = (100 ** 0.5) ** torch.linspace(0, 1, dim, dtype=torch.float64)
        rotate = torch.matmul(torch.matmul(rotate_gen(dim), torch.diag(scales)), rotate)
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Lunacek_bi_Rastrigin'

    def func(self, x):
        self.FES += x.shape[0]
        x_hat = 2. * torch.sign(self.shift) * x
        z = torch.matmul(x_hat - self.mu0, self.rotate.T)
        s = 1. - 1. / (2. * math.sqrt(self.dim + 20.) - 8.2)
        mu1 = -math.sqrt((self.mu0 ** 2 - 1) / s)
        return torch.minimum(torch.sum((x_hat - self.mu0) ** 2., dim=-1),
                             self.dim + s * torch.sum((x_hat - mu1) ** 2., dim=-1)) + \
               10. * (self.dim - torch.sum(torch.cos(2. * math.pi * z), dim=-1)) + 1e4 * pen_func(x, self.ub) + self.bias


class BBOB_Dataset_torch(Dataset):
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
    def get_datasets(suit,
                     dim,
                     upperbound,
                     shifted=True,
                     rotated=True,
                     biased=True,
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy',
                     dataset_seed=1035,
                     instance_seed=3849):
        # get functions ID of indicated suit
        if suit == 'bbob-torch':
            func_id = [i for i in range(1, 25)]  # [1, 24]
        elif suit == 'bbob-noisy-torch':
            func_id = [i for i in range(101, 131)]  # [101, 130]
        else:
            raise ValueError(f'{suit} function suit is invalid or is not supported yet.')
        # get problem instances
        if instance_seed > 0:
            np.random.seed(instance_seed)
        data = []
        assert upperbound >= 5, 'Argument upperbound must be at least 5.'
        ub = upperbound
        lb = -upperbound
        for id in func_id:
            if shifted:
                shift = 0.8 * torch.tensor(np.random.random(dim) * (ub - lb) + lb, dtype=torch.float64)
            else:
                shift = torch.zeros(dim, dtype=torch.float64)
            if rotated:
                H = rotate_gen(dim)
            else:
                H = torch.eye(dim, dtype=torch.float64)
            if biased:
                bias = torch.tensor(np.random.randint(1, 26)) * 100
            else:
                bias = 0
            data.append(eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub))
        # apart train set and test set
        if difficulty == 'easy':
            train_set_ratio = 0.75
        elif difficulty == 'difficult':
            train_set_ratio = 0.25
        else:
            raise ValueError
        if dataset_seed > 0:
            np.random.seed(dataset_seed)
        else:
            np.random.seed(None)
        np.random.shuffle(data)
        n_train_func = int(len(data) * train_set_ratio)
        return BBOB_Dataset_torch(data[:n_train_func], train_batch_size), BBOB_Dataset_torch(data[n_train_func:], test_batch_size)

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

    def shuffle(self):
        self.index = np.random.permutation(self.N)
