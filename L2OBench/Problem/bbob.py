from L2OBench.Problem import sr_func, rotate_gen, Problem
import numpy as np
from torch.utils.data import Dataset


def osc_transform(x):
    """
    Implementing the oscillating transformation on objective values or/and decision values.

    :param x: If x represents objective values, x is a 1-D array in shape [NP] if problem is single objective,
              or a 2-D array in shape [NP, number_of_objectives] if multi-objective.
              If x represents decision values, x is a 2-D array in shape [NP, dim].
    :return: The array after transformation in the shape of x.
    """
    y = x.copy()
    idx = (x > 0.)
    y[idx] = np.log(x[idx]) / 0.1
    y[idx] = np.exp(y[idx] + 0.49 * (np.sin(y[idx]) + np.sin(0.79 * y[idx]))) ** 0.1
    idx = (x < 0.)
    y[idx] = np.log(-x[idx]) / 0.1
    y[idx] = -np.exp(y[idx] + 0.49 * (np.sin(0.55 * y[idx]) + np.sin(0.31 * y[idx]))) ** 0.1
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
    y = x.copy()
    y[idx] = y[idx] ** (1. + beta * np.linspace(0, 1, dim).reshape(1, -1).repeat(repeats=NP, axis=0)[idx] * np.sqrt(y[idx]))
    return y


def pen_func(x):
    """
    Implementing the penalty function on decision values.

    :param x: Decision values in shape [NP, dim].
    :return: Penalty values in shape [NP].
    """
    return np.sum(np.maximum(0., np.abs(x) - 5) ** 2, axis=-1)


class NoisyProblem:
    def noisy(self, ftrue):
        raise NotImplementedError

    def eval(self, x):
        ftrue = super().eval(x)
        return ftrue, self.noisy(ftrue)

    @staticmethod
    def boundaryHandling(x):
        return 100. * pen_func(x)


class GaussNoisyProblem(NoisyProblem):
    """
    Attribute 'gause_beta' need to be defined in subclass.
    """
    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        fnoisy = ftrue * np.exp(self.gauss_beta * np.random.randn(*ftrue.shape))
        return np.where(ftrue >= 1e-8, fnoisy + 1.01 * 1e-8, ftrue)


class UniformNoisyProblem(NoisyProblem):
    """
    Attributes 'uniform_alpha' and 'uniform_beta' need to be defined in subclass.
    """
    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        fnoisy = ftrue * (np.random.rand(*ftrue.shape) ** self.uniform_beta) * np.maximum(1., (1e9 / (ftrue + 1e-99)) ** (self.uniform_alpha * np.random.rand(*ftrue.shape)))
        return np.where(ftrue >= 1e-8, fnoisy + 1.01 * 1e-8, ftrue)


class CauchyNoisyProblem(NoisyProblem):
    """
    Attributes 'cauchy_alpha' and 'cauchy_p' need to be defined in subclass.
    """
    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        fnoisy = ftrue + self.cauchy_alpha * np.maximum(0., 1e3 + (np.random.rand(*ftrue.shape) < self.cauchy_p) * np.random.randn(*ftrue.shape) / (np.abs(np.random.randn(*ftrue.shape)) + 1e-199))
        return np.where(ftrue >= 1e-8, fnoisy + 1.01 * 1e-8, ftrue)

    
class _Sphere(Problem):
    """
    Abstract Sphere
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        super().__init__(dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = (x - self.shift) * self.shrink
        return np.sum(z ** 2, axis=-1) + self.bias + self.boundaryHandling(x * self.shrink)


class F1(_Sphere):
    def boundaryHandling(self, x):
        return 0.


class F101(GaussNoisyProblem, _Sphere):
    gauss_beta = 0.01


class F102(UniformNoisyProblem, _Sphere):
    uniform_alpha = 0.01
    uniform_beta = 0.01


class F103(CauchyNoisyProblem, _Sphere):
    cauchy_alpha = 0.01
    cauchy_p = 0.05


class F107(GaussNoisyProblem, _Sphere):
    gauss_beta = 1.


class F108(UniformNoisyProblem, _Sphere):
    uniform_alpha = 1.
    uniform_beta = 1.


class F109(CauchyNoisyProblem, _Sphere):
    cauchy_beta = 1.
    cauchy_p = 0.2


class F2(Problem):
    """
    Ellipsoidal
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, np.eye(dim), bias)

    def func(self, x):
        self.FES += x.shape[0]
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = osc_transform(z)  # 跟Ellipsoidal的唯一区别
        i = np.arange(nx)
        return np.sum(np.power(10, 6 * i / (nx - 1)) * (z ** 2), -1) + self.bias


class F3(Problem):
    """
    Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        self.scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = self.scales * asy_transform(osc_transform((x - self.shift) * self.shrink), beta=0.2)
        return 10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + self.bias


class F4(Problem):
    """
    Bueche_Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        shift[::2] = np.abs(shift[::2])
        Problem.__init__(self, dim, shift, np.eye(dim), bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = osc_transform(z)
        even = z[:, ::2]
        even[even > 0.] *= 10.
        z *= (np.sqrt(10.) ** np.linspace(0, 1, self.dim))
        return 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + 100 * pen_func(x * self.shrink) + self.bias


class F5(Problem):
    """
    Linear_Slope
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        shift = np.sign(shift)
        shift[shift == 0.] = np.random.choice([-1., 1.], size=(shift == 0.).sum())
        # shift = shift * self.ub
        shift = shift * 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = x.copy()
        exceed_bound = (x * self.shift) > (self.ub ** 2)
        # z[exceed_bound] = np.sign(z[exceed_bound]) * self.ub  # clamp back into the domain
        z[exceed_bound] = np.sign(z[exceed_bound]) * 100  # clamp back into the domain
        s = np.sign(self.shift) * (10 ** np.linspace(0, 1, self.dim))
        return np.sum(self.ub * self.shrink * np.abs(s) - z * self.shrink * s, axis=-1) + self.bias


class F6(Problem):
    """
    Attractive_Sector
    """
    condition = 10.

    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        scales = (self.condition ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(scales)), rotate)
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        idx = (z * self.get_optimal()) > 0.
        z[idx] *= 100.
        return osc_transform(np.sum(z ** 2, -1)) ** 0.9 + self.bias


class _Step_Ellipsoidal(Problem):
    """
    Abstract Step_Ellipsoidal
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5. / 100
        self.scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        self.Q_rotate = rotate_gen(dim)
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z_hat = self.scales * sr_func(x, self.shift, self.rotate, self.shrink)
        z = np.matmul(np.where(np.abs(z_hat) > 0.5, np.floor(0.5 + z_hat), np.floor(0.5 + 10. * z_hat) / 10.), self.Q_rotate.T)
        return 0.1 * np.maximum(np.abs(z_hat[:, 0]) / 1e4, np.sum(100 ** np.linspace(0, 1, self.dim) * (z ** 2), axis=-1)) + \
               self.boundaryHandling(x * self.shrink) + self.bias


class F7(_Step_Ellipsoidal):
    def boundaryHandling(self, x):
        return pen_func(x)


class F113(GaussNoisyProblem, _Step_Ellipsoidal):
    gauss_beta = 1.


class F114(UniformNoisyProblem, _Step_Ellipsoidal):
    uniform_alpha = 1.
    uniform_Beta = 1.


class F115(CauchyNoisyProblem, _Step_Ellipsoidal):
    cauchy_alpha = 1.
    cauchy_p = 0.2


class _Rosenbrock(Problem):
    """
    Abstract Rosenbrock (no rotation)
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        shift *= 0.75  # [-3, 3]
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = max(1., self.dim ** 0.5 / 8.) * (x - self.shift) * self.shrink + 1
        return np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=-1) + self.bias + self.boundaryHandling(x * self.shrink)


class F8(_Rosenbrock):
    def boundaryHandling(self, x):
        return 0.


class F104(GaussNoisyProblem, _Rosenbrock):
    gauss_beta = 0.01


class F105(UniformNoisyProblem, _Rosenbrock):
    uniform_alpha = 0.01
    uniform_beta = 0.01


class F106(CauchyNoisyProblem, _Rosenbrock):
    cauchy_alpha = 0.01
    cauchy_p = 0.05


class F110(GaussNoisyProblem, _Rosenbrock):
    gauss_beta = 1.


class F111(UniformNoisyProblem, _Rosenbrock):
    uniform_alpha = 1.
    uniform_beta = 1.


class F112(CauchyNoisyProblem, _Rosenbrock):
    cauchy_alpha = 1.
    cauchy_p = 0.2


class F9(Problem):
    """
    Rosenbrock
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        scale = max(1., dim ** 0.5 / 8.)
        self.linearTF = scale * rotate
        shift = np.matmul(0.5 * np.ones(dim), self.linearTF) / (scale ** 2)
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = np.matmul(x, self.linearTF.T) + 0.5
        return np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=-1) + self.bias


class _Ellipsoidal(Problem):
    """
    Abstract Ellipsoidal
    """
    condition = None

    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = osc_transform(z)  # 跟Ellipsoidal的唯一区别
        i = np.arange(nx)
        return np.sum((self.condition ** (i / (nx - 1))) * (z ** 2), -1) + self.bias + self.boundaryHandling(x * self.shrink)


class F10(_Ellipsoidal):
    condition = 1e6
    def boundaryHandling(self, x):
        return 0.


class F116(GaussNoisyProblem, _Ellipsoidal):
    condition = 1e4
    gauss_beta = 1.


class F117(UniformNoisyProblem, _Ellipsoidal):
    condition = 1e4
    uniform_alpha = 1.
    uniform_beta = 1.


class F118(CauchyNoisyProblem, _Ellipsoidal):
    condition = 1e4
    cauchy_alpha = 1.
    cauchy_p = 0.2


class F11(Problem):
    """
    Discus
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = osc_transform(z)  # 跟class Discus的唯一区别
        return np.power(10, 6) * (z[:, 0] ** 2) + np.sum(z[:, 1:] ** 2, -1) + self.bias


class F12(Problem):
    """
    Bent_cigar
    """
    beta = 0.5

    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = asy_transform(z, beta=self.beta)
        z = np.matmul(z, self.rotate)
        return z[:, 0] ** 2 + np.sum(np.power(10, 6) * (z[:, 1:] ** 2), -1) + self.bias


class F13(Problem):
    """
    Sharp Ridge
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        scales = (10 ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(scales)), rotate)
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        return z[:, 0] ** 2. + 100. * np.sqrt(np.sum(z[:, 1:] ** 2., axis=-1)) + self.bias


class _Dif_powers(Problem):
    """
    Abstract Different Powers
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        i = np.arange(self.dim)
        return np.power(np.sum(np.power(np.fabs(z), 2 + 4 * i / max(1, self.dim - 1)), -1), 0.5) + self.bias + self.boundaryHandling(x * self.shrink)


class F14(_Dif_powers):
    def boundaryHandling(self, x):
        return 0.


class F119(GaussNoisyProblem, _Dif_powers):
    gauss_beta = 1.


class F120(UniformNoisyProblem, _Dif_powers):
    uniform_alpha = 1.
    uniform_beta = 1.


class F121(CauchyNoisyProblem, _Dif_powers):
    cauchy_alpha = 1.
    cauchy_p = 0.2


class F15(Problem):
    """
    Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5. / 100
        scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF = np.matmul(np.matmul(rotate, np.diag(scales)), rotate_gen(dim))
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = asy_transform(osc_transform(z), beta=0.2)
        z = np.matmul(z, self.linearTF.T)
        return 10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + self.bias


class F16(Problem):
    """
    Weierstrass
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        scales = (0.01 ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF = np.matmul(np.matmul(rotate, np.diag(scales)), rotate_gen(dim))
        self.aK = 0.5 ** np.arange(12)
        self.bK = 3.0 ** np.arange(12)
        self.f0 = np.sum(self.aK * np.cos(np.pi * self.bK))
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = np.matmul(osc_transform(z), self.linearTF.T)
        return 10 * np.power(np.mean(np.sum(self.aK * np.cos(np.matmul(2 * np.pi * (z[:, :, None] + 0.5), self.bK[None, :])), axis=-1), axis=-1) - self.f0, 3) + \
               10 / self.dim * pen_func(x * self.shrink) + self.bias


class _Scaffer(Problem):
    """
    Abstract Scaffer
    """
    condition = None  # need to be defined in subclass

    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        scales = (self.condition ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF = np.matmul(np.diag(scales), rotate_gen(dim))
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate, self.shrink)
        z = np.matmul(asy_transform(z, beta=0.5), self.linearTF.T)
        s = np.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)
        return np.power(1 / (self.dim - 1) * np.sum(np.sqrt(s) * (np.power(np.sin(50 * np.power(s, 0.2)), 2) + 1), axis=-1), 2) + \
               self.boundaryHandling(x * self.shrink) + self.bias


class F17(_Scaffer):
    condition = 10.
    def boundaryHandling(self, x):
        return 10 * pen_func(x)


class F18(_Scaffer):
    condition = 1000.
    def boundaryHandling(self, x):
        return 10 * pen_func(x)


class F122(GaussNoisyProblem, _Scaffer):
    condition = 10.
    gauss_beta = 1.


class F123(UniformNoisyProblem, _Scaffer):
    condition = 10.
    uniform_alpha = 1.
    uniform_beta = 1.


class F124(CauchyNoisyProblem, _Scaffer):
    condition = 10.
    cauchy_alpha = 1.
    cauchy_p = 0.2


class _Composite_Grie_rosen(Problem):
    """
    Abstract Composite_Grie_rosen
    """
    factor = None

    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        scale = max(1., dim ** 0.5 / 8.)
        self.linearTF = scale * rotate
        shift = np.matmul(0.5 * np.ones(dim) / (scale ** 2.), self.linearTF)
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = np.matmul(x, self.linearTF.T) + 0.5
        s = 100. * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (1. - z[:, :-1]) ** 2
        return self.factor + self.factor * np.sum(s / 4000. - np.cos(s), axis=-1) / (self.dim - 1.) + self.bias + self.boundaryHandling(x * self.shrink)


class F19(_Composite_Grie_rosen):
    factor = 10.
    def boundaryHandling(self, x):
        return 0.


class F125(GaussNoisyProblem, _Composite_Grie_rosen):
    factor = 1.
    gauss_beta = 1.


class F126(UniformNoisyProblem, _Composite_Grie_rosen):
    factor = 1.
    uniform_alpha = 1.
    uniform_beta = 1.


class F127(CauchyNoisyProblem, _Composite_Grie_rosen):
    factor = 1.
    cauchy_alpha = 1.
    cauchy_p = 0.2


class F20(Problem):
    """
    Schwefel
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5. / 100
        shift = 0.5 * 4.2096874633 * np.random.choice([-1., 1.], size=dim) / self.shrink
        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        x *= self.shrink
        tmp = 2 * np.abs(self.shift)
        scales = (10 ** 0.5) ** np.linspace(0, 1, self.dim)
        z = 2 * np.sign(self.shift) * x
        z[:, 1:] += 0.25 * (z[:, :-1] - tmp[:-1])
        z = 100. * (scales * (z - tmp) + tmp)
        b = 4.189828872724339
        return b - 0.01 * np.mean(z * np.sin(np.sqrt(np.abs(z))), axis=-1) + 100 * pen_func(z / 100) + self.bias


class _Gallagher(Problem):
    """
    Abstract Gallagher
    """
    n_peaks = None

    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100

        # problem param config
        if self.n_peaks == 101:  # F21
            opt_shrink = 1.  # shrink of global & local optima
            global_opt_alpha = 1e3
        elif self.n_peaks == 21:  # F22
            opt_shrink = 0.98  # shrink of global & local optima
            global_opt_alpha = 1e6
        else:
            raise ValueError(f'{self.n_peaks} peaks Gallagher is not supported yet.')

        # generate global & local optima y[i], i=0,1,...,n_peaks-1
        # self.y = opt_shrink * (np.random.rand(self.n_peaks, dim) * (self.ub - self.lb) + self.lb)  # [n_peaks, dim]
        self.y = opt_shrink * (np.random.rand(self.n_peaks, dim) * 200 - 100)  # [n_peaks, dim]
        self.y[0] = shift * opt_shrink * 0.8  # the global optimum
        shift = self.y[0]

        # generate the matrix C[i], i=0,1,...,n_peaks-1
        sqrt_alpha = 1000 ** np.random.permutation(np.linspace(0, 1, self.n_peaks - 1))
        sqrt_alpha = np.insert(sqrt_alpha, obj=0, values=np.sqrt(global_opt_alpha))
        self.C = [np.random.permutation(sqrt_alpha[i] ** np.linspace(-0.5, 0.5, dim)) for i in range(self.n_peaks)]
        self.C = np.vstack(self.C)  # [n_peaks, dim]

        # generate the weight w[i], i=0,1,...,n_peaks-1
        self.w = np.insert(np.linspace(1.1, 9.1, self.n_peaks - 1), 0, 10.)  # [n_peaks]

        Problem.__init__(self, dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        z = np.matmul(np.expand_dims(x, axis=1).repeat(self.n_peaks, axis=1) - self.y, self.rotate.T)  # [NP, n_peaks, dim]
        z = np.max(self.w * np.exp((-0.5 / self.dim) * np.sum(self.C * (z ** 2), axis=-1)), axis=-1)  # [NP]
        return osc_transform(10 - z) ** 2 + self.bias + self.boundaryHandling(x * self.shrink)


class F21(_Gallagher):
    n_peaks = 101
    def boundaryHandling(self, x):
        return pen_func(x)


class F22(_Gallagher):
    n_peaks = 21
    def boundaryHandling(self, x):
        return pen_func(x)


class F128(GaussNoisyProblem, _Gallagher):
    n_peaks = 101
    gauss_beta = 1.


class F129(UniformNoisyProblem, _Gallagher):
    n_peaks = 101
    unifrom_alpha = 1.
    unifrom_beta = 1.


class F130(CauchyNoisyProblem, _Gallagher):
    n_peaks = 101
    cauchy_alpha = 1.
    cauchy_p = 0.2


class F23(Problem):
    """
    Katsuura
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        scales = (100. ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(scales)), rotate)
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
        return res * tmp - tmp + pen_func(x * self.shrink) + self.bias


class F24(Problem):
    """
    Lunacek_bi_Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias):
        self.shrink = 5 / 100
        self.mu0 = 2.5
        shift = np.random.choice([-1., 1.], size=dim) * self.mu0 / 2 / self.shrink
        scales = (100 ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(scales)), rotate)
        super().__init__(dim, shift, rotate, bias)

    def func(self, x):
        self.FES += x.shape[0]
        # shift = self.shift * self.shrink
        x *= self.shrink
        x_hat = 2. * np.sign(self.shift) * x
        z = np.matmul(x_hat - self.mu0, self.rotate)
        s = 1. - 1. / (2. * np.sqrt(self.dim + 20.) - 8.2)
        mu1 = -np.sqrt((self.mu0 ** 2 - 1) / s)
        return np.minimum(np.sum((x_hat - self.mu0) ** 2., axis=-1), self.dim + s * np.sum((x_hat - mu1) ** 2., axis=-1)) + \
               10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + 1e4 * pen_func(x) + self.bias


class bbob_Dataset(Dataset):
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
                     train_batch_size=1,
                     test_batch_size=1,
                     shifted=True,
                     rotated=True,
                     biased=True,
                     train_set_ratio=0.75,
                     dataset_seed=1035,
                     instance_seed=3849):
        # get functions ID of indicated suit
        if suit == 'bbob':
            func_id = [i for i in range(1, 25)]     # [1, 24]
        elif suit == 'bbob-noisy':
            func_id = [i for i in range(101, 131)]  # [101, 130]
        else:
            raise ValueError
        # get problem instances
        if instance_seed > 0:
            np.random.seed(instance_seed)
        data = []
        for id in func_id:
            if shifted:
                shift = np.random.random(dim) * 160 - 80  # [-80, 80]
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
            data.append(eval(f'F{id}')(dim, shift, H, bias))
        # get train set and test set
        if dataset_seed > 0:
            np.random.seed(dataset_seed)
        else:
            np.random.seed(None)
        np.random.shuffle(data)
        n_train_func = int(len(data) * train_set_ratio)
        return bbob_Dataset(data[:n_train_func], train_batch_size), bbob_Dataset(data[n_train_func:], test_batch_size)

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


train_set, test_set = bbob_Dataset.get_datasets('bbob', 10, instance_seed=0, shifted=True, rotated=True, biased=True)
for i in range(len(train_set)):
    print(train_set[i][0].__class__, '  ', train_set[i][0].optimum)
print('')
for i in range(len(test_set)):
    print(test_set[i][0].__class__, '  ', test_set[i][0].optimum)

