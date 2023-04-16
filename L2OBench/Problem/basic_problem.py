import numpy as np


class Basic_Problem:
    """
    Abstract super class for problems and applications.
    """

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
        raise NotImplementedError
