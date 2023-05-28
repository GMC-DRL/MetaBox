import numpy as np
from typing import Union, Iterable


def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)


def random(x: Union[np.ndarray, Iterable],
           lb: Union[np.ndarray, Iterable, int, float],
           ub: Union[np.ndarray, Iterable, int, float]
           ) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(ub, np.ndarray):
        ub = np.array(ub)
    cro_bnd = (x < lb) | (x > ub)
    return ~cro_bnd * x + cro_bnd * (np.random.rand(*x.shape) * (ub - lb) + lb)


def reflection(x: Union[np.ndarray, Iterable],
               lb: Union[np.ndarray, Iterable, int, float],
               ub: Union[np.ndarray, Iterable, int, float]
               ) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    cro_lb = x < lb
    cro_ub = x > ub
    no_cro = ~(cro_lb | cro_ub)
    return no_cro * x + cro_lb * (2 * lb - x) + cro_ub * (2 * ub - x)


def periodic(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float],
             ub: Union[np.ndarray, Iterable, int, float]
             ) -> np.ndarray:
    if not isinstance(ub, np.ndarray):
        ub = np.array(ub)
    return (x - ub) % (ub - lb) + lb


def halving(x: Union[np.ndarray, Iterable],
            lb: Union[np.ndarray, Iterable, int, float],
            ub: Union[np.ndarray, Iterable, int, float]
            ) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    cro_lb = x < lb
    cro_ub = x > ub
    no_cro = ~(cro_lb | cro_ub)
    return no_cro * x + cro_lb * (x + lb) / 2 + cro_ub * (x + ub) / 2


def parent(x: Union[np.ndarray, Iterable],
           lb: Union[np.ndarray, Iterable, int, float],
           ub: Union[np.ndarray, Iterable, int, float],
           par: Union[np.ndarray, Iterable]
           ) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(par, np.ndarray):
        par = np.array(par)
    cro_lb = x < lb
    cro_ub = x > ub
    no_cro = ~(cro_lb | cro_ub)
    return no_cro * x + cro_lb * (par + lb) / 2 + cro_ub * (par + ub) / 2
