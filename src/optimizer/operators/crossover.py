import numpy as np
from typing import Union
import copy


def binomial(x: np.ndarray, v: np.ndarray, Cr: Union[np.ndarray, float]) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
        v = v.reshape(1, -1)
    NP, dim = x.shape
    jrand = np.random.randint(dim, size=NP)
    if isinstance(Cr, np.ndarray) and Cr.ndim == 1:
        Cr = Cr.reshape(-1, 1)
    u = np.where(np.random.rand(NP, dim) < Cr, v, x)
    u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
    if u.shape[0] == 1:
        u = u.squeeze(axis=0)
    return u


def exponential(x: np.ndarray, v: np.ndarray, Cr: Union[np.ndarray, float]) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
        v = v.reshape(1, -1)
    NP, dim = x.shape
    u = copy.deepcopy(x)
    L = np.random.randint(dim, size=(NP, 1)).repeat(dim).reshape(NP, dim)
    R = np.ones(NP) * dim
    rvs = np.random.rand(NP, dim)
    i = np.arange(dim).repeat(NP).reshape(dim, NP).transpose()
    if isinstance(Cr, np.ndarray) and Cr.ndim == 1:
        Cr = Cr.reshape(-1, 1)
    rvs[rvs > Cr] = np.inf
    rvs[i <= L] = -np.inf
    k = np.where(rvs == np.inf)
    ki = np.stack(k).transpose()
    if ki.shape[0] > 0:
        k_ = np.concatenate((ki, ki[None, -1] + 1), 0)
        _k = np.concatenate((ki[None, 0] - 1, ki), 0)
        ind = ki[(k_[:, 0] != _k[:, 0]).reshape(-1, 1).repeat(2).reshape(-1, 2)[:-1]].reshape(-1, 2).transpose()
        R[ind[0]] = ind[1]
    R = R.repeat(dim).reshape(NP, dim)
    u[(i >= L) * (i < R)] = v[(i >= L) * (i < R)]
    if u.shape[0] == 1:
        u = u.squeeze(axis=0)
    return u
