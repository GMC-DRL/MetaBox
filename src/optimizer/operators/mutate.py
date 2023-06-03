import numpy as np
from typing import Union


def generate_random_int_single(NP: int, cols: int, pointer: int) -> np.ndarray:
    r = np.random.randint(low=0, high=NP, size=cols)
    while pointer in r:
        r = np.random.randint(low=0, high=NP, size=cols)
    return r


def generate_random_int(NP: int, cols: int) -> np.ndarray:
    """
    Generate a matrix of random integers used by mutation.

    :param NP: Population size.
    :param cols: The number of random integers generated for each individual.
    :return: A random integers matrix in shape[''NP'', ''cols''], and elements are in a range of [0, ''NP''-1].
             The ''cols'' elements at dimension[1] are different from each other.
    """
    r = np.random.randint(low=0, high=NP, size=(NP, cols))
    # validity checking and modification for r
    for col in range(0, cols):
        while True:
            is_repeated = [np.equal(r[:, col], r[:, i]) for i in range(col)]
            is_repeated.append(np.equal(r[:, col], np.arange(NP)))
            repeated_index = np.nonzero(np.any(np.stack(is_repeated), axis=0))[0]
            repeated_sum = repeated_index.size
            if repeated_sum != 0:
                r[repeated_index[:], col] = np.random.randint(low=0, high=NP, size=repeated_sum)
            else:
                break
    return r


def rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer)
    return x[r[0]] + F * (x[r[1]] - x[r[2]])


def rand_1(x: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 3)
    return x[r[:, 0]] + F * (x[r[:, 1]] - x[r[:, 2]])


def rand_2_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 5, pointer)
    return x[r[0]] + F * (x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]])


def rand_2(x: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 5)
    return x[r[:, 0]] + F * (x[r[:, 1]] - x[r[:, 2]] + x[r[:, 3]] - x[r[:, 4]])


def best_1_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 2, pointer)
    return best + F * (x[r[0]] - x[r[1]])


def best_1(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 2)
    return best + F * (x[r[:, 0]] - x[r[:, 1]])


def best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 4, pointer)
    return best + F * (x[r[0]] - x[r[1]] + x[r[2]] - x[r[3]])


def best_2(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 4)
    return best + F * (x[r[:, 0]] - x[r[:, 1]] + x[r[:, 2]] - x[r[:, 3]])


def rand_to_best_1_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer)
    return x[r[0]] + F * (best - x[r[0]] + x[r[1]] - x[r[2]])


def rand_to_best_1(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 3)
    return x[r[:, 0]] + F * (best - x[r[:, 0]] + x[r[:, 1]] - x[r[:, 2]])


def rand_to_best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 5, pointer)
    return x[r[0]] + F * (best - x[r[0]] + x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]])


def rand_to_best_2(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 5)
    return x[r[:, 0]] + F * (best - x[r[:, 0]] + x[r[:, 1]] - x[r[:, 2]] + x[r[:, 3]] - x[r[:, 4]])


def cur_to_best_1_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 2, pointer)
    return x[pointer] + F * (best - x[pointer] + x[r[0]] - x[r[1]])


def cur_to_best_1(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 2)
    return x + F * (best - x + x[r[:, 0]] - x[r[:, 1]])


def cur_to_best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 4, pointer)
    return x[pointer] + F * (best - x[pointer] + x[r[0]] - x[r[1]] + x[r[2]] - x[r[3]])


def cur_to_best_2(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 4)
    return x + F * (best - x + x[r[:, 0]] - x[r[:, 1]] + x[r[:, 2]] - x[r[:, 3]])


def cur_to_rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer)
    return x[pointer] + F * (x[r[0]] - x[pointer] + x[r[1]] - x[r[2]])


def cur_to_rand_1(x: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 3)
    return x + F * (x[r[:, 0]] - x + x[r[:, 1]] - x[r[:, 2]])


def cur_to_rand_2_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None) -> np.ndarray:
    if r is None:
        r = generate_random_int_single(x.shape[0], 5, pointer)
    return x[pointer] + F * (x[r[0]] - x[pointer] + x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]])


def cur_to_rand_2(x: np.ndarray, F: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 5)
    return x + F * (x[r[:, 0]] - x + x[r[:, 1]] - x[r[:, 2]] - x[r[:, 3]] + x[r[:, 4]])
