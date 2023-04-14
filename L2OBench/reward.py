"""
Argument **cost contains 5 elements:
    'cur':          shape[NP];  cost of population at current generation
    'pre':          shape[NP];  cost of population at previous generation
    'init':         shape[NP];  cost of population at first generation randomly generated
    'cur_gbest':    scalar;     cost of the gbest at current generation
    'pre_gbest':    scalar;     cost of the gbest at previous generation
"""


def binary(**cost):
    return 1 if cost['cur_gbest'] < cost['pre_gbest'] else -1


def relative(**cost):
    return (cost['pre_gbest'] - cost['cur_gbest']) / cost['pre_gbest']


def direct(**cost):
    return (cost['pre_gbest'] - cost['cur_gbest']) / cost['init'].min()


def triangle(**cost):
    cur_p = (cost['init'].min() - cost['cur_gbest']) / cost['init'].min()
    pre_p = (cost['init'].min() - cost['pre_gbest']) / cost['init'].min()
    return 0.5 * (cur_p ** 2 - pre_p ** 2)
