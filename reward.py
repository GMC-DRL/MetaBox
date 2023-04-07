"""
Argument **cost contains 3 elements:
    'cur':      cost of current generation,
    'parent':   cost of parent generation,
    'init':     cost of first generation generated randomly.
"""


def binary(**cost):
    return 1 if cost['cur'].min() < cost['parent'].min() else -1


def relative_to_parent(**cost):
    return (cost['parent'].min() - cost['cur'].min()) / cost['parent'].min()


def relative_to_init(**cost):
    return (cost['parent'].min() - cost['cur'].min()) / cost['init'].min()


def triangle(**cost):
    cur_p = (cost['init'] - cost['cur']) / cost['init']
    parent_p = (cost['init'] - cost['parent']) / cost['init']
    return 0.5 * (cur_p ** 2 - parent_p ** 2)
