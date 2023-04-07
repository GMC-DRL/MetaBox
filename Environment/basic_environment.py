class Env:
    """
    Treated as a pure problem for traditional algorithms without any agent.
    """
    def __init__(self, problem):
        self.problem = problem


class PBO_Env(Env):
    """
    Env with problem and optimizer.
    """
    def __init__(self,
                 problem,
                 optimizer,
                 reward_func):
        Env.__init__(self, problem)
        self.optimizer = optimizer
        self.reward_func = reward_func

    def reset(self):
        self.optimizer.init_population(self.problem)

    def step(self, action):
        parent_cost = self.optimizer.cost.copy()
        self.optimizer.evolve(self.problem, action)

        state = {'population': self.optimizer.population,
                 'cost': self.optimizer.cost,
                 'fes': self.optimizer.fes}
        reward = self.reward_func(cur=self.optimizer.cost, parent=parent_cost, init=self.optimizer.init_cost)
        is_done = self.optimizer.fes >= self.optimizer.maxFEs or self.optimizer.cost.min() <= 1e-8
        return state, reward, is_done
