class Env:
    """
    Treated as a pure problem for traditional algorithms without any agent.
    """
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance

    def reset(self):
        pass

    def step(self, population):
        return self.problem_instance.func(population)


class PBO_Env(Env):
    def __init__(self,
                 problem_instance,
                 dim,
                 lower_bound,
                 upper_bound,
                 population_size,
                 FEs):
        Env.__init__(self, problem_instance)
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.FEs = FEs

    def reset(self):
        pass

    def step(self, action):
        pass

    def get_feature(self):
        pass

    def get_reward(self):
        pass
