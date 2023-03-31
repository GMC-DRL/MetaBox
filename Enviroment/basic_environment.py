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
