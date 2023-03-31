


class deap_optimizer():
    def __init__(self):
        pass

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch

class learnable_Agent():
    def __init__(self,superparam,vector_env):
        self.superparam = superparam
        self.env = vector_env
        # self.net = xxx
        pass

    def get_feature(self):

        pass


    def inference(self,need_gd):
        # get aciton/fitness
        pass


    def cal_loss(self):
        pass

    def update_env(self):
        # self.env.step()
        pass

    def learning(self):
        # cal_loss
        # update nets
        pass


    def memory(self):
        # record some info
        pass
