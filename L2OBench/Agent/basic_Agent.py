"""
This is a basic agent class for L2O benchmark.
All agents should inherit from this class.
Your own agent should have the following functions:
    1. __init__(self, problem, config) to initialize the agent
    2.get_feature(self,env) to get the feature from env.state to feed net
    3.inference(self,env,need_train) to get action from net
    4.cal_loss(self,env) to calculate the loss
    5.learn(self,env) to update the net
You can use the class Memory to record some info which should be initialized in __init__ and you can use memory to get the info.
"""



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
    def __init__(self,config):
        self.config = config
        # self.net should be a tuple/list/dict of several nets
        self.nets = None

        self.memory = Memory()

        pass

    def get_feature(self,env):
        # get feature from env.state to feed net
        pass


    def inference(self,env,need_gd):
        # get_feature
        # use feature to get aciton
        pass


    def cal_loss(self,env):

        pass


    def learning(self):
        # select optimizer(Adam,SGD...)
        # cal_loss
        # update nets

        pass


    def memory(self):
        # record some info
        return self.memory
        pass
