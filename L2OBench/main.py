import torch
from trainer import Trainer
from tester import *
from config import get_config
from logger import *

from optimizer.random_search import Random_search
import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    config = get_config()
    assert ((config.train is not None) ^ (config.test is not None)  ^ (config.rollout is not None)) and not (config.train and config.test and config.rollout), 'Between train&test&rollout, only one mode can be given in one time'
    # Trainer
    if config.train:
        torch.set_grad_enabled(True)
        trainer = Trainer(config)
        trainer.train()

    # Tester
    if config.test:
        torch.set_grad_enabled(False)
        tester = Tester(config)
        tester.test()
        post_processing_test_statics(config.test_log_dir)

    # Rollout
    if config.rollout:
        torch.set_grad_enabled(False)
        rollout(config)
        post_processing_rollout_statics(config.rollout_log_dir)
