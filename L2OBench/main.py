import torch
from trainer import Trainer
from tester import Tester
from config import get_config
from logger import *
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    config = get_config()
    assert (config.train is None) ^ (config.test is None), 'Between train&test, only one mode can be given in one time'
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
