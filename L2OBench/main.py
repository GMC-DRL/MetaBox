import utils
from trainer import Trainer
from tester import Tester
from config import get_config
from logger import *
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    config = get_config()
    # Trainer
    trainer = Trainer(config)
    trainer.train()
    #
    # Tester
    # tester = Tester(config)
    # tester.test()
    # post_processing_test_statics(config.test_log_dir)
