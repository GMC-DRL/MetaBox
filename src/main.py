import torch
from trainer import Trainer
from tester import *
from config import get_config
from logger import *
import shutil
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    config = get_config()
    assert ((config.train is not None) +
            (config.rollout is not None) +
            (config.test is not None) +
            (config.run_experiment is not None) +
            (config.mgd_test is not None) +
            (config.mte_test is not None)) == 1, \
        'Among train, rollout, test, run_experiment, mgd_test & mte_test, only one mode can be given at one time.'

    # train
    if config.train:
        torch.set_grad_enabled(True)
        trainer = Trainer(config)
        trainer.train()

    # rollout
    if config.rollout:
        torch.set_grad_enabled(False)
        rollout(config)
        post_processing_rollout_statics(config.rollout_log_dir, Logger(config))

    # test
    if config.test:
        torch.set_grad_enabled(False)
        tester = Tester(config)
        tester.test()
        post_processing_test_statics(config.test_log_dir, Logger(config))

    # run_experiment
    if config.run_experiment:
        # train
        torch.set_grad_enabled(True)
        trainer = Trainer(config)
        trainer.train()

        # rollout
        agent_save_dir = config.agent_save_dir  # user defined agent_save_dir + agent name + run_time
        rollout_save_dir = os.path.join(agent_save_dir, config.train_agent)  # user defined agent_save_dir + agent name + run_time + agent_name
        if not os.path.exists(rollout_save_dir):
            os.makedirs(rollout_save_dir)
        # copy models from agent_save_dir to rollout_save_dir
        for filename in os.listdir(agent_save_dir):
            if os.path.isfile(os.path.join(agent_save_dir, filename)):
                shutil.copy(os.path.join(agent_save_dir, filename), rollout_save_dir)
        config.agent_load_dir = agent_save_dir  # let config.agent_load_dir = config.agent_save_dir to load model
        config.agent_for_rollout = [config.train_agent]
        config.optimizer_for_rollout = [config.train_optimizer]
        torch.set_grad_enabled(False)
        rollout(config)
        shutil.rmtree(rollout_save_dir)  # remove rollout model files after rollout
        post_processing_rollout_statics(config.rollout_log_dir, Logger(config))

        # test
        test_model_file = agent_save_dir + config.train_agent + '.pkl'
        shutil.copy(os.path.join(agent_save_dir, 'checkpoint20.pkl'), test_model_file)  # copy checkpoint20.pkl to agent_name.pkl
        config.agent_for_cp.append(config.train_agent)
        config.l_optimizer_for_cp.append(config.train_optimizer)
        torch.set_grad_enabled(False)
        tester = Tester(config)
        tester.test()
        os.remove(test_model_file)  # remove test model files after test
        post_processing_test_statics(config.test_log_dir, Logger(config))

    # mgd_test
    if config.mgd_test:
        torch.set_grad_enabled(False)
        mgd_test(config)

    # mte_test
    if config.mte_test:
        mte_test(config)
