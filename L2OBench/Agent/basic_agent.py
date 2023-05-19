"""
This is a basic agent class for L2O benchmark.
All agents should inherit from this class.
Your own agent should have the following functions:
    1. __init__(self, config) : to initialize the agent
    2. train_episode(self, env, epoch_id, problem_id, logger) : to train the agent for an episode.
    3. rollout_episode(self, env, epoch_id, problem_id, logger) : to rollout the agent for an episode.
You can use the class Memory to record some info which should be initialized in __init__ and you can use memory to get the info.
"""

from environment import PBO_Env
from logger import Logger
from typing import Tuple


class Basic_Agent:
    def __init__(self, config):
        self.__config = config

    def train_episode(self,
                      env: PBO_Env,
                      epoch_id: int = None,
                      logger: Logger = None) -> Tuple[bool, dict]:
        raise NotImplementedError

    def rollout_episode(self,
                        env: PBO_Env,
                        epoch_id: int = None,
                        logger: Logger = None) -> dict:
        raise NotImplementedError
