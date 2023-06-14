# MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

This is a **reinforcement learning benchmark platform** for benchmarking and MetaBBO-RL methods. You can develop your own MetaBBO-RL approach and complare it with baseline approaches built-in following the **Train-Test-Log** philosophy automated by MetaBox.

Post-processed data are provided in [content.md](post_processed_data/content.md).

## Contents

* [Requirements](#Requirements)

* [Quick Start](#Quick-Start)

* [Overview](#Overview)

* [Datasets](#Datasets)

* [Baseline Library](#Baseline-Library)

* [Fit your own MetaBBO-RL into MetaBox](#Fit-your-own-MetaBBO-RL-into-MetaBox)

* [Run Experiment](#run-experiment)

* [Training](#Training)
  * [How to Train](#How-to-Train)
  * [Train Results](#Train-Results)
  
* [Rollout](#Rollout)
  * [How to Rollout](#How-to-Rollout)
  * [Rollout Results](#Rollout-Results)
  
* [Testing](#Testing)
  * [How to Test](#How-to-Test)
  * [Test Results](#Test-Results)
  * [MGD Test](#MGD-Test)
  * [MTE Test](#MTE-Test)

## Requirements

`Python` >=3.7.1 with following packages installed:  

* `numpy`==1.21.2  
* `torch`==1.9.0  
* `matplotlib`==3.4.3  
* `pandas`==1.3.3  
* `scipy`==1.7.1
* `scikit_optimize`==0.9.0  
* `deap`==1.3.3  
* `tqdm`==4.62.3  
* `openpyxl`==3.1.2

## Quick Start

* To obtain the figures in our paper, run the following commands: 

  ```shell
  cd for_review
  python paper_experiment.py
  ```
  then corresponding figures will be output to `for_revivew/pics`.
  
  ---
  
  The quick usage of four main running interfaces are listed as follows, in the following command we specifically take `RLEPSO` as example.
  
  Firstly, get into main code folder src:
  ```shell
  cd ../src
  ```
  
* To trigger the entire workflow including **train, rollout and test**, run the following command:

  ```shell
  python main.py --run_experiment --problem bbob --difficulty easy --train_agent RLEPSO_Agent --train_optimizer RLEPSO_Optimizer
  ```

* To trigger the standalone process of **training**:

  ```shell
  python main.py --train --problem bbob --difficulty easy --train_agent RLEPSO_Agent --train_optimizer RLEPSO_Optimizer 
  ```

* To trigger the standalone process of **rollout**:

  ```shell
  python main.py --rollout --problem bbob --difficulty easy --agent_load_dir agent_model/rollout/bbob_easy/ --agent_for_rollout RLEPSO_Agent --optimizer_for_rollout RLEPSO_Optimizer
  ```

* To trigger the standalone process of **testing**:

  ```shell
  python main.py --test --problem bbob --difficulty easy --agent_load_dir agent_model/test/bbob_easy/ --agent_for_cp RLEPSO_Agent --l_optimizer_for_cp RLEPSO_Optimizer --t_optimizer_for_cp DEAP_CMAES Random_search
  ```

## Overview

![overview](docs/overview.png)

`MetaBox` can be divided into six modules: **Template, Test suites, Baseline Library, Trainer, Tester and Logger.**

* `Template` comprises two main components: the **meta-level RL agent** and the **lower-level optimizer**, which provides a unified interface protocol for users to develop their own MetaBBO-RL with ease. 
* `Test suites` are used for generating training and testing sets, including **Synthetic**, **Noisy-Synthetic**, and **Protein-Docking**.
* `Baseline Library` comprises proposed algorithms including **MetaBBO-RL**, **MetaBBO-SL** and **classic** optimizers that we implemented for comparison study.
* `Trainer` **manages the entire learning process** of the agent by building environments consisting of a backbone optimizer and a problem sampled from train set and letting the agent interact with environments sequentially.
* `Tester` is used to **evaluate** the optimization performance of the MetaBBO-RL. By using the test set to test the baselines and the trained MetaBBO agent, it produces test log for logger to generate statistic test results.
* `Logger` implements multiple interfaces for **displaying** the logs of the training process and the results of the testing process, which facilitates the improvement of the training process and the observation of MetaBBO-RL's performance.

**Data Stream**
![datastream](docs/datastream.png)

When using MetaBox, after the training interface `Trainer.train()` called, 21 pickle files of training agent in different training process and pictures of training process will be output to `src/agent_model/train` and `src/output/train` respectively. If you want to compare performance among baselines built-in or your own approach, `Tester.test` is needed to be called. For those learnable agent in your comparing list, you need to first collect these agent model pickle files (one agent one file) to a specific folder, where the agent model file may have been outputed by `Trainer.train()` and you can find that and then copy it to the right place. When `Tester.test()` finished, tables containing per-instance result and algorithm complexity, pictures depicting comparison results or singal approach performance, original testing result will be output to `src/output/test`. In addition, for `Rollout` interface, before that you need to collect all of checkpoints of all of learning agents which can be copy from output of `Trainer.train()`. When `Rollout` finished, pictures containing average return process and optimization cost process will be output to `src/output/rollout` and the original data file will also be saved.

## Datasets


Currently, three benchmark suites are included:  

* `Synthetic` containing 24 noiseless functions, borrowed from [coco](https://github.com/numbbo/coco):bbob with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Noisy-Synthetic` containing 30 noisy functions, borrowed from [coco](https://github.com/numbbo/coco):bbob-noisy with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Protein-Docking` containing 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem, borrowed from [LOIS](https://github.com/Shen-Lab/LOIS) with [original paper](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms).

By setting the argument `--problem` to `bbob`, `bbob-noisy` or `protein` in command line to use the corresponding suite, for example:

```bash
python main.py --train --problem protein --train_agent MyAgent --train_optimizer MyOptimizer
```

For the usage of  `--train`  `--train_agent`  `--train_optimizer`, see [Training](#Training) for more details.

Each test suites are regarded as a dataset, which is split into training set and test set in different proportions with respect to two difficulty levels:  

* `easy` training set accounts for 75% and test set accounts for 25%.
* `difficult` training set accounts for 25% and test set accounts for 75%.

By setting the argument `--difficulty` to `easy` or `difficult` in command line to specify the difficulty level like the following command. Note that `easy` difficulty is used by default.

```bash
python main.py --train --problem bbob --difficulty difficult --train_agent MyAgent --train_optimizer MyOptimizer
```

## Baseline Library

**7 MetaBBO-RL optimizers, 1 MetaBBO-SL optimizer and 11 classic optimizers have been integrated into this platform.** Choose one or more of them to be the baseline(s) to test the performance of your own optimizer.

**Supported MetaBBO-RL optimizers**:

|   Name   | Year |                        Related paper                         |
| :------: | :--: | :----------------------------------------------------------: |
| DE-DDQN  | 2019 | [Deep reinforcement learning based parameter control in differential evolution](https://dl.acm.org/doi/10.1145/3321707.3321813) |
|  QLPSO   | 2019 | [A reinforcement learning-based communication topology in particle swarm optimization](https://link.springer.com/article/10.1007/s00521-019-04527-9) |
|  DEDQN   | 2021 | [Differential evolution with mixed mutation strategy based on deep reinforcement learning](https://www.sciencedirect.com/science/article/pii/S1568494621005998) |
|   LDE    | 2021 | [Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient](https://ieeexplore.ieee.org/document/9359652) |
|  RL-PSO  | 2021 | [Employing reinforcement learning to enhance particle swarm optimization methods](https://www.tandfonline.com/doi/full/10.1080/0305215X.2020.1867120) |
|  RLEPSO  | 2022 | [RLEPSO:Reinforcement learning based Ensemble particle swarm optimizer✱](https://dl.acm.org/doi/abs/10.1145/3508546.3508599) |
| RL-HPSDE | 2022 | [Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning](https://www.sciencedirect.com/science/article/pii/S2210650222001602) |

**Supported MetaBBO-SL optimizer**:

| Name | Year |                        Related paper                         |
| :--: | :--: | :----------------------------------------------------------: |
| RNN-OI  | 2017 | [Learning to learn without gradient descent by gradient descent](https://dl.acm.org/doi/10.5555/3305381.3305459) |

**Supported classic optimizers**:

|         Name          | Year |                        Related paper                         |
| :-------------------: | :--: | :----------------------------------------------------------: |
|          PSO          | 1995 | [Particle swarm optimization](https://ieeexplore.ieee.org/abstract/document/488968) |
|          DE           | 1997 | [Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces](https://dl.acm.org/doi/abs/10.1023/A%3A1008202821328) |
|        CMA-ES         | 2001 | [Completely Derandomized Self-Adaptation in Evolution Strategies](https://ieeexplore.ieee.org/document/6790628) |
| Bayesian Optimization | 2014 | [Bayesian Optimization: Open source constrained global optimization tool for Python](https://github.com/bayesian-optimization/BayesianOptimization) |
|        GL-PSO         | 2015 | [Genetic Learning Particle Swarm Optimization](https://ieeexplore.ieee.org/abstract/document/7271066/) |
|       sDMS_PSO        | 2015 | [A Self-adaptive Dynamic Particle Swarm Optimizer](https://ieeexplore.ieee.org/document/7257290) |
|          j21          | 2021 | [Self-adaptive Differential Evolution Algorithm with Population Size Reduction for Single Objective Bound-Constrained Optimization: Algorithm j21](https://ieeexplore.ieee.org/document/9504782) |
|         MadDE         | 2021 | [Improving Differential Evolution through Bayesian Hyperparameter Optimization](https://ieeexplore.ieee.org/document/9504792) |
|        SAHLPSO        | 2021 | [Self-Adaptive two roles hybrid learning strategies-based particle swarm optimization](https://www.sciencedirect.com/science/article/pii/S0020025521006988) |
|     NL_SHADE_LBC      | 2022 | [NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization](https://ieeexplore.ieee.org/abstract/document/9870295) |
|     Random Search     |  -   |                              -                               |

Note that `Random Search` performs uniformly random sampling to optimize the fitness.

For running commands, use the **corresponding agent class name** and **corresponding optimizer class name** to specify the algorithms:

`MetaBBO-RL` baselines:

| Algorithm Name | Corresponding Agent Class | Corresponding Backbone Optimizer Class |
| :------------: | :-----------------------: | :------------------------------------: |
|    DE-DDQN     |       DE_DDQN_Agent       |           DE_DDQN_Optimizer            |
|     QLPSO      |        QLPSO_Agent        |            QLPSO_Optimizer             |
|     DEDQN      |        DEDQN_Agent        |            DEDQN_Optimizer             |
|      LDE       |         LDE_Agent         |             LDE_Optimizer              |
|     RL-PSO     |       RL_PSO_Agent        |            RL_PSO_Optimizer            |
|     RLEPSO     |       RLEPSO_Agent        |            RLEPSO_Optimizer            |
|    RL-HPSDE    |      RL_HPSDE_Agent       |           RL_HPSDE_Optimizer           |

`MetaBBO-SL` baseline:

| Algorithm Name | Corresponding Agent Class | Corresponding Backbone Optimizer Class |
| :------------: | :-----------------------: | :------------------------------------: |
|      RNN-OI       |         L2L_Agent         |             L2L_Optimizer              |

`classic` baselines:

|    Algorithm Name     | Corresponding Optimizer Class |
| :-------------------: | :---------------------------: |
|          PSO          |           DEAP_PSO            |
|          DE           |            DEAP_DE            |
|        CMA-ES         |          DEAP_CMAES           |
| Bayesian Optimization |       BayesianOptimizer       |
|        GL-PSO         |            GL_PSO             |
|       sDMS_PSO        |           sDMS_PSO            |
|          j21          |             JDE21             |
|         MadDE         |             MadDE             |
|        SAHLPSO        |            SAHLPSO            |
|     NL_SHADE_LBC      |         NL_SHADE_LBC          |
|     Random Search     |         Random_search         |


## Fit your own MetaBBO-RL into MetaBox

If you want to develop your own MetaBBO-RL approach, to fit into `MetaBox` running logic, you should meet with the following protocol about the `Agent` and `Optimizer`. 

`Agent` is the same definition in RL area, taking the state from `env` as input and `action` as output. But to fit into MetaBox pre-defined `Trainer` and `Tester` calling logic, `Agent` should has `train_episode` interface which will be called in `Trainer` and `rollout_episode` interface which will be called in `Tester`. 

`Optimizer` is a component of `env` in MetaBBO task. It's controlled by `Agent` and take `action` from `Agent` to perfrom corresponding change like hyper-parameters adjusting or operators selection. But to fit into `env` calling logic. Interfaces namely `init_population` and `update` is needed.

* Your agent should follow this template:

  ```python
  from agent.basic_agent import Basic_Agent
  from agent.utils import save_class
  
  class MyAgent(Basic_Agent):
      def __init__(self, config):
          """
          Parameter
          ----------
          config: An argparse. Namespace object for passing some core configurations such as max_learning_step.
  
          Must To Do
          ----------
          1. Save the model of initialized agent, which will be used in "rollout" to study the training process.
          2. Initialize a counter to record the number of accumulated learned steps
          3. Initialize a counter to record the current checkpoint of saving agent
          """
          super().__init__(config)
          self.config = config
          save_class(self.config.agent_save_dir, 'checkpoint0', self)  # save the model of initialized agent.
          self.learned_steps = 0   # record the number of accumulated learned steps
          self.cur_checkpoint = 1  # record the current checkpoint of saving agent
          """
          Do whatever other setup is needed
          """
          
  	def get_action(self, state):
          """
          Parameter
          ----------
          state: state features defined by developer.
          
          Return
          ----------
          action: the action inferenced by using state.
          """
          
      def train_episode(self, env):
          """ Called by Trainer.
              Optimize a problem instance in training set until reaching max_learning_step or satisfy the convergence condition.
              During every train_episode,you need to train your own network.
  
          Parameter
          ----------
          env: an environment consisting of a backbone optimizer and a problem sampled from train set.
  
          Must To Do
          ----------
          1. record total reward
          2. record current learning steps and check if reach max_learning_step
          3. save agent model if checkpoint arrives
  
          Return
          ----------
          A boolean that is true when fes reaches max_learning_step otherwise false
          A dict: {'normalizer': float,
                   'gbest': float,
                   'return': float,
                   'learn_steps': int
                   }
          """
          state = env.reset()
          R = 0  # total reward
          """
          begin loop：
          """
              action = self.get_action(state)
              next_state, reward, is_done = env.step(action) # feed the action to environment
              R += reward  # accumulate reward
              """
              perform update strategy of agent, which is defined by you. Every time update your agent, please increase self.learned_step accordingly
              """
  
              # save agent model if checkpoint arrives
              if self.learned_steps >= (self.config.save_interval * self.cur_checkpoint):
                  save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
                  self.cur_checkpoint += 1
  
              state = next_state
  
              """
              check if finish loop
              """
          return self.learned_steps >= self.config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                                       'gbest': env.optimizer.cost[-1],
                                                                       'return': R,
                                                                       'learn_steps': self.learned_steps}
  
      def rollout_episode(self, env):
          """ Called by method rollout and Tester.test
  
          Parameter
          ----------
          env: an environment consisting of a backbone optimizer and a problem sampled from test set
  
          Return
          ----------
          A dict: {'cost': list, 
                   'fes': int, 
                   'return': float
                   }
          """
          state = env.reset()
          is_done = False
          R = 0  # total reward
          while not is_done:
              action = self.get_action(state)
              next_state, reward, is_done = env.step(action) # feed the action to environment
              R += reward  # accumulate reward
              state = next_state
  
          return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes, 'return': R}
  ```
  
* Your backbone optimizer should follow this template:

  ```python
  from optimizer.learnable_optimizer import Learnable_Optimizer
  
  class MyOptimizer(Learnable_Optimizer):
      def __init__(self, config):
          """
          Parameter
          ----------
          config: An argparse.Namespace object for passing some core configurations such as maxFEs.
          """
          super().__init__(config)
          self.config = config
          """
          Do whatever other setup is needed
          """
  
      def init_population(self, problem):
          """ Called by method PBOEnv.reset.
              Init the population for optimization.
  
          Parameter
          ----------
          problem: a problem instance, you can call `problem.eval` to evaluate one solution.
  
          Must To Do
          ----------
          1. Initialize a counter named "fes" to record the number of function evaluations used.
          2. Initialize a list named "cost" to record the best cost at logpoints.
          3. Initialize a counter to record the current logpoint.
  
          Return
          ----------
          state: state features defined by developer.
          """
  
          """
          Initialize the population, calculate the cost using method problem.eval and renew everything (such as some records) that related to the current population.
          """
          self.fes = self.population_size  # record the number of function evaluations used
          self.cost = [self.best_cost]     # record the best cost of first generation
          self.cur_logpoint = 1            # record the current logpoint
          """
          calculate the state
          """
          return state
  
      def update(self, action, problem):
          """ update the population using action and problem.
              Used in Environment's step
  
          Parameter
          ----------
          action: the action inferenced by agent.
          problem: a problem instance.
  
          Must To Do
          ----------
          1. Update the counter "fes".
          2. Update the list "cost" if logpoint arrives.
  
          Return
          ----------
          state: represents the observation of current population.
          reward: the reward obtained for taking the given action.
          is_done: whether the termination conditions are met.
          """
  
          """
          update population using the given action and update self.fes
          """
          # append the best cost if logpoint arrives
          if self.fes >= self.cur_logpoint * self.config.log_interval:
              self.cur_logpoint += 1
              self.cost.append(self.best_cost)
          """
          get state, reward and check if it is done
          """
          if is_done:
              if len(self.cost) >= self.config.n_logpoint + 1:
                  self.cost[-1] = self.best_cost
              else:
                  self.cost.append(self.best_cost)
          return state, reward, is_done
  ```

By the way, if you are developing classic optimizer, please refer to [example classic optimizer](src/optimizer/deap_de.py).

After that, you should put your own declaring files in directory `src/agent/` and `src/optimizer/` respectively. Then the file structure should be like:

```
src
│        
├─ agent
│   │
│   ├─ de_ddqn_agent.py
│   ├─ ...
│   ├─ rlepso_agent.py
│   └─ my_agent.py
└─ optimizer
    │
    ├─ dq_ddqn_optimizer.py
    ├─ ...
    ├─ rlepso_optimizer.py
    └─ my_optimizer.py
```

In addition, you should register you own agent and backbone optimizer in files `src/agent/__init__.py` and `src/optimizer/__init__.py`. For example, to register the previous class *MyAgent*, you should add one line into the `src/agent/__init__.py` file as below:
```python
from .my_agent import *
```

Meanwhile, you should also import your own agent and backbone optimizer into `src/trainer.py` and `src/tester.py`. Take trainer as an example, you should add two lines into file `src/trainer.py` as follows:
```python
...
# import your agent
from agent import{
     ...
     MyAgent
}
# import your optimizer
from optimizer import{
     ...
     MyOptimizer
}
```
The same action should be done also in `src/tester.py`.

As mentioned, four modes are available:

* `run_experiment` your MetaBBO-RL optimizer.

   `run_experiment` mode implements fully automated workflow. Assume that you've written an agent class named *MyAgent* and a backbone optimizer class named *MyOptimizer*, the entire processes of train, rollout and test can be triggered by running command:

   ```shell
   python main.py --run_experiment --train_agent MyAgent --train_optimizer MyOptimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search
   ```

   See [Run Experiment](#Run-Experiment) for more details.

* `train` your agent.

  ```shell
  python main.py --train --train_agent MyAgent --train_optimizer MyOptimizer
  ```

  Once you run the above command, the `runName` which is generated based on the run time and benchmark suite will appear at the command line. 

  See [Training](#Training) for more details.

* `rollout` your agent models.

  Fetch your 21 trained agent models named `checkpointN.pkl` in directory `src/agent_model/train/MyAgent/runName/` and move them to directory `src/agent_model/rollout/MyAgent/`. Rollout the models with train set using:

  ```shell
  python main.py --rollout --agent_load_dir agent_model/rollout/ --agent_for_rollout MyAgent --optimizer_for_rollout MyOptimizer
  ```

  When the rollout ends, check the result data in `src/output/rollout/runName/rollout.pkl` and pick the best model to test.

  See [Rollout](#Rollout) for more details.

* `test` your MetaBBO-RL optimizer.

  Move the best `.pkl` model file to directory `src/agent_model/test/`, and rename the file to `MyAgent.pkl`. Now use the test set to test `MyAgent` with `DEAP_DE`, `JDE21`, `DEAP_CMAES` and `Random_search`:

  ```shell
  python main.py --test --agent_load_dir agent_model/test/ --agent_for_cp MyAgent --l_optimizer_for_cp MyOptimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search
  ```

  See [Testing](#Testing) for more details.

## Run Experiment

In `MetaBox`, you can select the run_experiment mode by using the `--run_experiment` option. We will help you automatically organize the four functions including `train`, `rollout`, `test`, and log, and help you automatically plan the file directory to save the model, load the model, and save the test results during the process of train, test and etc. Note that you need to initialize your defined agent and optimizer and select the learning-based and traditional optimizers you need to compare before starting the `run_experiment` mode.

```shell
python main.py --run_experiment --problem bbob --difficulty easy --train_agent MyAgent --train_optimizer MyOptimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search --log_dir YourLogDir
```

Notice that during `test` function in `run_experiment`, although we rollout the 21 models generated in the train and save the results, we will choose the last checkpoint i.e. checkpoint20.pkl for the comparison of the test.

After `run_experiment`, we will save the generated results of train, rollout, test in `src/output/` respectively. Check sections [Train Results](#Train-Results),  [Rollout Results](#Rollout-Results) and [Test Results](#Test-Results) for more details of the generated results.

## Training

### How to Train
In `MetaBox`, to facilitate training with our dataset and observing logs during training, we suggest that you put your own MetaBBO Agent declaration file in the folder [agent](src/agent) and **import** it in [trainer.py](src/trainer.py). Additionally, if you are using your own optimizer instead of the one provided by `MetaBox`, you need to put your own backbone optimizer declaration file in the folder [optimizer](src/optimizer) and **import** it in [trainer.py](src/trainer.py).

You will then be able to train your agent using the following command line:

```bash
python main.py --train --train_agent MyAgent --train_optimizer MyOptimizer --agent_save_dir MyAgentSaveDir --log_dir MyLogDir
```

For the above commands, `--train` is to specify the training mode. `--train_agent MyAgent` `--train_optimizer MyOptimizer` is to use your agent class named *MyAgent* and your optimizer class named *MyOptimizer*  for training. `--agent_save_dir MyAgentSaveDir` specifies the save directory of the agent models obtained from training or they will be saved in directory `src/agent_model/train` by default.  `--log_dir MyLogDir` specifies the save directory of the log files during training or directory `src/output/train` by default.

Once you run the above command, `MetaBox` will initialize a `Trainer` object and use your configuration to build the agent and optimizer, as well as generate the training and test sets. After that, the `Trainer` will control the entire training process, optimize the problems in the train set one by one using the declared agent and optimizer, and record the corresponding information.

### Train Results

After training, **21 agent models named `checkpointN.pkl` (*N* is a number from 0 to 20) will be saved in `MyAgentSaveDir/train/MyAgent/runName/` or `agent_model/train/MyAgent/runName/` by default.** `checkpoint0.pkl` is the agent without any learning and remaining 20 models are agents saved uniformly along the whole training process, i.e., `checkpoint20.pkl` is the one that learned the most, for `--max_learning_step` times. You can choose the best one in [Rollout](#Rollout).

In addition, 2 types of data files will be generated in `MyLogDir/train/MyAgent/runName/` or `output/train/MyAgent/runName/` by default: 

* `.npy` files in `MyLogDir/train/MyAgent/runName/log/`, which you can use to draw your own graphs or tables.
* `.png` files in `MyLogDir/train/MyAgent/runName/pic/`. In this folder, 3 types of graphs are provided by our unified interfaces which draw the same graph for different agents for comparison:
	* `draw_cost`: The cost change for the agent facing different runs for different problems and save it to `MyLogDir/train/MyAgent/runName/pic/problem_name_cost.png`.
	* `draw_average_cost`: It will plot the average cost of the agent against all problems and save it to `MyLogDir/train/MyAgent/runName/pic/all_problem_cost.png`.
	* `draw_return`: The return value from the agent training process will be plotted and saved to `MyLogDir/train/MyAgent/runName/pic/return.png`.

## Rollout

### How to Rollout

By using the following command, you can rollout your agent models obtained from training process above using problems in train set: 

```bash
python main.py --rollout --agent_load_dir MyAgentLoadDir --agent_for_rollout MyAgent --optimizer_for_rollout MyOptimizer --log_dir MyLogDir 
```

But before running it, **please make sure that the 21 agent models named `checkpointN.pkl` saved from training process are in a folder named your agent class name *MyAgent*, and this folder is in directory *MyAgentLoadDir***, which seems like:

```
MyAgentLoadDir
│        
└─ MyAgent
    │
    ├─ checkpoint0.pkl
    ├─ checkpoint1.pkl
    ├─ ...
    └─ checkpoint20.pkl
```

### Rollout Results

After rollout, in `MyLogDir/rollout/runName` or `output/rollout/runName` by default, `MetaBox` will generate a file named `rollout.pkl` which is a dictionary containing:

* `cost` is the best costs sampled every 400 function evaluations along the rollout process of each checkpoint model running on each problem in train set.
* `fes` is the function evaluation times used by each checkpoint model running on each problem in train set.
* `return` is the total reward in the rollout process of each checkpoint model running on each problem in train set.

## Testing

### How to Test

In `MetaBox`, you can select the test mode by using the `--test` option. When conducting evaluations, we first instantiate a `Tester` object and load all agents and optimizers. Then, we build the test sets and, for each problem in the test set, we call each instantiated optimizer to test the problem and obtain a solution, recording 51 runs of optimization performance.

Currently, we have implemented 7 MetaBBO-RL optimizers, 1 MetaBBO-SL optimizer and 11 classic optimizers, which are listed in [Baselines](#Baselines). You can also find their implementations in [src/agent](src/agent) and [src/optimizer](src/optimizer). **We have imported all of these agents and optimizers in [tester.py](src/tester.py) for you to compare, and you are supposed to import your own agent and optimizer in it**.

You can use the `--agent_for_cp xxx` option to select the agent(s) for comparison and `--l_optimizer_for_cp xxx` option to select the learnable optimizer(s) for comparison. Please note that the agent needs to support the corresponding learnable optimizer. Additionally, you can use `--t_optimizer_for_cp xxx` to select the traditional optimizer(s) for comparison.  **`--agent_load_dir` option specifies the directory that contains the `.pkl` model files of your own agent and all comparing agents, and make sure that the model files are named after the class name of corresponding agent**, for example, `DE_DDQN_Agent.pkl`. `--log_dir` option specifies the directory where log files will be saved. 

You can test your own agent *MyAgent* and optimizer *MyOptimizer* with DE_DDQN, LDE, DEAP_DE, JDE21, DEAP_CMAES, Random_search using the following command:

```shell
python main.py --test --agent_load_dir MyAgentLoadDir --agent_for_cp MyAgent DE_DDQN_Agent LDE_Agent --l_optimizer_for_cp MyOptimizer DE_DDQN_Optimizer LDE_Optimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search --log_dir MyLogDir
```

For the above command, `MetaBox` will first load the trained model from *MyAgentLoadDir*, then initialize the agents and optimizers of yours, DE_DDQN and LDE and the selected traditional optimizers, and use the generated test set to optimize all the selected problems for testing. 

### Test Results

After testing, 3 types of data files will be generated in `MyLogDir/test/runName` or `output/test/runName` by default: 

* `test.pkl` is a dictionary containing the following testing data of time complexity and optimization performance, which can be used to generate your own graphs and tables.

  * `T0` is the time of running following computations *max function evaluations* times and is consistent for all algorithms.

    ```python
    x = np.random.rand(dim)
    x + x
    x / (x+2)
    x * x
    np.sqrt(x)
    np.log(x)
    np.exp(x)
    ```

  * `T1` is the evaluation time of the first problem in test set.

  * `T2` is the test time of a specific algorithm running on the first problem in test set.

  * `cost` is the best costs sampled every 400 function evaluations along the test process of each algorithm running on each problem for 51 times.

  * `fes` is the function evaluation times used by each algorithm running on each problem for 51 times.

* `.xlsx` files in `MyLogDir/test/runName/tables/`, contains 3 types of excel tables:

  * `algorithm_complexity.xlsx` contains time complexity calculated by `T0`, `T1` and `T2` for each comparing algorithms.
  * `algorithm_name_concrete_performance_table.xlsx` such as *RLEPSO_Agent_concrete_performance_table.xlsx* and *GL_PSO_concrete_performance_table.xlsx*, contains the specific algorithm's performance , i.e., the worst, best, median, mean, std of the costs the optimizer obtained on each problem in test set.
  * `overall_table.xlsx` contains optimization performance of all comparing algorithms on each problem of test set.
* `.png` files in `MyLogDir/test/runName/pics/`, contains 4 types of graphs:
  * `algorithm_name_concrete_performance_hist.png`, such as *RLEPSO_Agent_concrete_performance_hist.png* and *GL_PSO_concrete_performance_hist.png*, draws the performance histogram of the specific algorithm on each problem.
  * `problem_name_cost_curve.png` such as *Schwefel_cost_curve.png*, draws the cost curve of each algorithm's optimization process on the specific problem.
  * `all_problem_cost_curve.png` draws each algorithm's average cost curve on all problems in test set.
  * `rank_hist.png` plots a histogram of each algorithm's AEI score, which take the best objective value, the budget to achieve a predefined accuracy (convergence rate), and the runtime complexity into account to assign a comprehensive score to measure BBO performance.

### MGD Test

Meta Generalization Decay (MGD) metric is to assess the generalization performance of MetaBBO-RL for unseen tasks. Before running MGD test, you should prepare two agent models that have trained on two different problem sets respectively.

Run the following command to execute MGD test:

```shell
python main.py --mgd_test --problem_from bbob-noisy --difficulty_from easy --problem_to bbob --difficulty_to easy --agent MyAgent --optimizer MyOptimizer --model_from path_of_model_trained_on_problem_from --model_to path_of_model_trained_on_problem_to
```

Then, the program will print the MGD score when it ends.

### MTE Test

Meta Transfer Efficiency (MTE) metric is to evaluate the transfer learning capacity of a MetaBBO-RL approach. Before running MTE test, two agent models need to be prepared. One is trained on *--problem_to*, and another one is pre-trained on *--problem_from* then is transferred to continue training on *--problem_to*. After that, rollout these two models on *--problem_to* respectively to obtain two `rollout.pkl` files.

Run the following command to execute MTE test:

```shell
python main.py --mte_test --problem_from bbob-noisy --difficulty_from easy --problem_to bbob --difficulty_to easy --agent MyAgent --pre_train_rollout path_of_pkl_result_file_of_pretrain_rollout --scratch_rollout path_of_pkl_result_file_of_scratch_rollout
```

When the test ends, it will print the MTE score and save a figure at directory `src/output/mte_test/runName/`.

