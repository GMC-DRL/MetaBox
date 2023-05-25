# RELOPS: Reinforcement Learning Benchmark Platform for Black Box Optimizer Search

This is a reinforcement learning benchmark platform that supports benchmarking and exploration of black box optimizers. You can train your own optimizer and compare it with several popular RL-based optimizers and traditional optimizers.

## Contents

* [Overview](#Overview)
* [Requirements](#Requirements)
* [Datasets](#Datasets)
* [Training](#Training)
  * [How to Train](#How-to-Train)
  * [Train Results](#Train-Results)
* [Baselines](#Baselines)
* [Testing](#Testing)
  * [How to Test](#How-to-Test)
  * [Test Results](#Test-Results)

## Overview

![overview](docs/overview.png)

We have divided `RELOPS` into five modules: **Black Box Optimizer, Test Suite, Trainer, Tester and Logger.**

`MetaBBO-RL` is used for optimizing black box problems and consists of a reinforcement learning agent and backbone optimizer.

`Testsuites` are used for generating training and testing sets, including **bbob**, **bbob-noisy**, and **protein docking**.

`Trainer` is used to integrate the entire training process of the optimizer and consists of the instantiated agent, the **ENV** consisting of the optimizer and Train set. The trainer organizes the MDP process of reinforcement learning.

`Tester` is used to evaluate the optimization effect of the optimizer. It contains baseline algorithms, trained MetaBBO Agent, and Test set and can generate statistical test results using the above information.

`Logger` implements several functions for displaying the logs of the training process and the results on the test set, which facilitate the improvement of the training process and observation of the optimizer's effect.

## Requirements

`Python` >=3.7.1 with following packages installed:  

* `numpy`==1.21.2  
* `torch`==1.9.0  
* `matplotlib`==3.4.3  
* `pandas`==1.3.3  
* `scipy`==1.7.1
* `bayesian_optimization`==1.4.3  
* `deap`==1.3.3  
* `tqdm`==4.62.3  
* `openpyxl`==3.1.2

## Datasets


Currently, three benchmark suites are included:  

* `bbob` containing 24 noiseless functions<sup>1</sup> 
* `bbob-noisy` containing 30 noisy functions<sup>1</sup> 
* `protein docking` containing 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem<sup>2</sup>     

By setting the argument `--problem` in command line to specify the suite like:

```bash
python main.py --train --problem bbob --train_agent MyAgent --train_optimizer MyOptimizer
```

```bash
python main.py --train --problem bbob-noisy --train_agent MyAgent --train_optimizer MyOptimizer
```

```bash
python main.py --train --problem protein --train_agent MyAgent --train_optimizer MyOptimizer
```

For the usage of  `--train`  `--train_agent`  `--train_optimizer` , see [Training](#Training) for more details.

> 1. `bbob` and `bbob-noisy` suites come from [COCO](https://github.com/numbbo/coco) with original paper [COCO: a platform for comparing continuous optimizers in a black-box setting](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
> 2. `protein docking` comes from [LOIS](https://github.com/Shen-Lab/LOIS) with original paper [Learning to Optimize in Swarms](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms).

The data set is split into training set and test set in different proportions with respect to two difficulty levels:  

* `easy` training set accounts for 75% 
* `difficult` training set accounts for 25%  

By setting the argument `--difficulty` in command line to specify the difficulty level like: 

(Note that `easy` difficulty is used by default.)

```bash
python main.py --train --problem bbob --difficulty easy --train_agent MyAgent --train_optimizer MyOptimizer
```

```bash
python main.py --train --problem bbob --difficulty difficult --train_agent MyAgent --train_optimizer MyOptimizer
```


## Training
### How to Train
In `RELOPS`, to facilitate training with our dataset and observing logs during training, we suggest that you put your own MetaBBO Agent declaration file in the folder [agent](RELOPS/agent) and **import** it in [trainer.py](RELOPS/trainer.py). Additionally, if you are using your own optimizer instead of the one provided by `RELOPS`, you need to put your own backbone optimizer declaration file in the folder [optimizer](RELOPS/optimizer) and **import** it in [trainer.py](RELOPS/trainer.py).

Your own agent are supposed to derive from class `Basic_Agent` in [basic_agent.py](RELOPS/agent/basic_agent.py) and implement methods `train_episode` and `rollout_episode`. Your own optimizer should derive from class `Learnable_Optimizer` in [learnable_optimizer.py](RELOPS/optimizer/learnable_optimizer.py) and implement methods `init_population` and `update`. See [example agent](RELOPS/agent/de_ddqn_agent.py) and [example optimizer](RELOPS/optimizer/de_ddqn_optimizer.py) for more details.

You will then be able to train your agent using the following command line:

```bash
python main.py --train --train_agent MyAgent --train_optimizer MyOptimizer --agent_save_dir MyAgentSaveDir --log_dir MyLogDir
```

For the above commands, `--train` is to specify the training mode. `--train_agent MyAgent` `--train_optimizer MyOptimizer` is to use your agent class named *MyAgent* and your optimizer class named *MyOptimizer*  for training. `--agent_save_dir MyAgentSaveDir` specifies the save directory of the agent models obtained from training or they will be saved in directory `RELOPS/agent_model/train` by default.  `--log_dir MyLogDir` specifies the save directory of the log files during training or directory `RELOPS/output/train` by default.

Once you run the above command, `RELOPS` will initialize a `Trainer` object and use your configuration to build the agent and optimizer, as well as generate the training and test sets. After that, the `Trainer` will control the entire training process, optimize the problems in the train set one by one using the declared agent and optimizer, and record the corresponding information.

### Train Results

After training, 2 types of data files will be generated in `MyLogDir/train/MyAgent/runName` or `output/train/MyAgent/runName` by default: 

* `.npy` files in `MyLogDir/train/MyAgent/runName/log/`, which you can use to draw your own graphs or tables.
* `.png` files in `MyLogDir/train/MyAgent/runName/pic/`. In this folder, 3 types of graphs are provided by our unified interfaces which draw the same graph for different agents for comparison:
	* `draw_cost`: The cost change for the agent facing different runs for different problems and save it to `MyLogDir/train/MyAgent/runName/pic/problem_name_cost.png`.
	* `draw_average_cost`: It will plot the average cost of the agent against all problems and save it to `MyLogDir/train/MyAgent/runName/pic/all_problem_cost.png`.
	* `draw_return`: The return value from the agent training process will be plotted and saved to `MyLogDir/train/MyAgent/runName/pic/return.png`.

Note that the `runName` is automatically generated by `RELOPS` based on the run time and benchmark suites for distinguishment. 

## Baselines

**7 MetaBBO-RL optimizers, 1 MetaBBO-SL optimizer and 11 BBO optimizers have been integrated into this platform.** Choose one or more of them to be the baseline(s) to test the performance of your own optimizer.

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
| L2L  | 2017 | [Learning to learn without gradient descent by gradient descent](https://dl.acm.org/doi/10.5555/3305381.3305459) |

**Supported BBO optimizers**:

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

## Testing

### How to Test

In `RELOPS`, you can select the test mode by using the `--test` option. When conducting evaluations, we first instantiate a `Tester` object and load all agents and optimizers. Then, we build the test sets and, for each problem in the test set, we call each instantiated optimizer to test the problem and obtain a solution, recording 51 runs of optimization performance.

Currently, we have implemented 7 MetaBBO-RL learnable optimizers, 1 MetaBBO-SL optimizer and 11 BBO optimizers, which are listed in [Baselines](#Baselines). You can also find their implementations in [RELOPS/agent](RELOPS/agent) and [RELOPS/optimizer](RELOPS/optimizer). We have imported all of these agents and optimizers in [tester.py](RELOPS/tester.py) for you to compare, and you are supposed to import your own agent and optimizer in it.

You can use the `--agent_for_cp xxx` option to select the agent(s) for comparison and `--l_optimizer_for_cp xxx` option to select the learnable optimizer(s) for comparison. Please note that the agent needs to support the corresponding learnable optimizer. Additionally, you can use `--t_optimizer_for_cp xxx` to select the traditional optimizer(s) for comparison.  `--agent_load_dir` option specifies the directory that contains the `.pkl` model files of your own agent and all comparing agents, and **the model files should be named after the class name of corresponding agent**, for example, `DE_DDQN_Agent.pkl`. `--log_dir` option specifies the directory where log files will be saved. 

You can test your own agent *MyAgent* and optimizer *MyOptimizer* with DE_DDQN, LDE, DEAP_DE, JDE21, DEAP_CMAES, Random_search using the following command:

```shell
python main.py --agent_load_dir MyAgentLoadDir --agent MyAgent --optimizer MyOptimizer --agent_for_cp DE_DDQN_Agent LDE_Agent --l_optimizer_for_cp DE_DDQN_Optimizer LDDE_Optimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search --log_dir MyLogDir
```

For the above command, `RELOPS` will first load the trained model from *MyAgentLoadDir*, then initialize the agents and optimizers of yours, DE_DDQN and LDE and the selected traditional optimizers, and use the generated test set to optimize all the selected problems for testing. 

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

  * `T1` is the evaluation time of the problem in test set and is consistent for all algorithms.

  * `T2` is the test time of specific algorithm.

  * `cost` is the best costs sampled every 400 function evaluations along the test process of each algorithm running on each problem for 51 times.

  * `fes` is the function evaluation times used by each algorithm running on each problem for 51 times.

* `.xlsx` files in `MyLogDir/test/runName/tables/`, contains 3 types of excel tables:
  * `algorithm_complexity.xlsx` contains time complexity calculated by `T0`, `T1` and `T2` for each comparing algorithms.
  * `algorithm_name_concrete_performance_table.xlsx` such as *RLEPSO_Agent_concrete_performance_table.xlsx* and *GL_PSO_concrete_performance_table.xlsx*, contains specific algorithm's performance , i.e., the worst, best, median, mean, std of the costs the optimizer obtained on each problem in test set.
  * `overall_table.xlsx` contains optimization performance of all comparing algorithms on each problem of test set.
* `.png` files in `MyLogDir/test/runName/pics/`, contains 4 types of graphs:
  * `algorithm_name_concrete_performance_hist.png`, such as *RLEPSO_Agent_concrete_performance_hist.png* and *GL_PSO_concrete_performance_hist.png*, draws the performance histogram of the specific algorithm on each problem.
  * `problem_name_cost_curve.png` such as *Schwefel_cost_curve.png*, draws the cost curve of each algorithm's optimization process on the specific problem.
  * `all_problem_cost_curve.png` draws each algorithm's average cost curve on all problems in test set.
  * `rank_hist.png` plots a histogram of each algorithm's score.
