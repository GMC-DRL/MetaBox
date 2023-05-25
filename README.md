# RELOPS: Reinforcement Learning Benchmark Platform for Black Box Optimizer Search

This is a reinforcement learning benchmark platform that supports benchmarking and exploration of black box optimizers. You can train your own optimizer or compare it with several popular RL-based optimizers and traditional optimizers.

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
* `opencv_python`==4.5.4.58  
* `tqdm`==4.62.3  
* `openpyxl`==3.1.2

## Dataset


Currently, three benchmark suites are included:  

* `bbob` containing 24 noiseless functions<sup>1</sup> 
* `bbob-noisy` containing 30 noisy functions<sup>1</sup> 
* `protein docking` containing 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem<sup>2</sup>     

Set the argument `--problem` in command line to specify the suite like:

```bash
python main.py --train --problem bbob --train_agent YourAgent --train_optimizer YourOptimizer
```

```bash
python main.py --train --problem bbob-noisy --train_agent YourAgent --train_optimizer YourOptimizer
```

```bash
python main.py --train --problem protein --train_agent YourAgent --train_optimizer YourOptimizer
```

For the usage of  `--train`  `--train_agent`  `--train_optimizer` , see [Training](#Training) for more details.

> 1. `bbob` and `bbob-noisy` suites come from [COCO](https://github.com/numbbo/coco) with original paper [COCO: a platform for comparing continuous optimizers in a black-box setting](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
> 2. `protein docking` comes from [LOIS](https://github.com/Shen-Lab/LOIS) with original paper [Learning to Optimize in Swarms](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms).

The data set is split into training set and test set in different proportions with respect to two difficulty levels:  

* `easy` training set accounts for 75% 
* `difficult` training set accounts for 25%  

You can specify the difficulty level by setting the argument `--difficulty` in command line like:

```bash
python main.py --train --problem bbob --difficulty easy --train_agent YourAgent --train_optimizer YourOptimizer
```

```bash
python main.py --train --problem bbob --difficulty difficult --train_agent YourAgent --train_optimizer YourOptimizer
```


## Training
In `RELOPS`, to facilitate training with our dataset and observing logs during training, we suggest that you put your own MetaBBO Agent declaration file in the Agent folder and declare it as a package in Agent. Additionally, if you are using your own optimizer instead of the one provided by `RELOPS`, you need to put your own backbone optimizer declaration file in the Optimizer folder and declare it as a package in Optimizer.

If you specify the training mode, `RELOPS` will initialize a Trainer object for you and use your configuration to build the objects of the comparison Agent and the comparison Optimizer, as well as generate the training and test sets. After that, the Trainer will control the entire training process, optimize the problems in the train set one by one using the declared agent and optimizer, and record the corresponding information.

[example agent](https://github.com/GMC-DRL/L2OBench/blob/dev/L2OBench/agent/de_ddqn_agent.py)

[example optimizer](https://github.com/GMC-DRL/L2OBench/blob/dev/L2OBench/optimizer/de_ddqn_optimizer.py)

You will then be able to train it using the following command line:
```bash
python main.py --train --train_agent xxx --train_optimizer xxx --agent_save_dir xxx --log_dir xxx
```
For the above commands, --train is to specify the training mode, --train_agent xxx --train_optimizer xxx is to use xxx as the agent and xxx as the optimzier for training. --agent_save_dir xxx specifies the save directory of the agent obtained from training, --log_dir xxx specifies the save directory of the log file during training.

After that, you can query all the data generated during the training process, including return and cost, in `log_dir/train/agent.name/log`, and you can use the above data to draw your own graphs or tables. In addition, we also provide a unified interface to draw the same graph for different agents for comparison.

we provided：

1. `draw_cost`：The cost change for the agent facing different runs for different problems and save it to`log_dir/train/agent.name/pic/run/problem_name/cost.png`.
2. `draw_average_cost`:It will plot the average cost of the agent against all problems and save it to`log_dir/train/agent.name/pic/run/all_problem_cost.png`.
3. `draw_return`The return value from the agent training process will be plotted and saved to`log_dir/train/agent.name/pic/run/return.png`.

## Baselines

7 RL-based optimizers, 1 meta-learning optimizer and 11 traditional optimizers have been integrated into this platform. Choose one or more of them to be the baseline(s) to test the performance of your own optimizer.

Supported RL-based optimizers:

|   Name   | Year |                        Related paper                         |
| :------: | :--: | :----------------------------------------------------------: |
| DE-DDQN  | 2019 | [Deep reinforcement learning based parameter control in differential evolution](https://dl.acm.org/doi/10.1145/3321707.3321813) |
|   LDE    | 2021 | [Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient](https://ieeexplore.ieee.org/document/9359652) |
|  DEDQN   | 2021 | [Differential evolution with mixed mutation strategy based on deep reinforcement learning](https://www.sciencedirect.com/science/article/pii/S1568494621005998) |
| RL-HPSDE | 2022 | [Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning](https://www.sciencedirect.com/science/article/pii/S2210650222001602) |
|  RL-PSO  | 2021 | [Employing reinforcement learning to enhance particle swarm optimization methods](https://www.tandfonline.com/doi/full/10.1080/0305215X.2020.1867120) |
|  RLEPSO  | 2022 | [RLEPSO:Reinforcement learning based Ensemble particle swarm optimizer✱](https://dl.acm.org/doi/abs/10.1145/3508546.3508599) |
|  QLPSO   | 2019 | [A reinforcement learning-based communication topology in particle swarm optimization](https://link.springer.com/article/10.1007/s00521-019-04527-9) |

Supported meta-learning optimizer:

| Name | Year |                        Related paper                         |
| :--: | :--: | :----------------------------------------------------------: |
| L2L  | 2017 | [Learning to learn without gradient descent by gradient descent](https://dl.acm.org/doi/10.5555/3305381.3305459) |

Supported traditional optimizers:

|     Name      | Year |                        Related paper                         |
| :-----------: | :--: | :----------------------------------------------------------: |
|    GL-PSO     | 2015 | [Genetic Learning Particle Swarm Optimization](https://ieeexplore.ieee.org/abstract/document/7271066/) |
|      j21      | 2021 | [Self-adaptive Differential Evolution Algorithm with Population Size Reduction for Single Objective Bound-Constrained Optimization: Algorithm j21](https://ieeexplore.ieee.org/document/9504782) |
|     MadDE     | 2021 | [Improving Differential Evolution through Bayesian Hyperparameter Optimization](https://ieeexplore.ieee.org/document/9504792) |
| NL_SHADE_LBC  | 2022 | [NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization](https://ieeexplore.ieee.org/abstract/document/9870295) |
|    SAHLPSO    | 2021 | [Self-Adaptive two roles hybrid learning strategies-based particle swarm optimization](https://www.sciencedirect.com/science/article/pii/S0020025521006988) |
|   sDMS_PSO    | 2015 | [A Self-adaptive Dynamic Particle Swarm Optimizer](https://ieeexplore.ieee.org/document/7257290) |
|    CMA-ES     | 2001 | [Completely Derandomized Self-Adaptation in Evolution Strategies](https://ieeexplore.ieee.org/document/6790628) |
|      DE       | 1997 | [Differential evolution-a simple and efficient heuristic for global optimization over continuous spaces](https://dl.acm.org/doi/abs/10.1023/A%3A1008202821328)                                                             |
|      PSO      | 1995 | [Particle swarm optimization](https://ieeexplore.ieee.org/abstract/document/488968)                                                          |
|   Bayesian    | 2014 | [Bayesian Optimization: Open source constrained global optimization tool for Python](https://github.com/bayesian-optimization/BayesianOptimization) |
| Random search |  -   |                              -                               |

Note that `Random search` performs uniformly random sampling to optimize the fitness.


## Testing

In RELOPS, you can select the test mode by using the `--test` option. When conducting evaluations, we first instantiate a Tester object and load all agents and optimizers. Then, we build the test sets and, for each problem in the test set, we call each instantiated optimizer to test the problem and obtain a solution, recording 51 runs of optimization performance.

You can use `utils.construct_problem_set` to generate training and test sets. Currently, we have implemented 8 RL-based learnable optimizers and 11 traditional optimizers, which are listed in `Baselines`. You can also find their implementations in RELOPS.agent and RELOPS.optimizer. In the Tester, we have imported all of these optimizers for you to compare.

You can use the `--agent_for_cp xxx` option to select the agent for comparison and `--l_optimizer_for_cp xxx` option to select the learnable optimizer for comparison. Please note that the agent needs to support the corresponding learnable optimizer. Additionally, you can use `--t_optimizer_for_cp xxx` to select the traditional optimizer for comparison.

You can test the algorithm with the following command line:

```bash
python main.py --agent_load_dir agent_model/test/bbob_easy/ --agent_for_cp DE_DDQN_Agent --l_optimizer_for_cp DE_DDQN_Optimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search
```

For the above command, `RELOPS` will first load the trained model from `agent_model/test/bbob_easy/`, then initialize the DE_DDQN agent and optimizer to be compared with the selected traditional optimizer, and use the generated test set to optimize all the selected problems for testing.

`RELOPS` plots the change in cost as fes grows.

todo:add T0 T1 T2



