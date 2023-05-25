# RELOPS: Reinforcement Learning Benchmark Platform for Black Box Optimizer Search

This is a reinforcement learning benchmark platform that supports benchmarking and exploration of black box optimizers. You can train your own optimizer and compare it with several popular RL-based optimizers and traditional optimizers.

## Overview

![overview](docs/overview.png)

We have divided `RELOPS` into five modules: **MetaBBO-RL**, **Testsuites**, **Trainer**, **Tester** and **Logger**.

`MetaBBO-RL` is an optimizer, which is consists of a reinforcement learning agent and a backbone optimizer, can be used for optimizing black box problems.

`Testsuites` are used for generating training and testing sets, including **bbob**, **bbob-noisy**, and **protein docking**.

`Trainer` is used to integrate the entire training process of the optimizer and consists of the instantiated agent, the **ENV** consisting of the optimizer and Train set. The trainer organizes the MDP process of reinforcement learning.What's more,it can record information from the training process.

`Tester` is used to evaluate the optimization effect of the optimizer. It contains baseline algorithms, trained MetaBBO Agent, and Test set. What's more,it can record information from the evaluation process and generate statistical test results using the above information.

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

`requirements.txt` is defined [here](requirements.txt)

## Datasets


Currently, three benchmark suites are included:  

* `bbob` containing 24 noiseless functions<sup>1</sup> 
* `bbob-noisy` containing 30 noisy functions<sup>1</sup> 
* `protein docking` containing 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem<sup>2</sup>     

By setting the argument `--problem` ,you can specify the suite.More details and examples can be found [here](docs/dataset.md)

For the usage of  `--train`  `--train_agent`  `--train_optimizer` , see [Training](#Training) for more details.

> 1. `bbob` and `bbob-noisy` suites come from [COCO](https://github.com/numbbo/coco) with original paper [COCO: a platform for comparing continuous optimizers in a black-box setting](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
> 2. `protein docking` comes from [LOIS](https://github.com/Shen-Lab/LOIS) with original paper [Learning to Optimize in Swarms](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms).

The data set is split into training set and test set in different proportions with respect to two difficulty levels:  

* `easy` training set accounts for 75% 
* `difficult` training set accounts for 25%  

By setting the argument `--difficulty` ,you can specify the difficulty level. More details and examples can be found [here](docs/dataset.md)


## Training
### How to Train
In `RELOPS`, to facilitate training with our dataset and observing logs during training, we suggest that you put your own MetaBBO Agent declaration file in the folder [agent](RELOPS/agent) and **import** it in [trainer.py](RELOPS/trainer.py). Additionally, if you are using your own optimizer instead of the one provided by `RELOPS`, you need to put your own backbone optimizer declaration file in the folder [optimizer](RELOPS/optimizer) and **import** it in [trainer.py](docs/trainer.py).

Also, when using RELOPS, you must implement your MetaBBO Agent and Optimizer in a certain form.

Get more details [here](docs/training.md).

After that ,you will be able to train your agent using the following command line:

```bash
python main.py --train --train_agent MyAgent --train_optimizer MyOptimizer --agent_save_dir MyAgentSaveDir --log_dir MyLogDir
```

More details and explanation can be found [here](docs/training.md)

### Train Results

After training, 2 types of data files will be generated in `MyLogDir/train/MyAgent/runName` or `output/train/MyAgent/runName` by default: 

* `.npy` files in `MyLogDir/train/MyAgent/runName/log/`, which you can use to draw your own graphs or tables.
* `.png` files in `MyLogDir/train/MyAgent/runName/pic/`. In this folder, 3 types of graphs are provided by our unified interfaces which draw the same graph for different agents for comparison.

More details can be found [here](docs/training.md)

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

In `RELOPS`, you can select the test mode by using the `--test` option. 

Currently, we have implemented 7 MetaBBO-RL learnable optimizers, 1 MetaBBO-SL optimizer and 11 BBO optimizers, which are listed in [Baselines](#Baselines). You can also find their implementations in [RELOPS/agent](RELOPS/agent) and [RELOPS/optimizer](RELOPS/optimizer). We have imported all of these agents and optimizers in [tester.py](RELOPS/tester.py) for you to compare, and you are supposed to import your own agent and optimizer in it.

You can specify all the baselines optimizers that you want to compare with your own optimizers.  In addition,you can use`--agent_load_dir` option specifies the directory that contains the `.pkl` model files of your own agent and all comparing agents,  and `--log_dir` option specifies the directory where log files will be saved. See more detaisl in [here](docs/testing.md)

You can test your own agent *MyAgent* and optimizer *MyOptimizer* with DE_DDQN, LDE, DEAP_DE, JDE21, DEAP_CMAES, Random_search using the following command:

```shell
python main.py --agent_load_dir MyAgentLoadDir --agent MyAgent --optimizer MyOptimizer --agent_for_cp DE_DDQN_Agent LDE_Agent --l_optimizer_for_cp DE_DDQN_Optimizer LDDE_Optimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search --log_dir MyLogDir
```

More details and explanation can be found [here](docs/testing.md)

### Test Results

After testing, 3 types of data files will be generated in `MyLogDir/test/runName` or `output/test/runName` by default: 

* `test.pkl` is a dictionary containing the following testing data of time complexity and optimization performance, which can be used to generate your own graphs and tables.
* `.xlsx` files in `MyLogDir/test/runName/tables/`, contains 3 types of excel tables.
* `.png` files in `MyLogDir/test/runName/pics/`, contains 4 types of graphs.

You can see the details of the test results [here](docs/testing.md)
