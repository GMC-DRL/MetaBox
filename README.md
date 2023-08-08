# MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

This is a **reinforcement learning benchmark platform** for benchmarking and MetaBBO-RL methods. You can develop your own MetaBBO-RL approach and complare it with baseline approaches built-in following the **Train-Test-Log** philosophy automated by MetaBox.

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

## Documentation

See the [MetaBox User's Guide](https://pgj-0419.github.io/PGJ-0419/) for Metabox documentation.

## Datasets


Currently, three benchmark suites are included:  

* `Synthetic` containing 24 noiseless functions, borrowed from [coco](https://github.com/numbbo/coco):bbob with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Noisy-Synthetic` containing 30 noisy functions, borrowed from [coco](https://github.com/numbbo/coco):bbob-noisy with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Protein-Docking` containing 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem, borrowed from [LOIS](https://github.com/Shen-Lab/LOIS) with [original paper](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms).

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

|  Name  | Year |                        Related paper                         |
| :----: | :--: | :----------------------------------------------------------: |
| RNN-OI | 2017 | [Learning to learn without gradient descent by gradient descent](https://dl.acm.org/doi/10.5555/3305381.3305459) |

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

