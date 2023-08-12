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
|     RNN-OI     |         L2L_Agent         |             L2L_Optimizer              |

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

For all baselines, we display their control parameter settings in:

Classic: [control_parameters_classic](https://github.com/GMC-DRL/MetaBox/tree/main/src/control_parameters_classic.md)

Metabbo: [control_parameters_metabbo](https://github.com/GMC-DRL/MetaBox/tree/main/src/control_parameters_metabbo.md)

## Baselines' performance

To facilitate the observation of our baselines and related metrics, we tested our baselines on two levels of difficulty on three datasets. All data are provided in [content.md](https://github.com/GMC-DRL/MetaBox/tree/main/post_processed_data/content.md).
