# MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

MetaBox is the first benchmark platform expressly tailored for developing and evaluating MetaBBO-RL methods. MetaBox offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform. Moreover, it provides a broad spectrum of over 300 problem instances, collected from synthetic to realistic scenarios, and an extensive library of 19 baseline methods, including both traditional black-box optimizers and recent MetaBBO-RL methods. Besides, MetaBox introduces three standardized performance metrics, enabling a more thorough assessment of the methods. The github repos can be referred [here](https://github.com/GMC-DRL/MetaBox) and the paper can be found [here](https://arxiv.org/abs/2310.08252).


## Overview

![overview](overview.png)

`MetaBox` can be divided into six modules: **Template, Test suites, Baseline Library, Trainer, Tester and Logger.**

* `Template` comprises two main components: the **meta-level RL agent** and the **lower-level optimizer**, which provides a unified interface protocol for users to develop their own MetaBBO-RL with ease. 
* `Test suites` are used for generating training and testing sets, including **Synthetic**, **Noisy-Synthetic**, and **Protein-Docking**.
* `Baseline Library` comprises proposed algorithms including **MetaBBO-RL**, **MetaBBO-SL** and **classic** optimizers that we implemented for comparison study.
* `Trainer` **manages the entire learning process** of the agent by building environments consisting of a backbone optimizer and a problem sampled from train set and letting the agent interact with environments sequentially.
* `Tester` is used to **evaluate** the optimization performance of the MetaBBO-RL. By using the test set to test the baselines and the trained MetaBBO agent, it produces test log for logger to generate statistic test results.
* `Logger` implements multiple interfaces for **displaying** the logs of the training process and the results of the testing process, which facilitates the improvement of the training process and the observation of MetaBBO-RL's performance.

**Data Stream**
![datastream](datastream.png)

When using MetaBox, after the training interface `Trainer.train()` called, 21 pickle files of training agent in different training process and pictures of training process will be output to `src/agent_model/train` and `src/output/train` respectively. If you want to compare performance among baselines built-in or your own approach, `Tester.test` is needed to be called. For those learnable agent in your comparing list, you need to first collect these agent model pickle files (one agent one file) to a specific folder, where the agent model file may have been outputed by `Trainer.train()` and you can find that and then copy it to the right place. When `Tester.test()` finished, tables containing per-instance result and algorithm complexity, pictures depicting comparison results or singal approach performance, original testing result will be output to `src/output/test`. In addition, for `Rollout` interface, before that you need to collect all of checkpoints of all of learning agents which can be copy from output of `Trainer.train()`. When `Rollout` finished, pictures containing average return process and optimization cost process will be output to `src/output/rollout` and the original data file will also be saved.

## Datasets and Baselines

- [Datasets](Datasets.md)
- [Baseline Library](Baselines.md)

## **Basic Tutorials**

- [Fit your own MetaBBO-RL into MetaBox](Customization.md)
- [Run Experiment](RunExperiment.md)
- [Training](Train.md)
- [Rollout](Rollout.md)
- [Testing](Test.md)

## Citing MetaBox

If you find our MetaBox useful, please cite it in your publications or projects.

```latex
@inproceedings{metabox,
author={Ma, Zeyuan and Guo, Hongshu and Chen, Jiacheng and Li, Zhenrui and Peng, Guojun and Gong, Yue-Jiao and Ma, Yining and Cao, Zhiguang},
title={MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning},
booktitle = {Advances in Neural Information Processing Systems},
year={2023},
volume = {36}
}
```

## Acknowledgements
 
The code and the framework are based on the repos [DEAP](https://github.com/DEAP/deap), [coco](https://github.com/numbbo/coco) and [Protein-protein docking V4.0](https://zlab.umassmed.edu/benchmark/).
