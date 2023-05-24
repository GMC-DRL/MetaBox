# RELOPS: Reinforcement Learning Benchmark Platform for Black Box Optimizer Search

This is a reinforcement learning benchmark platform that supports benchmarking and exploration of black box optimizers. You can train your own optimizer or compare it with several popular RL-based optimizers and traditional optimizers.

## Overview

框架图+概述

## Requirements  

* `Python` >=3.6 with following packages installed:  
> * `bayesian_optimization`==1.4.3  
> * `deap`==1.3.3  
> * `matplotlib`==3.4.3  
> * `numpy`==1.21.2  
> * `opencv_python`==4.5.4.58  
> * `pandas`==1.3.3  
> * `scipy`==1.7.1  
> * `torch`==1.9.0  
> * `tqdm`==4.62.3  
> * `openpyxl`==3.1.2  
___

## Getting Started



## Dataset


Currently, three benchmark suites are included:  

`bbob` containing 24 noiseless functions<sup>1</sup>  
`bbob-noisy` containing 30 noisy functions<sup>1</sup>   
`protein docking` containing 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem<sup>2</sup>  
    
> 1. `bbob` and `bbob-noisy` suites come from [coco](https://github.com/numbbo/coco) ([original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977)).
> 2. `protein docking` comes from [LOIS](https://github.com/Shen-Lab/LOIS) ([original paper](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms)).

The data set is split into training set and test set in different proportions with respect to two difficulty levels:  

`easy` training set accounts for 75%  
`difficult` training set accounts for 25%  

## Baselines




