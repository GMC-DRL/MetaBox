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
