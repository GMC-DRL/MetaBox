## Datasets


Currently, three benchmark suites are included:  

* `Synthetic` containing 24 noiseless functions, borrowed from [coco](https://github.com/numbbo/coco):bbob with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Noisy-Synthetic` containing 30 noisy functions, borrowed from [coco](https://github.com/numbbo/coco):bbob-noisy with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Protein-Docking` containing 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem, borrowed from [LOIS](https://github.com/Shen-Lab/LOIS) with [original paper](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms).

By setting the argument `--problem` to `bbob`, `bbob-noisy` or `protein` in command line to use the corresponding suite, for example:

```bash
python main.py --train --problem protein --train_agent MyAgent --train_optimizer MyOptimizer
```

For the usage of  `--train`  `--train_agent`  `--train_optimizer`, see [Training](Train.md) for more details.

Each test suites are regarded as a dataset, which is split into training set and test set in different proportions with respect to two difficulty levels:  

* `easy` training set accounts for 75% and test set accounts for 25%.
* `difficult` training set accounts for 25% and test set accounts for 75%.

By setting the argument `--difficulty` to `easy` or `difficult` in command line to specify the difficulty level like the following command. Note that `easy` difficulty is used by default.

```bash
python main.py --train --problem bbob --difficulty difficult --train_agent MyAgent --train_optimizer MyOptimizer
```

