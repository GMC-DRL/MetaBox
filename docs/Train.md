## Training

### How to Train

In `MetaBox`, to facilitate training with our dataset and observing logs during training, we suggest that you put your own MetaBBO Agent declaration file in the folder [agent](https://github.com/GMC-DRL/MetaBox/tree/main/src/agent) and **import** it in [trainer.py](https://github.com/GMC-DRL/MetaBox/tree/main/src/trainer.py). Additionally, if you are using your own optimizer instead of the one provided by `MetaBox`, you need to put your own backbone optimizer declaration file in the folder [optimizer](https://github.com/GMC-DRL/MetaBox/tree/main/src/optimizer) and **import** it in [trainer.py](https://github.com/GMC-DRL/MetaBox/tree/main/src/trainer.py).

You will then be able to train your agent using the following command line:

```bash
python main.py --train --train_agent MyAgent --train_optimizer MyOptimizer --agent_save_dir MyAgentSaveDir --log_dir MyLogDir
```

For the above commands, `--train` is to specify the training mode. `--train_agent MyAgent` `--train_optimizer MyOptimizer` is to use your agent class named *MyAgent* and your optimizer class named *MyOptimizer*  for training. `--agent_save_dir MyAgentSaveDir` specifies the save directory of the agent models obtained from training or they will be saved in directory `src/agent_model/train` by default.  `--log_dir MyLogDir` specifies the save directory of the log files during training or directory `src/output/train` by default.

Once you run the above command, `MetaBox` will initialize a `Trainer` object and use your configuration to build the agent and optimizer, as well as generate the training and test sets. After that, the `Trainer` will control the entire training process, optimize the problems in the train set one by one using the declared agent and optimizer, and record the corresponding information.

### Train Results

After training, **21 agent models named `checkpointN.pkl` (*N* is a number from 0 to 20) will be saved in `MyAgentSaveDir/train/MyAgent/runName/` or `agent_model/train/MyAgent/runName/` by default.** `checkpoint0.pkl` is the agent without any learning and remaining 20 models are agents saved uniformly along the whole training process, i.e., `checkpoint20.pkl` is the one that learned the most, for `--max_learning_step` times. You can choose the best one in [Rollout](Rollout.md).

In addition, 2 types of data files will be generated in `MyLogDir/train/MyAgent/runName/` or `output/train/MyAgent/runName/` by default: 

* `.npy` files in `MyLogDir/train/MyAgent/runName/log/`, which you can use to draw your own graphs or tables.
* `.png` files in `MyLogDir/train/MyAgent/runName/pic/`. In this folder, 3 types of graphs are provided by our unified interfaces which draw the same graph for different agents for comparison:
  * `draw_cost`: The cost change for the agent facing different runs for different problems and save it to `MyLogDir/train/MyAgent/runName/pic/problem_name_cost.png`.
  * `draw_average_cost`: It will plot the average cost of the agent against all problems and save it to `MyLogDir/train/MyAgent/runName/pic/all_problem_cost.png`.
  * `draw_return`: The return value from the agent training process will be plotted and saved to `MyLogDir/train/MyAgent/runName/pic/return.png`.

