## Run Experiment

In `MetaBox`, you can select the run_experiment mode by using the `--run_experiment` option. We will help you automatically organize the four functions including `train`, `rollout`, `test`, and log, and help you automatically plan the file directory to save the model, load the model, and save the test results during the process of train, test and etc. Note that you need to initialize your defined agent and optimizer and select the learning-based and traditional optimizers you need to compare before starting the `run_experiment` mode.

```shell
python main.py --run_experiment --problem bbob --difficulty easy --train_agent MyAgent --train_optimizer MyOptimizer --agent_load_dir agent_model/test/bbob_easy/ --agent_for_cp LDE_Agent --l_optimizer_for_cp LDE_Optimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search --log_dir YourLogDir 
```

`--train_agent MyAgent` `--train_optimizer MyOptimizer` is to use your agent class named *MyAgent* and your optimizer class named *MyOptimizer*  for training. In `run_experiment` mode, you can also select built-in baselines in MetaBox to compare via `--t_optimizer_for_cp` to specify classic baselines and `--agent_for_cp` `--l_optimizer_for_cp` to specify learnable baselines. Noting that when you specify `--agent_for_cp`, you must provide `--agent_load_dir` to specify the directory saving trained baselines models, in the example above, `agent_load_dir` is provided as `agent_model/test/bbob_easy/` in which trained model on 10D easy Synthetic we provide is located.

Notice that during `test` function in `run_experiment`, although we rollout the 21 models generated in the train and save the results, we will choose the last checkpoint i.e. checkpoint20.pkl for the comparison of the test.

After `run_experiment`, we will save the generated results of train, rollout, test in `src/output/` respectively. Check sections [Train Results](Train.md),  [Rollout Results](Rollout.md) and [Test Results](Test.md) for more details of the generated results.