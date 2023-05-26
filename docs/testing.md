## Testing

### How to Test

In `RELOPS`, you can select the test mode by using the `--test` option. When conducting evaluations, we first instantiate a `Tester` object and load all agents and optimizers. Then, we build the test sets and, for each problem in the test set, we call each instantiated optimizer to test the problem and obtain a solution, recording 51 runs of optimization performance.

Currently, we have implemented 7 MetaBBO-RL learnable optimizers, 1 MetaBBO-SL optimizer and 11 BBO optimizers, which are listed in [Baselines](#Baselines). You can also find their implementations in [RELOPS/agent](RELOPS/agent) and [RELOPS/optimizer](RELOPS/optimizer). We have imported all of these agents and optimizers in [tester.py](RELOPS/tester.py) for you to compare, and you are supposed to import your own agent and optimizer in it.

You can use the `--agent_for_cp xxx` option to select the agent(s) for comparison and `--l_optimizer_for_cp xxx` option to select the learnable optimizer(s) for comparison. Please note that the agent needs to support the corresponding learnable optimizer. Additionally, you can use `--t_optimizer_for_cp xxx` to select the traditional optimizer(s) for comparison.  `--agent_load_dir` option specifies the directory that contains the `.pkl` model files of your own agent and all comparing agents, and **the model files should be named after the class name of corresponding agent**, for example, `DE_DDQN_Agent.pkl`. `--log_dir` option specifies the directory where log files will be saved. 

You can test your own agent *MyAgent* and optimizer *MyOptimizer* with DE_DDQN, LDE, DEAP_DE, JDE21, DEAP_CMAES, Random_search using the following command:

```shell
python main.py --agent_load_dir MyAgentLoadDir --agent MyAgent --optimizer MyOptimizer --agent_for_cp DE_DDQN_Agent LDE_Agent --l_optimizer_for_cp DE_DDQN_Optimizer LDDE_Optimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search --log_dir MyLogDir
```

For the above command, `RELOPS` will first load the trained model from *MyAgentLoadDir*, then initialize the agents and optimizers of yours, DE_DDQN and LDE and the selected traditional optimizers, and use the generated test set to optimize all the selected problems for testing. 

### Test Results

After testing, 3 types of data files will be generated in `MyLogDir/test/runName` or `output/test/runName` by default: 

* `test.pkl` is a dictionary containing the following testing data of time complexity and optimization performance, which can be used to generate your own graphs and tables.

  * `T0` is the time of running following computations *max function evaluations* times and is consistent for all algorithms.

    ```python
    x = np.random.rand(dim)
    x + x
    x / (x+2)
    x * x
    np.sqrt(x)
    np.log(x)
    np.exp(x)
    ```

  * `T1` is the evaluation time of the first problem in test set and is consistent for all algorithms.

  * `T2` is the test time of a specific algorithm running on the first problem in test set.

  * `cost` is the best costs sampled every 400 function evaluations along the test process of each algorithm running on each problem for 51 times.

  * `fes` is the function evaluation times used by each algorithm running on each problem for 51 times.

* `.xlsx` files in `MyLogDir/test/runName/tables/`, contains 3 types of excel tables:

  * `algorithm_complexity.xlsx` contains time complexity calculated by `T0`, `T1` and `T2` for each comparing algorithms.
  * `algorithm_name_concrete_performance_table.xlsx` such as *RLEPSO_Agent_concrete_performance_table.xlsx* and *GL_PSO_concrete_performance_table.xlsx*, contains the specific algorithm's performance , i.e., the worst, best, median, mean, std of the costs the optimizer obtained on each problem in test set.
  * `overall_table.xlsx` contains optimization performance of all comparing algorithms on each problem of test set.

* `.png` files in `MyLogDir/test/runName/pics/`, contains 4 types of graphs:

  * `algorithm_name_concrete_performance_hist.png`, such as *RLEPSO_Agent_concrete_performance_hist.png* and *GL_PSO_concrete_performance_hist.png*, draws the performance histogram of the specific algorithm on each problem.
  * `problem_name_cost_curve.png` such as *Schwefel_cost_curve.png*, draws the cost curve of each algorithm's optimization process on the specific problem.
  * `all_problem_cost_curve.png` draws each algorithm's average cost curve on all problems in test set.
  * `rank_hist.png` plots a histogram of each algorithm's score.