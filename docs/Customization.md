## Fit your own MetaBBO-RL into MetaBox

If you want to develop your own MetaBBO-RL approach, to fit into `MetaBox` running logic, you should meet with the following protocol about the `Agent` and `Optimizer`. 

`Agent` is the same definition in RL area, taking the state from `env` as input and `action` as output. But to fit into MetaBox pre-defined `Trainer` and `Tester` calling logic, `Agent` should has `train_episode` interface which will be called in `Trainer` and `rollout_episode` interface which will be called in `Tester`. 

`Optimizer` is a component of `env` in MetaBBO task. It's controlled by `Agent` and take `action` from `Agent` to perfrom corresponding change like hyper-parameters adjusting or operators selection. But to fit into `env` calling logic. Interfaces namely `init_population` and `update` is needed.

* Your agent should follow this template:

  ```python
  from agent.basic_agent import Basic_Agent
  from agent.utils import save_class
  
  class MyAgent(Basic_Agent):
      def __init__(self, config):
          """
          Parameter
          ----------
          config: An argparse. Namespace object for passing some core configurations such as max_learning_step.
  
          Must To Do
          ----------
          1. Save the model of initialized agent, which will be used in "rollout" to study the training process.
          2. Initialize a counter to record the number of accumulated learned steps
          3. Initialize a counter to record the current checkpoint of saving agent
          """
          super().__init__(config)
          self.config = config
          save_class(self.config.agent_save_dir, 'checkpoint0', self)  # save the model of initialized agent.
          self.learned_steps = 0   # record the number of accumulated learned steps
          self.cur_checkpoint = 1  # record the current checkpoint of saving agent
          """
          Do whatever other setup is needed
          """
          
  	def get_action(self, state):
          """
          Parameter
          ----------
          state: state features defined by developer.
          
          Return
          ----------
          action: the action inferenced by using state.
          """
          
      def train_episode(self, env):
          """ Called by Trainer.
              Optimize a problem instance in training set until reaching max_learning_step or satisfy the convergence condition.
              During every train_episode,you need to train your own network.
  
          Parameter
          ----------
          env: an environment consisting of a backbone optimizer and a problem sampled from train set.
  
          Must To Do
          ----------
          1. record total reward
          2. record current learning steps and check if reach max_learning_step
          3. save agent model if checkpoint arrives
  
          Return
          ----------
          A boolean that is true when fes reaches max_learning_step otherwise false
          A dict: {'normalizer': float,
                   'gbest': float,
                   'return': float,
                   'learn_steps': int
                   }
          """
          state = env.reset()
          R = 0  # total reward
          """
          begin loop：
          """
              action = self.get_action(state)
              next_state, reward, is_done = env.step(action) # feed the action to environment
              R += reward  # accumulate reward
              """
              perform update strategy of agent, which is defined by you. Every time update your agent, please increase self.learned_step accordingly
              """
  
              # save agent model if checkpoint arrives
              if self.learned_steps >= (self.config.save_interval * self.cur_checkpoint):
                  save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
                  self.cur_checkpoint += 1
  
              state = next_state
  
              """
              check if finish loop
              """
          return self.learned_steps >= self.config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                                       'gbest': env.optimizer.cost[-1],
                                                                       'return': R,
                                                                       'learn_steps': self.learned_steps}
  
      def rollout_episode(self, env):
          """ Called by method rollout and Tester.test
  
          Parameter
          ----------
          env: an environment consisting of a backbone optimizer and a problem sampled from test set
  
          Return
          ----------
          A dict: {'cost': list, 
                   'fes': int, 
                   'return': float
                   }
          """
          state = env.reset()
          is_done = False
          R = 0  # total reward
          while not is_done:
              action = self.get_action(state)
              next_state, reward, is_done = env.step(action) # feed the action to environment
              R += reward  # accumulate reward
              state = next_state
  
          return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes, 'return': R}
  ```

* Your backbone optimizer should follow this template:

  ```python
  from optimizer.learnable_optimizer import Learnable_Optimizer
  
  class MyOptimizer(Learnable_Optimizer):
      def __init__(self, config):
          """
          Parameter
          ----------
          config: An argparse.Namespace object for passing some core configurations such as maxFEs.
          """
          super().__init__(config)
          self.config = config
          """
          Do whatever other setup is needed
          """
  
      def init_population(self, problem):
          """ Called by method PBOEnv.reset.
              Init the population for optimization.
  
          Parameter
          ----------
          problem: a problem instance, you can call `problem.eval` to evaluate one solution.
  
          Must To Do
          ----------
          1. Initialize a counter named "fes" to record the number of function evaluations used.
          2. Initialize a list named "cost" to record the best cost at logpoints.
          3. Initialize a counter to record the current logpoint.
  
          Return
          ----------
          state: state features defined by developer.
          """
  
          """
          Initialize the population, calculate the cost using method problem.eval and renew everything (such as some records) that related to the current population.
          """
          self.fes = self.population_size  # record the number of function evaluations used
          self.cost = [self.best_cost]     # record the best cost of first generation
          self.cur_logpoint = 1            # record the current logpoint
          """
          calculate the state
          """
          return state
  
      def update(self, action, problem):
          """ update the population using action and problem.
              Used in Environment's step
  
          Parameter
          ----------
          action: the action inferenced by agent.
          problem: a problem instance.
  
          Must To Do
          ----------
          1. Update the counter "fes".
          2. Update the list "cost" if logpoint arrives.
  
          Return
          ----------
          state: represents the observation of current population.
          reward: the reward obtained for taking the given action.
          is_done: whether the termination conditions are met.
          """
  
          """
          update population using the given action and update self.fes
          """
          # append the best cost if logpoint arrives
          if self.fes >= self.cur_logpoint * self.config.log_interval:
              self.cur_logpoint += 1
              self.cost.append(self.best_cost)
          """
          get state, reward and check if it is done
          """
          if is_done:
              if len(self.cost) >= self.config.n_logpoint + 1:
                  self.cost[-1] = self.best_cost
              else:
                  self.cost.append(self.best_cost)
          return state, reward, is_done
  ```

By the way, if you are developing classic optimizer, please refer to [example classic optimizer](../src/optimizer/deap_de.py).

After that, you should put your own declaring files in directory `src/agent/` and `src/optimizer/` respectively. Then the file structure should be like:

```
src
│        
├─ agent
│   │
│   ├─ de_ddqn_agent.py
│   ├─ ...
│   ├─ rlepso_agent.py
│   └─ my_agent.py
└─ optimizer
    │
    ├─ dq_ddqn_optimizer.py
    ├─ ...
    ├─ rlepso_optimizer.py
    └─ my_optimizer.py
```

In addition, you should register you own agent and backbone optimizer in files `src/agent/__init__.py` and `src/optimizer/__init__.py`. For example, to register the previous class *MyAgent*, you should add one line into the `src/agent/__init__.py` file as below:

```python
from .my_agent import *
```

Meanwhile, you should also import your own agent and backbone optimizer into `src/trainer.py` and `src/tester.py`. Take trainer as an example, you should add two lines into file `src/trainer.py` as follows:

```python
...
# import your agent
from agent import{
     ...
     MyAgent
}
# import your optimizer
from optimizer import{
     ...
     MyOptimizer
}
```

The same action should be done also in `src/tester.py`.

As mentioned, four modes are available:

* `run_experiment` your MetaBBO-RL optimizer.

  `run_experiment` mode implements fully automated workflow. Assume that you've written an agent class named *MyAgent* and a backbone optimizer class named *MyOptimizer*, the entire processes of train, rollout and test can be triggered by running command:

  ```shell
  python main.py --run_experiment --train_agent MyAgent --train_optimizer MyOptimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search
  ```

  See [Run Experiment](RunExperiment.md) for more details.

* `train` your agent.

  ```shell
  python main.py --train --train_agent MyAgent --train_optimizer MyOptimizer
  ```

  Once you run the above command, the `runName` which is generated based on the run time and benchmark suite will appear at the command line. 

  See [Training](Train.md) for more details.

* `rollout` your agent models.

  Fetch your 21 trained agent models named `checkpointN.pkl` in directory `src/agent_model/train/MyAgent/runName/` and move them to directory `src/agent_model/rollout/MyAgent/`. Rollout the models with train set using:

  ```shell
  python main.py --rollout --agent_load_dir agent_model/rollout/ --agent_for_rollout MyAgent --optimizer_for_rollout MyOptimizer
  ```

  When the rollout ends, check the result data in `src/output/rollout/runName/rollout.pkl` and pick the best model to test.

  See [Rollout](Rollout.md) for more details.

* `test` your MetaBBO-RL optimizer.

  Move the best `.pkl` model file to directory `src/agent_model/test/`, and rename the file to `MyAgent.pkl`. Now use the test set to test `MyAgent` with `DEAP_DE`, `JDE21`, `DEAP_CMAES` and `Random_search`:

  ```shell
  python main.py --test --agent_load_dir agent_model/test/ --agent_for_cp MyAgent --l_optimizer_for_cp MyOptimizer --t_optimizer_for_cp DEAP_DE JDE21 DEAP_CMAES Random_search
  ```

  See [Testing](Test.md) for more details.

Notice that we record 21 checkpoint models during the whole training process. `Rollout` could help you pick the suitable or best model to test and calculate  the metrics.