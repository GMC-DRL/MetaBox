## Rollout

### How to Rollout

By using the following command, you can rollout your agent models obtained from training process above using problems in train set: 

```bash
python main.py --rollout --agent_load_dir MyAgentLoadDir --agent_for_rollout MyAgent --optimizer_for_rollout MyOptimizer --log_dir MyLogDir 
```

But before running it, **please make sure that the 21 agent models named `checkpointN.pkl` saved from training process are in a folder named your agent class name *MyAgent*, and this folder is in directory *MyAgentLoadDir***, which seems like:

```
MyAgentLoadDir
│        
└─ MyAgent
    │
    ├─ checkpoint0.pkl
    ├─ checkpoint1.pkl
    ├─ ...
    └─ checkpoint20.pkl
```

### Rollout Results

After rollout, in `MyLogDir/rollout/runName` or `output/rollout/runName` by default, `MetaBox` will generate a file named `rollout.pkl` which is a dictionary containing:

* `cost` is the best costs sampled every 400 function evaluations along the rollout process of each checkpoint model running on each problem in train set.
* `fes` is the function evaluation times used by each checkpoint model running on each problem in train set.
* `return` is the total reward in the rollout process of each checkpoint model running on each problem in train set.