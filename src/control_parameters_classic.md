***
#### Control paramters for JDE21
| Parameter   | Value               | Description                                            |
|-------------|---------------------|--------------------------------------------------------|
| $sNP$       | 10                  | size of small population                               |
| $bNP$       | 160                 | size of big population                                 |
| $F_{l,b}$   | 0.1                 | lower limit of scale factor of big population          |
| $F_{l,s}$   | 0.17                | lower limit of scale factor of small population        |
| $F_u$       | 1.1                 | upper limit of scale factor                            |
| $CR_{l,b}$  | 0.0                 | lower limit of crossover rate for big population       |
| $CR_{l,s}$  | 0.1                 | lower limit of crossover rate for small population     |
| $CR_{u,b}$  | 1.1                 | upper limit of crossover rate for big population       |
| $CR_{u,s}$  | 0.8                 | upper limit of crossover rate for small population     |
| $F_{init}$  | 0.5                 | initial value of scale factor                          |
| $CR_{init}$ | 0.9                 | initial value of crossover rate                        |
| $\tau_1$    | 0.1                 | probability to self-adapt scale factor                 |
| $\tau_2$    | 0.1                 | probability to self-adapt crossover rate               |
| $ageLmt$    | $\frac{maxFEs}{10}$ | reinitialize threshold for big population              |
| $eps$       | $10^{-8}$           | convergence precision                                  |
| $myEqs$     | 25                  | reinitialization if $myEqs\%$ individuals are similar. |

***
#### Control paramters for MadDE
| Parameter  | Value | Description                                 |
|------------|-------|---------------------------------------------|
| $P_{qBX}$  | 0.01  | probability of qBX crossover                |
| $p$        | 0.18  | percentage of population in p-best mutation |
| $A_{rate}$ | 2.3   | Archive size multiplier                     |
| $H_m$      | 10    | Memory size multiplier                      |
| $NP_m$     | 2     | initial population size size multiplier     |
| $F_0$      | 0.2   | initial value for scale factor memory       |
| $CR_0$     | 0.2   | initial value for crossover rate memory     |

***
#### Control paramters for NL-SHADE-LBC
| Parameter       | Value                   | Description                                                         |
|-----------------|-------------------------|---------------------------------------------------------------------|
| $NP_{max}$      | $23\times problem dim$ | initial population size                                             |
| $NP_{min}$      | 4                       | final population size                                               |
| $N_A$           | $NP_{max}$              | initial Archive size                                                |
| $H$             | $20\times problem dim$ | historical memory size                                              |
| $p_a$           | 0.5                     | probability of selecting individule from Archive to mutate          |
| $p_b$           | 0.4                     | proportion of best individuals                                      |
| $m$             | 1.5                     | linear bias change for scale factor and crossover rate              |
| $p_{F}^{init}$  | 3.5                     | initial parameter values in weighted Lehmer mean for scale factor   |
| $p_{CR}^{init}$ | 1.0                     | initial parameter values in weighted Lehmer mean for crossover rate |
| $p^{final}$     | 1.5                     | final parameter values in weighted Lehmer mean                      |
| $CR_{init}$     | 0.9                     | initial value of crossover rate                                     |
| $F_{init}$      | 0.5                     | initial value of scale factor                                       |


***
#### Control paramters for GLPSO
| Parameter       | Value   | Description                                 |
|-----------------|---------|---------------------------------------------|
| $NP$            | 50      | population size                             |
| $w$             | 0.7298  | inertia weight                              |
| $c$             | 1.49618 | accelerate coefficient                      |
| $p_m$           | 0.01    | mutation probability                        |
| $sg$            | 7       | stopping gap before selecting best exampler |
| $N_{selection}$ | 10      | proportion of best examplers                |
| $\rho$          | 0.2     | the maximum velocity multiplier             |

***
#### Control paramters for sDMSPSO
| Parameter   | Value                | Description                                   |
|-------------|----------------------|-----------------------------------------------|
| $w$         | $0.729$              | inertia weight                                |
| $c_1,c_2$   | 1.49445              | accelerate coefficients                       |
| $m$         | 3                    | niching swarm size                            |
| $NP$        | 99                   | population size                               |
| $R$         | 10                   | regrouping period                             |
| $LP$        | 10                   | learning period                               |
| $LA$        | 8                    | length of Archive                             |
| $L$         | 100                  | local refining period                         |
| $L_{FEs}$   | 200                  | max fitness evaluations using in local search |
| $w_{decay}$ | True                 | inertia weight linearly decays                |
| $max_v$     | $0.1 \times (ub-lb)$ | maximum permitted velocity                    |

***
#### Control paramters for SAHLPSO
| Parameter | Value                                           | Description                             |
|-----------|-------------------------------------------------|-----------------------------------------|
| $NP$      | 40                                              | population size                         |
| $max_v$   | $0.1 \times (ub-lb)$                            | maximum permitted velocity              |
| $H_{CR}$  | 5                                               | initial length of crossover rate memory |
| $M_{CR}$  | $[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]$ | candidates of crossover rate            |
| $M_{ls}$  | $range(1,16)$                                   | candidates of learning step             |
| $LP$      | 5                                               | learning period                         |
| $Lg$      | 0.2                                             | ratio of exploration particles          |
| $p$       | 0.2                                             | selection ratio for donor vector        |
| $c$       | 1.49445                                         | accelerate coefficient                  |
| $w$       | $randc(0.7,0.1), randc(0.3,0.1)$                | inertia weight                          |



