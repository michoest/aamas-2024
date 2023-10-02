# RAISE the Bar: Restriction of Action Spaces for Improved Social Welfare and Equity in Traffic Management 
Code package for the reproduction of experimental results

## Structure
### `/experiments`
Contains the experiments `/braess` (Generalized Braess graphs) and `/gnp` (Erdös-Renyi graphs) for the effect of restriction-based and reward-based governance of multi-agent systems in the domain of traffic management.

### `/src`
Contains the definition of the traffic model and the agents (`environment.py`), functions for the analysis and visualization of the results (`analysis.py`) and utility functions (`util.py`).

## Installation
To install the required dependencies, run
```
$ pip install raise-the-bar
```

## Execution
The notebooks in the `/experiments` folder can be simply run from start to end to reproduce the results reported in the paper. Using different seeds will give slightly different results, showing that the claims are robust with respect to randomization.

Any plots are generated in the respective experiment folders.