## Code for Master's Thesis: Learning Decision Diagrams for Classification via Local Search

This repository contains the code developed for the Master's thesis on learning decision diagrams for classification using local search methods.

### Overview

The project is divided into two main components:

- **C++ Implementation**: The core local search algorithms are implemented in C++ using the [EasyLocal++](https://github.com/iolab-uniud/easylocal) framework, which supports solving combinatorial optimization problems through local search.
  
- **Python Interface**: Python is used for data preparation, visualization, and hyperparameter tuning. It acts as a bridge to call the C++ local search code and manage experimentation workflows.

### Local Search Methods

We implement and evaluate three metaheuristics:

- [Hill Climbing](/run/example_HC.py/)
- [Simulated Annealing](/run/example_SA.py/)
- [Iterated Local Search](/run/example_ILS.py/)

Each script above demonstrates how to run the respective method.

### Hyperparameter Tuning

For hyperparameter tuning, we use Bayesian Optimization. An example script demonstrating this setup can be found here:

- [Local Search with Hyperparameter Tuning](/run/local_search_final.py/)

### Output Format

The results produced by the C++ local search component are structured as follows:
```
solver,filename,metaheuristic,best_objective,initialization_method,diagram_size,num_features,num_classes,num_samples,train_acc,test_acc,runtime,last_improvement,depth,question_length,average_fragmentation
```

## Data Sources

This project uses datasets retrieved from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

The datasets are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
