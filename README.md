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

We use **Bayesian Optimization** for hyperparameter tuning.  

- [Local Search with Hyperparameter Tuning](run/local_search_final.py) – Runs DDLS with hyperparameter tuning on all UCI datasets.  
- [Decision Tree-limited DDLS](run/local_search_tree.py) – Runs DDLS with hyperparameter tuning on all UCI datasets while restricting the structure to decision trees instead of diagrams.



### Synthetic Data

Synthetic datasets can be generated using:

- [Dataset Generator](run/create_synthetic_datasets.py) – Produces separate training and test datasets from decision diagrams.

You can run DDLS on these datasets using:

- [Normal DDLS](/run/local_search_synthetic.py/)
- [DDLS with Random Restarts](/run/local_search_synthetic_restarts.py/)

### Ohter Methods
Some other methods are also included:
#### [MILP](https://github.com/vidalt/Decision-Diagrams)
- [Training Accuracy](other/MILP/src/run_train_acc.py) – Computes training accuracy across all UCI datasets.  
- [Hyperparameter Tuning](other/MILP/src/run_test_acc.py) – Performs hyperparameter tuning on 8 UCI datasets for which the optimal solution can be found.

#### [Tree in Tree (TnT)](https://github.com/BingzhaoZhu/TnTDecisionGraph)
- [UCI Datasets](other/TnT/TreeInTree/tnt_run.py)
- [Synthetic Datasets](other/TnT/TreeInTree/tnt_run_synthetic.py)

#### CART
- [UCI Datasets](other/CART/cart_cc.py)
- [Synthetic Datasets](other/CART/cart_cc_synthetic.py)

#### [Interpretable AI (IAI)](https://www.interpretable.ai/)
- [UCI Datasets](other/IAI/iai_run.py)
- [Synthetic Datasets](other/IAI/iai_run_synthetic.py)

### Output Format

The results produced by the C++ local search component are structured as follows:
```
solver,filename,metaheuristic,best_objective,initialization_method,diagram_size,num_features,num_classes,num_samples,train_acc,test_acc,runtime,last_improvement,depth,question_length,average_fragmentation
```

## Data Sources

This project uses datasets retrieved from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

The datasets are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
