import os
import glob
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import heuristic

# Runs the synthetic datasets in ../instances_synthetic/

seed = 41

def objective_HC(params, X_train, filename_train, filename_test):
    alpha, width, depth = params

    structure = heuristic.get_structure(width, depth)
    heuristic.initialize_diagram(X_train, structure)                    # Remove for random initialization
    heuristic.save_diagram(structure, True)                             # Use False for random initialization

    command = [
        "../bin/bdd.exe",
        "--HC1::max_evaluations", "50000",
        "--HC1::max_idle_iterations", "10000",
        '--main::method', 'HC',
        '--main::initialization', "greedy",         # greedy for information gain initialization or random for random
        '--main::alpha', str(alpha),
        '--main::filename_train', filename_train,
        '--main::filename_test', filename_test,
        '--main::seed', str(seed),
        '--main::random_move', '0.35',
        '--main::optimal_move', '0.6',
        '--main::edge_move', '0.05'
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # print("Full Output:", result.stdout)

        output_lines = result.stdout.strip().split("\n")
        target_line = output_lines[-1]
        values = target_line.split(",")
        print(target_line)
        test_accuracy = float(values[10])
        return -test_accuracy
    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e)


def run_HC(X_train_full, X_train, X_val, X_test, filename):
    # Wrapping function for the bayesian search using the training and validation set
    def wrapped_objective_HC(params):
        return objective_HC(params, X_train, f"{filename}train.csv", f"{filename}val.csv")

    # Define search space for hyperparameters
    search_space = [
        Real(1e-6, 0.1, "log-uniform"),     # Alpha
        Categorical([2, 4, 6, 8, -1]),      # Structure type
        Integer(2, 10, "uniform"),          # Structure depth
    ]

    # Get best hyperparameters using Bayesian search
    result = gp_minimize(wrapped_objective_HC, search_space, n_calls=30, random_state=seed)

    best_alpha, best_width, best_depth = result.x
    best_accuracy = -result.fun

    print(f"Best alpha: {best_alpha}, Best width: {best_width}, Best depth: {best_depth}")
    print(f"Best Accuracy: {best_accuracy}")

    # Get the results using the full training and test set
    objective_HC([best_alpha, best_width, best_depth], X_train_full, f"{filename}train_full.csv", f"{filename}test.csv")


csv_files = glob.glob("../instances_synthetic/*.csv")
filenames = [os.path.basename(f) for f in csv_files if "Train" in os.path.basename(f)]

for filename in filenames:
    file_path = "../instances_synthetic/" + filename
    filename = filename[:-9]    # remove .csv
    df_train = pd.read_csv(file_path)

    X_train_full = df_train.iloc[:, :-1]
    y_train_full = df_train.iloc[:, -1]

    # Further split training set into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                      test_size=0.25, stratify=y_train_full, random_state=seed)

    X_train_full = np.column_stack((X_train_full, y_train_full))
    X_train = np.column_stack((X_train, y_train))
    X_val = np.column_stack((X_val, y_val))
    X_test = pd.read_csv("../instances_synthetic/" + filename + "Test.csv")

    np.savetxt(f"../temp/{filename}train_full.csv", X_train_full, delimiter=",")
    np.savetxt(f"../temp/{filename}train.csv", X_train, delimiter=",")
    np.savetxt(f"../temp/{filename}val.csv", X_val, delimiter=",")
    np.savetxt(f"../temp/{filename}test.csv", X_test, delimiter=",")

    run_HC(X_train_full, X_train, X_val, X_test, filename)
