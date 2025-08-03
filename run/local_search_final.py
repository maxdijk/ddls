import os
import glob
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import heuristic

seed = 41

def objective_SA(params, X_train, filename_train, filename_test):
    alpha, width, depth, cooling, start_temp = params

    structure = heuristic.get_structure(width, depth)
    heuristic.initialize_diagram(X_train, structure)                    # Remove for random initialization
    heuristic.save_diagram(structure, True)                             # Use False for random initialization

    command = [
        "../bin/bdd.exe",
        '--SA1::cooling_rate', str(cooling),
        '--SA1::max_evaluations', '50000',
        '--SA1::start_temperature', str(start_temp),
        '--SA1::min_temperature', '0.0001',
        '--main::method', 'SA',
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




def run_SA(X_train_full, X_train, X_val, X_test, filename):
    # Wrapping function for the bayesian search using the training and validation set
    def wrapped_objective_SA(params):
        return objective_SA(params, X_train, f"{filename}_train.csv", f"{filename}_val.csv")

    # Define search space for hyperparameters
    search_space = [
        Real(1e-6, 0.1, "log-uniform"),     # Alpha
        Categorical([2, 4, 6, 8, -1]),      # Structure type
        Integer(2, 10, "uniform"),          # Structure depth
        Real(0.8, .995, "uniform"),         # Cooling rate
        Integer(60, 100, "uniform")         # Starting temp
    ]

    # Get best hyperparameters using Bayesian search
    result = gp_minimize(wrapped_objective_SA, search_space, n_calls=30, random_state=seed)

    best_alpha, best_width, best_depth, best_cooling, best_start_temp = result.x
    best_accuracy = -result.fun

    print(f"Best alpha: {best_alpha}, Best width: {best_width}, Best depth: {best_depth}, best cooling :{best_cooling}, best start temp: {best_start_temp}")
    print(f"Best Accuracy: {best_accuracy}")

    # Get the results using the full training and test set
    objective_SA([best_alpha, best_width, best_depth, best_cooling, best_start_temp], X_train_full, f"{filename}_train_full.csv", f"{filename}_test.csv")




def objective_ILS(params, X_train, filename_train, filename_test):
    alpha, width, depth, perts, iterations = params

    structure = heuristic.get_structure(width, depth)
    heuristic.initialize_diagram(X_train, structure)                    # Remove for random initialization
    heuristic.save_diagram(structure, True)                             # Use False for random initialization

    num_eval = int(50000 / iterations)
    command = [
        "../bin/bdd.exe",
        "--HC1::max_evaluations", str(num_eval),
        "--HC1::max_idle_iterations", "10000",
        '--main::ILS_perturbations', str(perts),
        '--main::ILS_iterations', str(iterations),
        '--main::method', 'ILS',
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



def run_ILS(X_train_full, X_train, X_val, X_test, filename):
    # Wrapping function for the bayesian search using the training and validation set
    def wrapped_objective_ILS(params):
        return objective_ILS(params, X_train, f"{filename}_train.csv", f"{filename}_val.csv")

    # Define search space for hyperparameters
    search_space = [
        Real(1e-6, 0.1, "log-uniform"),     # Alpha
        Categorical([2, 4, 6, 8, -1]),      # Structure type
        Integer(2, 10, "uniform"),          # Structure depth
        Integer(2, 10),                     # Number of perturbations
        Integer(2, 5)                       # Number of ILS iterations
    ]

    # Get best hyperparameters using Bayesian search
    result = gp_minimize(wrapped_objective_ILS, search_space, n_calls=30, random_state=seed)

    best_alpha, best_width, best_depth, best_perts, best_iterations = result.x
    best_accuracy = -result.fun

    print(f"Best alpha: {best_alpha}, Best width: {best_width}, Best depth: {best_depth}, Best perturbations: {best_perts}, Best iterations: {best_iterations}")
    print(f"Best Accuracy: {best_accuracy}")

    # Get the results using the full training and test set
    objective_ILS([best_alpha, best_width, best_depth, best_perts, best_iterations], X_train_full, f"{filename}_train_full.csv", f"{filename}_test.csv")



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
        return objective_HC(params, X_train, f"{filename}_train.csv", f"{filename}_val.csv")

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
    objective_HC([best_alpha, best_width, best_depth], X_train_full, f"{filename}_train_full.csv", f"{filename}_test.csv")


csv_files = glob.glob("../instances/*.csv")
filenames = [os.path.basename(f) for f in csv_files]

for filename in filenames:
    file_path = "../instances/" + filename
    filename = filename[:-4]    # remove .csv
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Further split training set into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                          test_size=0.25, stratify=y_train_full, random_state=seed)

        X_train_full = np.column_stack((X_train_full, y_train_full))
        X_train = np.column_stack((X_train, y_train))
        X_val = np.column_stack((X_val, y_val))
        X_test = np.column_stack((X_test, y_test))

        np.savetxt(f"../temp/{filename}_train_full.csv", X_train_full, delimiter=",")
        np.savetxt(f"../temp/{filename}_train.csv", X_train, delimiter=",")
        np.savetxt(f"../temp/{filename}_val.csv", X_val, delimiter=",")
        np.savetxt(f"../temp/{filename}_test.csv", X_test, delimiter=",")

        run_HC(X_train_full, X_train, X_val, X_test, filename)
        #run_SA(X_train_full, X_train, X_val, X_test, filename)
        #run_ILS(X_train_full, X_train, X_val, X_test, filename)
