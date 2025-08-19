import glob
import os
import time
from interpretableai import iai
import numpy as np



# Runs IAI on synthetic datasets

seed = 41

csv_files = glob.glob("../../instances_synthetic/*.csv")
filenames = [os.path.basename(f) for f in csv_files if "Train" in os.path.basename(f)]


for filename in filenames:
    file_path = "../../instances_synthetic/" + filename
    data = np.loadtxt(file_path, delimiter=',',skiprows=1)

    X_train = np.loadtxt(file_path, delimiter=',',skiprows=1)[:, :-1]
    y_train = np.loadtxt(file_path, delimiter=',',skiprows=1)[:, -1]

    X_test = np.loadtxt(file_path.replace("Train", "Test"), delimiter=',',skiprows=1)[:, :-1]
    y_test = np.loadtxt(file_path.replace("Train", "Test"), delimiter=',',skiprows=1)[:, -1]

    num_samples = X_train.shape[0] + X_test.shape[0]
    num_features = X_train.shape[1]
    num_classes = y_train.max() + 1

    grid = iai.GridSearch(
        iai.OptimalTreeClassifier(
            random_seed=seed,
        ),
        max_depth=6,
    )

    # Run grid search and get the best alpha it found
    grid.fit(X_train, y_train)
    best_alpha = grid.get_learner().get_params()['cp']

    # Create new learner with best alpha found
    lnr = iai.OptimalTreeClassifier(max_depth=6, cp=best_alpha, random_seed=seed)

    start_time = time.time()
    lnr.fit(X_train, y_train)
    runtime = time.time() - start_time

    train_acc = lnr.score(X_train, y_train, criterion='misclassification')
    test_acc = lnr.score(X_test, y_test, criterion='misclassification')

    num_nodes = lnr.get_num_nodes()
    decision_nodes = []
    for i in range(1, num_nodes + 1):
        if not lnr.is_leaf(i):
            decision_nodes.append(i)

    max_depth = 0
    avg_samples_per_node = 0
    for node in decision_nodes:
        max_depth = max(max_depth, lnr.get_depth(node))
        avg_samples_per_node += (lnr.get_num_samples(node) / len(X_train)) / len(decision_nodes)

    path_matrix = np.array(lnr.decision_path(X_train))
    avg_decision_nodes_per_sample = (path_matrix.sum(axis=1) - 1).mean()

    print(
        f"IAI,{filename},,,,{len(decision_nodes)},{num_features},{num_classes},{num_samples},{train_acc},{test_acc},"
        f"{runtime},,{max_depth},{avg_decision_nodes_per_sample},{avg_samples_per_node},")

