import glob
import os
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Runs CART with cost complexity tuning over synthetic datasets


seed = 41

def calculate_avg_samples_per_node(tree, total_samples):
    node_samples = tree.tree_.n_node_samples
    is_internal = tree.tree_.children_left != -1
    internal_samples = node_samples[is_internal]
    return np.mean(internal_samples / total_samples) if len(internal_samples) > 0 else 0


def calculate_question_length(tree, total_samples):
    node_samples = tree.tree_.n_node_samples
    depths = np.zeros(tree.tree_.node_count, dtype=int)
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    # Compute depth for each node
    for node in range(1, tree.tree_.node_count):
        parent = np.where((children_left == node) | (children_right == node))[0]
        if parent.size > 0:
            depths[node] = depths[parent[0]] + 1

    # Select only leaf nodes
    is_leaf = (children_left == -1) & (children_right == -1)
    leaf_depth_product = np.sum(node_samples[is_leaf] * depths[is_leaf])

    return leaf_depth_product / total_samples


csv_files = glob.glob("../../instances_synthetic/*.csv")
filenames = [os.path.basename(f) for f in csv_files if "Train" in os.path.basename(f)]

for filename in filenames:

    file_path = "../../instances_synthetic/" + filename
    df = pd.read_csv(file_path)
    df_test = pd.read_csv(file_path.replace("Train", "Test"))

    X_train_full = df.iloc[:, :-1]
    y_train_full = df.iloc[:, -1]

    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    num_samples = X_train_full.shape[0] + X_test.shape[0]
    num_features = X_train_full.shape[1]
    num_classes = y_train_full.max() + 1

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                      test_size=0.25, stratify=y_train_full, random_state=seed)

    clf = DecisionTreeClassifier(random_state=seed)
    # clf = DecisionTreeClassifier(random_state=seed, max_depth=6)      # If using limited depth, use this

    clf.fit(X_train, y_train)

    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]

    best_alpha = None
    best_val_acc = 0
    for alpha in ccp_alphas:
        if alpha < 0:
            continue
        clf_pruned = DecisionTreeClassifier(random_state=seed, ccp_alpha=alpha)
        # clf_pruned = DecisionTreeClassifier(max_depth=6, random_state=seed, ccp_alpha=alpha) # If using limited depth, use this

        clf_pruned.fit(X_train, y_train)
        y_val_pred = clf_pruned.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_alpha = alpha

    clf_final = DecisionTreeClassifier(random_state=seed, ccp_alpha=best_alpha)
    # clf_final = DecisionTreeClassifier(max_depth=6, random_state=seed, ccp_alpha=best_alpha) # If using limited depth, use this

    start_time = time.time()
    clf_final.fit(X_train_full, y_train_full)
    end_time = time.time()
    elapsed_time = end_time - start_time

    y_train_pred = clf_final.predict(X_train_full)
    y_test_pred = clf_final.predict(X_test)

    train_acc = accuracy_score(y_train_full, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    max_depth = clf_final.get_depth()
    avg_samples_per_node = calculate_avg_samples_per_node(clf_final, X_train_full.shape[0])
    internal_nodes = clf_final.tree_.node_count - np.sum(clf_final.tree_.children_left == -1)
    question_length = calculate_question_length(clf_final, X_train_full.shape[0])

    print(f"CART,{filename},,,,{internal_nodes},{num_features},{num_classes},{num_samples},{train_acc},{test_acc},"
          f"{elapsed_time},,{max_depth},{question_length},{avg_samples_per_node},")




