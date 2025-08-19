import glob
import os
import time

import numpy as np
import tensorflow as tf
from TreeInTree import TnT
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold


# Run tnt with hyperparameter tuning on the synthetic datasets

seed = 41

def get_average_fragmentation(tnt, data):
    nodes, mask = tnt.graph_traverse(teX=data, node=tnt.graph)

    internal, l = tnt.check_complexity()
    if internal == 0:
        return 1.0

    total = 0.0
    for i in range(len(nodes)):
        node = nodes[i]
        if node.label is None:
            total += np.sum(mask[i]) / len(data)

    return total / internal

def get_depth(node):
    # Return leaf nodes and their depth
    if node.label is not None:
        return 0
    else:
        return max(get_depth((node.left)), get_depth(node.right)) + 1

def get_question_length(tnt, data):
    nodes, mask = tnt.graph_traverse(teX=data, node=tnt.graph)

    total = 0.0
    for i in range(len(nodes)):
        node = nodes[i]
        if node.label is None:
            total += sum(mask[i])

    return total / len(data)



csv_files = glob.glob("../../../instances_synthetic/*.csv")
filenames = [os.path.basename(f) for f in csv_files if "Train" in os.path.basename(f)]


for filename in filenames:
    file_path = "../../../instances_synthetic/" + filename
    data = np.loadtxt(file_path, delimiter=',',skiprows=1)

    X_train_full = np.loadtxt(file_path, delimiter=',',skiprows=1)[:, :-1]
    y_train_full = np.loadtxt(file_path, delimiter=',',skiprows=1)[:, -1]

    X_test = np.loadtxt(file_path.replace("Train", "Test"), delimiter=',',skiprows=1)[:, :-1]
    y_test = np.loadtxt(file_path.replace("Train", "Test"), delimiter=',',skiprows=1)[:, -1]

    num_samples = X_train_full.shape[0] + X_test.shape[0]
    num_features = X_train_full.shape[1]
    num_classes = y_train_full.max() + 1

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                      test_size=0.25, stratify=y_train_full, random_state=seed)

    start = np.log10(5e-5)
    stop = np.log10(1e-1)
    log_space = np.logspace(start, stop, num=30)  # Create 30 alpha values between start and stop for tuning

    best_alpha = None
    best_val_acc = 0
    for alpha in log_space:

        tnt = TnT(N1=2, N2=5, ccp_alpha=alpha, random_state=seed)
        tnt.fit(X_train, y_train)

        prediction_test = tnt.predict(teX=X_val)
        val_acc = accuracy_score(y_val, prediction_test)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_alpha = alpha

    tnt = TnT(N1=2, N2=5, ccp_alpha=best_alpha, random_state=seed)

    start_time = time.time()
    tnt.fit(X_train_full, y_train_full)
    end_time = time.time()
    elapsed_time = end_time - start_time

    prediction_train = tnt.predict(teX=X_train_full)
    accuracy_train = accuracy_score(y_train_full, prediction_train)

    prediction_test = tnt.predict(teX=X_test)
    accuracy_test = accuracy_score(y_test, prediction_test)

    i, l = tnt.check_complexity()
    depth = get_depth(tnt.graph)
    question_length = get_question_length(tnt, X_train_full)
    fragmentation = get_average_fragmentation(tnt, X_train_full)

    print(f"TNT,{filename},,,,{i},{num_features},{num_classes},{num_samples},{accuracy_train},{accuracy_test},"
          f"{elapsed_time},,{depth},{question_length},{fragmentation},")
