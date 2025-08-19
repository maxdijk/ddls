from dataset import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from topology import Topology
from heuristic import Heuristic
from optimizer import Optimizer
from visualizer import Visualizer
import time
import numpy as np
import pandas as pd


# Runs the MILP approach with hyperparameter tuning for the 8 datasets.


def get_size(solution):
    fragmentation = solution.fragmentation_per_node()

    internal_nodes = 0
    for key, value in fragmentation.items():
        if key < sum(skeleton):
            internal_nodes += 1

    return internal_nodes

def get_average_fragmentation(solution, skeleton):
    total = 0.0
    fragmentation = solution.fragmentation_per_node()

    internal_nodes = 0
    for key, value in fragmentation.items():
        if key < sum(skeleton):
            internal_nodes += 1
            total += value

    return total / internal_nodes

def get_depth(cur_node, left_dict, right_dict):
    if (cur_node not in left_dict) and (cur_node not in right_dict):
        return 0

    left_depth = get_depth(left_dict.get(cur_node), left_dict, right_dict)
    right_depth = get_depth(right_dict.get(cur_node), left_dict, right_dict)

    return max(left_depth, right_depth) + 1


def question_length(solution, training_data, skeleton):

    total_depth = 0
    for sample in training_data:
        depth = 0
        cur_node = solution.topology.root_node
        while cur_node < sum(skeleton):
            value = np.dot(solution.node_hyperplane[cur_node], sample)
            depth += 1

            if value < solution.node_intercept[cur_node]:
                cur_node = solution.node_negative_arc[cur_node]
            else:
                cur_node = solution.node_positive_arc[cur_node]

        total_depth += depth

    return total_depth / len(training_data)


skeletons = [[1,2,4,8],[1,2,4,4,4],[1,2,3,3,3,3],[1,2,2,2,2,2,2,2]]
alphas = [0.01, 0.1, 0.2, 0.5, 1.0]

best_val_acc = -1
best_skeleton = None
best_alpha = None

filenames = [ "soybean-small.csv" ,"echocardiogram.csv", "hepatitis.csv",
             "iris.csv", "thyroid-new.csv", "acute-inflammations-nephritis.csv", "acute-inflammations-urinary.csv", "wine.csv"]

for filename in filenames:
    file_path = "../../../instances/" + filename
    filename = filename[:-4]    # remove .csv
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Further split training set into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.25, stratify=y_train_full, random_state=41
        )

        train_full_df = pd.concat([X_train_full, y_train_full], axis=1)
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        data = Dataset(file_path)
        data.setup_train_test(train_df, val_df)     # Use train and validation set
        for skeleton in skeletons:
            for alpha in alphas:
                topology = Topology(skeleton, data, long_arcs=True)
                heuristic = Heuristic(data, topology, alpha=alpha)
                optimized = Optimizer(data, topology, alpha=alpha, initial_solution=heuristic.solution, univariate_split=True)

                if (optimized.solution.test_accuracy() > best_val_acc):
                    best_skeleton = skeleton
                    best_alpha = alpha
                    best_val_acc = optimized.solution.test_accuracy()

        data = Dataset(file_path)
        data.setup_train_test(train_full_df, test_df)     # Use full training data and test set

        topology = Topology(best_skeleton, data, long_arcs=True)

        num_features = data.p
        num_samples = data.n
        num_classes = len(data.classes)

        start = time.time()
        heuristic = Heuristic(data, topology, alpha=best_alpha)
        optimized = Optimizer(data, topology, alpha=best_alpha, initial_solution=heuristic.solution, univariate_split=True)
        end = time.time()

        size = get_size(optimized.solution)
        avg_fragmentation = get_average_fragmentation(optimized.solution, best_skeleton)
        depth = get_depth(optimized.solution.topology.root_node, optimized.solution.node_negative_arc, optimized.solution.node_positive_arc)
        question_l = question_length(optimized.solution, data.train_X, best_skeleton)

        if optimized.solution == heuristic.solution:
            print("MILP,{},heuristic,,,{},{},{},{},{},{},{},,{},{},{},".format(filename, size, num_features,
                                                                                 num_classes, num_samples,
                                                                                 optimized.solution.training_accuracy(),
                                                                                 optimized.solution.test_accuracy(),
                                                                                 end - start,
                                                                                 depth, question_l, avg_fragmentation))
        else:
            print("MILP,{},optimized,,,{},{},{},{},{},{},{},,{},{},{},".format(filename, size, num_features, num_classes, num_samples,
                                                                     optimized.solution.training_accuracy(),
                                                                     optimized.solution.test_accuracy(), end - start,
                                                                     depth, question_l, avg_fragmentation))


        best_alpha = None
        best_skeleton = None
        best_val_acc = -1
