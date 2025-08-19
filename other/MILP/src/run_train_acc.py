import glob
import os

from dataset import Dataset
from topology import Topology
from heuristic import Heuristic
from optimizer import Optimizer
from visualizer import Visualizer
import time
import numpy as np
import pandas as pd


# Runs the MILP approach with fixed structure and alpha=0 on all UCI datasets

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


skeleton = [1,2,4,4,4]
alpha = 0.0


csv_files = glob.glob("../../../instances/*.csv")
filenames = [os.path.basename(f) for f in csv_files]

for filename in filenames:
    file_path = "../../../instances/" + filename
    filename = filename[:-4]    # remove .csv
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    data = Dataset(file_path)
    data.setup_train_only()

    topology = Topology(skeleton, data, long_arcs=False)

    num_features = data.p
    num_samples = data.n
    num_classes = len(data.classes)

    start = time.time()
    heuristic = Heuristic(data, topology, alpha=alpha)
    optimized = Optimizer(data, topology, alpha=alpha, initial_solution=heuristic.solution, univariate_split=True)
    end = time.time()

    size = get_size(optimized.solution)
    avg_fragmentation = get_average_fragmentation(optimized.solution, skeleton)
    depth = get_depth(optimized.solution.topology.root_node, optimized.solution.node_negative_arc, optimized.solution.node_positive_arc)
    question_l = question_length(optimized.solution, data.train_X, skeleton)

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

