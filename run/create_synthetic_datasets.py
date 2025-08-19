import numpy as np
import pandas as pd
import heuristic


# Creates synthetic datasets for running the synthetic experiments. Creates a train and test set.
# Some of the datasets will have identical settings. I removed those manually.
# You can run the datsets using local_search_synehtic.py


def create_unlabelled_data(n, p):
    X = np.random.uniform(0, 1, size=(n, p))
    return X

def initialize_diagram(diagram_nodes, X, num_classes, num_features):
    i = 0
    for node in diagram_nodes:
        if node.leftChild is None:
            node.leftLabel = i % num_classes
            i += 1
            node.rightLabel = i % num_classes
            i += 1

    diagram_nodes[0].data = X
    for node in diagram_nodes:
        node.feature = np.random.randint(0, num_features)
        node.threshold = np.random.uniform(0, 1)


def classify_sample(node, sample):
    if node.leftChild is None:
        if sample[node.feature] <= node.threshold:
            node.leftCount += 1
            return node.leftLabel
        else:
            node.rightCount += 1
            return node.rightLabel
    else:
        if sample[node.feature] <= node.threshold:
            return classify_sample(node.leftChild, sample)
        else:
            return classify_sample(node.rightChild, sample)

def label_data(diagram, X):
    y = []
    for sample in X:
        label = classify_sample(diagram[0], sample)
        y.append(label)

    return y


def add_feature_noise(X, intensity):
    noise = np.random.uniform(-intensity, intensity, size=X.shape)
    noisy_array = X + noise

    return noisy_array

def add_class_noise(y, percentage):
    num_to_flip = int(len(y) * percentage)
    flip_indices = np.random.choice(len(y), size=num_to_flip, replace=False)

    for index in flip_indices:
        y[index] = 1 - y[index]


def create_datasets(width, depth, num_samples, num_features, num_classes, feature_noise, class_noise, name, set_index):
    X = create_unlabelled_data(num_samples, num_features)  # Samples and features
    X = add_feature_noise(X, feature_noise)

    diagram = heuristic.get_structure(width, depth)
    initialize_diagram(diagram, X, num_classes, num_features)  # Diagram, data, classes, features
    y = label_data(diagram, X)
    add_class_noise(y, class_noise)

    while True:
        valid = True
        for node in diagram:
            if node.leftChild is None:
                if node.leftCount < 5 or node.rightCount < 5:
                    valid = False

        if not valid:
            X = create_unlabelled_data(num_samples, num_features)
            X = add_feature_noise(X, feature_noise)
            diagram = heuristic.get_structure(width, depth)
            initialize_diagram(diagram, X, num_classes, num_features)
            y = label_data(diagram, X)
            add_class_noise(y, class_noise)
        else:
            break

    df_X = pd.DataFrame(X, columns=[f'feature_{i + 1}' for i in range(X.shape[1])])
    df_X['label'] = y
    df_X.to_csv(
        f'../instances_synthetic/Synthetic_{name}_W{width}_D{depth}_S{num_samples}_NC{num_classes}_NF{num_features}_CN{class_noise}_FN{feature_noise}_{set_index}_Train.csv',
        index=False, float_format='%.8f')

    test_set_size = len(diagram) * (num_features - 1) * 500
    X_test = create_unlabelled_data(test_set_size, num_features)
    y_test = label_data(diagram, X_test)

    df_X_test = pd.DataFrame(X_test, columns=[f'feature_{i + 1}' for i in range(X_test.shape[1])])
    df_X_test['label'] = y_test
    df_X_test.to_csv(
        f'../instances_synthetic/Synthetic_{name}_W{width}_D{depth}_S{num_samples}_NC{num_classes}_NF{num_features}_CN{class_noise}_FN{feature_noise}_{set_index}_Test.csv',
        index=False, float_format='%.8f')


num_samples = 1000
num_classes = 2
num_features = 5
class_noise = 0
feature_noise = 0

structures = [(2, 5), (2, 10), (6, 5), (6, 10), (0, 5), (0, 10)]

for (width, depth) in structures:
    for num_samples in [250, 500, 1000, 5000]:
        for i in range(5):
            create_datasets(width, depth, num_samples, num_features,num_classes,feature_noise, class_noise, "NumSamples", i)

for (width, depth) in structures:
    for num_classes in [2, 4, 6, 8]:
        for i in range(5):
            create_datasets(width, depth, num_samples, num_features,num_classes,feature_noise, class_noise, "NumClasses", i)

for (width, depth) in structures:
    for num_features in [2, 5, 15, 25]:
        for i in range(5):
            create_datasets(width, depth, num_samples, num_features,num_classes,feature_noise, class_noise, "NumFeatures", i)

for (width, depth) in structures:
    for class_noise in [0.0, 0.2, 0.4, 0.5]:
        for i in range(5):
            create_datasets(width, depth, num_samples, num_features,num_classes,feature_noise, class_noise, "ClassNoise", i)

for (width, depth) in structures:
    for feature_noise in [0.0, 0.2, 0.6, 1.0]:
        for i in range(5):
            create_datasets(width, depth, num_samples, num_features,num_classes,feature_noise, class_noise, "FeatureNoise", i)