import numpy as np

class Node:
    def __init__(self, node_id, layer):
        self.node_id = node_id
        self.leftChild = None
        self.rightChild = None
        self.feature = None
        self.threshold = None

        # Holds data that passes through
        self.data = np.array([])
        self.layer = layer

        self.leftLabel = None
        self.rightLabel = None

        self.leftCount = 0
        self.rightCount = 0

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(X, y, feature_index, threshold, node):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0  # No valid split

    H_parent = entropy(y)
    y_left = y[left_mask]
    y_right = y[right_mask]

    if node.leftChild and node.rightChild:
        if node.leftChild.data.size > 0:
            y_left = np.concatenate((y_left, node.leftChild.data[:, -1]), axis=0)
        if node.rightChild.data.size > 0:
            y_right = np.concatenate((y_right, node.rightChild.data[:, -1]), axis=0)


    H_left = entropy(y_left)
    H_right = entropy(y_right)

    w_left = np.sum(left_mask) / len(y)
    w_right = np.sum(right_mask) / len(y)

    IG = H_parent - (w_left * H_left + w_right * H_right)
    return IG

# Fins best feature and threshold that maximizes information gain
def best_split(X, node):
    y = X[:, -1]  # Last column is the label
    best_ig = -1000
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1] - 1):  # Exclude label column
        values = np.sort(np.unique(X[:, feature_index]))  # Unique sorted feature values
        thresholds = (values[:-1] + values[1:]) / 2  # Midpoints as thresholds

        for threshold in thresholds:
            ig = information_gain(X, y, feature_index, threshold, node)
            if ig > best_ig:
                best_ig = ig
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_ig

def get_structure_width_2(num_layers):
    nodes = []
    root = Node(0, 0)
    nodes.append(root)

    if num_layers == 1:
        return nodes

    node1 = Node(1, 1)
    node2 = Node(2, 1)
    nodes.append(node1)
    nodes.append(node2)
    root.leftChild = node1
    root.rightChild = node2

    if num_layers == 2:
        return nodes


    cur_id = 2
    layer = 2
    for i in range(layer, num_layers):
        new_node1 = Node(cur_id+1, layer)
        new_node2 = Node(cur_id+2, layer)
        layer += 1

        nodes.append(new_node1)
        nodes.append(new_node2)

        nodes[cur_id-1].leftChild = new_node1
        nodes[cur_id-1].rightChild = new_node2
        nodes[cur_id].leftChild =  new_node1
        nodes[cur_id].rightChild = new_node2
        cur_id += 2

    return nodes

def get_structure_width_4(num_layers):
    nodes = []
    root = Node(0, 0)
    nodes.append(root)
    if num_layers == 1:
        return nodes

    node1 = Node(1, 1)
    node2 = Node(2, 1)
    root.leftChild = node1
    root.rightChild = node2
    nodes.append(node1)
    nodes.append(node2)

    if num_layers == 2:
        return nodes

    node3 = Node(3, 2)
    node4 = Node(4, 2)
    node5 = Node(5, 2)
    node6 = Node(6, 2)
    node1.leftChild = node3
    node1.rightChild = node4
    node2.leftChild = node5
    node2.rightChild = node6
    nodes.append(node3)
    nodes.append(node4)
    nodes.append(node5)
    nodes.append(node6)

    if num_layers == 3:
        return nodes


    layer = 3
    cur_id = 6
    for i in range(layer, num_layers):
        new_node1 = Node(cur_id+1, layer)
        new_node2 = Node(cur_id+2, layer)
        new_node3 = Node(cur_id+3, layer)
        new_node4 = Node(cur_id+4, layer)
        layer += 1

        nodes.append(new_node1)
        nodes.append(new_node2)
        nodes.append(new_node3)
        nodes.append(new_node4)

        nodes[cur_id-3].leftChild = new_node1
        nodes[cur_id-3].rightChild = new_node2
        nodes[cur_id-2].leftChild = new_node1
        nodes[cur_id-2].rightChild = new_node3
        nodes[cur_id-1].leftChild = new_node2
        nodes[cur_id-1].rightChild = new_node4
        nodes[cur_id].leftChild =  new_node3
        nodes[cur_id].rightChild = new_node4
        cur_id += 4

    return nodes

def get_structure_width_6(num_layers):
    nodes = []
    root = Node(0, 0)
    nodes.append(root)

    if num_layers == 1:
        return nodes

    node1 = Node(1, 1)
    node2 = Node(2, 1)
    root.leftChild = node1
    root.rightChild = node2
    nodes.append(node1)
    nodes.append(node2)

    if num_layers == 2:
        return nodes

    node3 = Node(3, 2)
    node4 = Node(4, 2)
    node5 = Node(5, 2)
    node6 = Node(6, 2)
    node1.leftChild = node3
    node1.rightChild = node4
    node2.leftChild = node5
    node2.rightChild = node6
    nodes.append(node3)
    nodes.append(node4)
    nodes.append(node5)
    nodes.append(node6)

    if num_layers == 3:
        return nodes

    node7 = Node(7, 3)
    node8 = Node(8, 3)
    node9 = Node(9, 3)
    node10 = Node(10, 3)
    node11 = Node(11, 3)
    node12 = Node(12, 3)
    node3.leftChild = node7
    node3.rightChild = node9
    node4.leftChild = node8
    node4.rightChild = node10
    node5.leftChild = node9
    node5.rightChild = node11
    node6.leftChild = node10
    node6.rightChild = node12
    nodes.append(node7)
    nodes.append(node8)
    nodes.append(node9)
    nodes.append(node10)
    nodes.append(node11)
    nodes.append(node12)

    if num_layers == 4:
        return nodes


    layer = 4
    cur_id = 12
    for i in range(layer, num_layers):
        new_node1 = Node(cur_id+1, layer)
        new_node2 = Node(cur_id+2, layer)
        new_node3 = Node(cur_id+3, layer)
        new_node4 = Node(cur_id+4, layer)
        new_node5 = Node(cur_id+5, layer)
        new_node6 = Node(cur_id+6, layer)
        layer += 1

        nodes.append(new_node1)
        nodes.append(new_node2)
        nodes.append(new_node3)
        nodes.append(new_node4)
        nodes.append(new_node5)
        nodes.append(new_node6)

        nodes[cur_id - 5].leftChild = new_node1
        nodes[cur_id - 5].rightChild = new_node2
        nodes[cur_id - 4].leftChild = new_node1
        nodes[cur_id - 4].rightChild = new_node3
        nodes[cur_id-3].leftChild = new_node2
        nodes[cur_id-3].rightChild = new_node4
        nodes[cur_id-2].leftChild = new_node3
        nodes[cur_id-2].rightChild = new_node5
        nodes[cur_id-1].leftChild = new_node4
        nodes[cur_id-1].rightChild = new_node6
        nodes[cur_id].leftChild =  new_node5
        nodes[cur_id].rightChild = new_node6
        cur_id += 6

    return nodes

def get_structure_width_8(num_layers):
    nodes = []
    root = Node(0, 0)
    nodes.extend([root])

    if num_layers == 1:
        return nodes

    node1 = Node(1, 1)
    node2 = Node(2, 1)
    root.leftChild = node1
    root.rightChild = node2
    nodes.extend([node1, node2])

    if num_layers == 2:
        return nodes

    node3 = Node(3, 2)
    node4 = Node(4, 2)
    node5 = Node(5, 2)
    node6 = Node(6, 2)
    node1.leftChild = node3
    node1.rightChild = node4
    node2.leftChild = node5
    node2.rightChild = node6
    nodes.extend([node3, node4, node5, node6])

    if num_layers == 3:
        return nodes

    node7 = Node(7, 3)
    node8 = Node(8, 3)
    node9 = Node(9, 3)
    node10 = Node(10, 3)
    node11 = Node(11, 3)
    node12 = Node(12, 3)
    node13 = Node(13, 3)
    node14 = Node(14, 3)
    node3.leftChild = node7
    node3.rightChild = node8
    node4.leftChild = node9
    node4.rightChild = node10
    node5.leftChild = node11
    node5.rightChild = node12
    node6.leftChild = node13
    node6.rightChild = node14
    nodes.extend([node7, node8, node9, node10, node11, node12, node13, node14])

    if num_layers == 4:
        return nodes

    layer = 4
    cur_id = 14
    for i in range(layer, num_layers):
        new_nodes = [Node(cur_id + j + 1, layer) for j in range(8)]
        layer += 1
        nodes.extend(new_nodes)

        nodes[cur_id - 7].leftChild = new_nodes[0]
        nodes[cur_id - 7].rightChild = new_nodes[1]
        nodes[cur_id - 6].leftChild = new_nodes[0]
        nodes[cur_id - 6].rightChild = new_nodes[2]
        nodes[cur_id - 5].leftChild = new_nodes[1]
        nodes[cur_id - 5].rightChild = new_nodes[3]
        nodes[cur_id - 4].leftChild = new_nodes[2]
        nodes[cur_id - 4].rightChild = new_nodes[4]
        nodes[cur_id - 3].leftChild = new_nodes[3]
        nodes[cur_id - 3].rightChild = new_nodes[5]
        nodes[cur_id - 2].leftChild = new_nodes[4]
        nodes[cur_id - 2].rightChild = new_nodes[6]
        nodes[cur_id - 1].leftChild = new_nodes[5]
        nodes[cur_id - 1].rightChild = new_nodes[7]
        nodes[cur_id].leftChild = new_nodes[6]
        nodes[cur_id].rightChild = new_nodes[7]

        cur_id += 8

    return nodes

def get_structure_pascal(num_layers):
    if num_layers < 1:
        return []

    if num_layers == 1:
        num_layers = 2

    nodes = []
    layers = []

    node_id = 0
    for layer_num in range(1, num_layers + 1):
        layer = []
        for _ in range(layer_num):
            node = Node(node_id, layer_num-1)
            nodes.append(node)
            layer.append(node)
            node_id += 1
        layers.append(layer)

    # Link each node to two children in the next layer
    for i in range(len(layers) - 1):
        curr_layer = layers[i]
        next_layer = layers[i + 1]
        for j, node in enumerate(curr_layer):
            node.leftChild = next_layer[j]
            node.rightChild = next_layer[j + 1]

    return nodes

def get_structure_binary_tree(num_layers):
    if num_layers < 1:
        return []

    if num_layers == 1:
        num_layers = 2

    nodes = []
    layers = []

    node_id = 0
    for layer_num in range(num_layers):
        layer_size = 2 ** layer_num
        layer = []
        for _ in range(layer_size):
            node = Node(node_id, layer_num)
            nodes.append(node)
            layer.append(node)
            node_id += 1
        layers.append(layer)

    for i in range(len(layers) - 1):
        curr_layer = layers[i]
        next_layer = layers[i + 1]
        for j, node in enumerate(curr_layer):
            node.leftChild = next_layer[2 * j]
            node.rightChild = next_layer[2 * j + 1]

    return nodes

def get_structure(width, depth):
    if width == 2:
        return get_structure_width_2(depth)
    elif width == 4:
        return get_structure_width_4(depth)
    elif width == 6:
        return get_structure_width_6(depth)
    elif width == 8:
        return get_structure_width_8(depth)
    else:
        return get_structure_pascal(depth)


def initialize_diagram(X, nodes):

    nodes[0].data = X
    for node in nodes:

        if node.data.size == 0:
            node.feature = -1
            node.threshold = -999999
            continue

        last_column = node.data[:, -1]
        unique_values = np.unique(last_column)

        if len(unique_values) == 1:
            # Select random feature and threshold
            node.feature = -1
            node.threshold = -999999
        else:
            feature, threshold, ig = best_split(node.data, node)

            node.feature = feature
            node.threshold = threshold


        if len(node.data) == 0:
            continue

        if node.leftChild and node.rightChild:
            left_mask = node.data[:, node.feature] <= node.threshold
            right_mask = ~left_mask

            if node.leftChild.data.size > 0:
                node.leftChild.data = np.concatenate((node.data[left_mask], node.leftChild.data), axis=0)
            else:
                node.leftChild.data = node.data[left_mask]

            if node.rightChild.data.size > 0:
                node.rightChild.data = np.concatenate((node.data[right_mask], node.rightChild.data), axis=0)
            else:
                node.rightChild.data = node.data[right_mask]

def save_diagram(nodes, initialized):
    with open("../temp/diagram", "w") as f:
        f.write(str(len(nodes)) + "\n")

        if initialized:
            for node in nodes:
                if node.leftChild and node.rightChild:
                    f.write("normal,{},{},{},{},{},{}\n".format(
                        node.node_id, node.layer, node.leftChild.node_id, node.rightChild.node_id, node.feature, node.threshold))
                else:
                    f.write("final,{},{},{},{}\n".format(
                        node.node_id, node.layer, node.feature, node.threshold))
        else:
            for node in nodes:
                if node.leftChild and node.rightChild:
                    f.write("normal,{},{},{},{},{},{}\n".format(
                        node.node_id, node.layer, node.leftChild.node_id, node.rightChild.node_id, -1, -1))
                else:
                    f.write("final,{},{},{},{}\n".format(
                        node.node_id, node.layer, -1, -1))













