import graphviz
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

        self.left_class_count = None
        self.right_class_count = None
        self.class_count = None

        self.layer = layer

        self.leftLabel = None
        self.rightLabel = None

        self.leftCount = 0
        self.rightCount = 0


def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


def get_color(value, colors):
    # Find the appropriate color & intensity for a node
    # Classification tree
    if np.sum(value) > 0:
        value = value / np.sum(value)

    color = list(colors[np.argmax(value)])
    sorted_values = sorted(value, reverse=True)
    if len(sorted_values) == 1:
        alpha = 0.0
    else:
        alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])

    # compute the color as alpha against white
    color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
    # Return html color code in #RRGGBB format
    return "#%2x%2x%2x" % tuple(color)


def subscript_number(n):
    sub_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(sub_digits)


def visualize_diagram_improved(filename):
    dot = graphviz.Digraph("diagram", comment="Testing Diagram")

    with open(filename) as f:
        lines = f.readlines()
        number_decision_nodes = int(lines[0])
        number_classes = int(len(lines[2]))
        colors = _color_brew(number_classes)
        nodes = []

        for i in range(number_decision_nodes):
            nodes.append(Node(i, -1))


        for i in range(len(lines)):
            line = lines[i]

            if line.startswith("normal"):
                node = line.split(',')
                node_id = int(node[1])
                left_child = int(node[3])
                right_child = int(node[4])
                feature = int(node[5])
                threshold = float(node[6])

                left = np.array(list(map(int, lines[i + 1].split(','))))
                right = np.array(list(map(int, lines[i + 2].split(','))))
                total = left + right

                cur_node = nodes[node_id]
                cur_node.feature = feature
                cur_node.threshold = threshold
                cur_node.leftChild = nodes[left_child]
                cur_node.rightChild = nodes[right_child]
                cur_node.left_class_count = left
                cur_node.right_class_count = right
                cur_node.class_count = total

            if line.startswith("final"):
                node = line.split(',')
                node_id = int(node[1])
                feature = int(node[3])
                threshold = float(node[4])
                left = np.array(list(map(int, lines[i + 1].split(','))))
                right = np.array(list(map(int, lines[i + 2].split(','))))
                total = left + right

                cur_node = nodes[node_id]
                cur_node.feature = feature
                cur_node.threshold = threshold
                cur_node.left_class_count = left
                cur_node.right_class_count = right
                cur_node.class_count = total


        for node in nodes:
            if node.leftChild is None:
                continue

            left_node = node.leftChild
            while left_node is not None and (left_node.threshold < -1000000 or left_node.threshold > 10000000):
                if left_node.threshold < -1000000:
                    left_node = left_node.rightChild
                elif left_node.threshold > 10000000:
                    left_node = left_node.leftChild

            right_node = node.rightChild
            while right_node is not None and (right_node.threshold < -1000000 or right_node.threshold > 10000000):
                if right_node.threshold < -1000000:
                    right_node = right_node.rightChild
                elif right_node.threshold > 10000000:
                    right_node = right_node.leftChild

            node.leftChild = left_node
            node.rightChild = right_node

        visited_nodes = set()
        todo = []
        todo.append(nodes[0])
        while len(todo) > 0:
            cur_node = todo.pop(0)
            visited_nodes.add(cur_node)
            if cur_node.leftChild is not None:
                todo.append(cur_node.leftChild)
            if cur_node.rightChild is not None:
                todo.append(cur_node.rightChild)

        for node in list(visited_nodes):
            #print(f"ID {node.node_id}, left {node.leftChild}, right {node.rightChild}")


            dot.node("{}".format(node.node_id),
                     f"x{subscript_number(node.feature)} <= {node.threshold} \n [{', '.join(map(str, node.class_count))}]", shape="box",
                     style="filled", fillcolor=get_color(node.class_count, colors))

            if node.leftChild is not None:
                dot.edge("{}".format(node.node_id), "{}".format(node.leftChild.node_id))
            else:
                dot.node("dummy1_{}".format(node.node_id), f"[{', '.join(map(str, node.left_class_count))}]", shape="box", style="rounded,filled",
                         fillcolor=get_color(node.left_class_count, colors))
                dot.edge("{}".format(node.node_id), "dummy1_{}".format(node.node_id))


            if node.rightChild is not None:
                dot.edge("{}".format(node.node_id), "{}".format(node.rightChild.node_id), style='dotted')
            else:
                dot.node("dummy2_{}".format(node.node_id), f"[{', '.join(map(str, node.right_class_count))}]",
                         shape="box", style="rounded,filled",
                         fillcolor=get_color(node.right_class_count, colors))
                dot.edge("{}".format(node.node_id), "dummy2_{}".format(node.node_id), style='dotted')

        dot.render(directory="output", view=True)
