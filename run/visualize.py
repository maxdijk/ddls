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
    print(value)
    if np.sum(value) > 0:
        value = value / np.sum(value)
    print(value)

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


def visualize_diagram_old():
    dot = graphviz.Digraph("diagram", comment="Testing Diagram")

    with open("diagram") as f:
        lines = f.read().splitlines()

        number_decision_nodes = int(lines[0])

        for i in range(1, 1 + number_decision_nodes):
            node = lines[i].split(',')

            if node[0] == "normal":
                node_id = int(node[1])
                left_child = int(node[3])
                right_child = int(node[4])
                feature = int(node[5])
                threshold = float(node[6])

                # dot.node("{}".format(node_id), "x{} <= {}".format(feature, threshold),shape="box")
                dot.node("{}".format(node_id), "{}".format(node_id), shape="box")
                dot.edge("{}".format(node_id), "{}".format(left_child))
                dot.edge("{}".format(node_id), "{}".format(right_child), style='dotted')
            else:
                node = lines[i].split(',')

                node_id = int(node[1])
                feature = int(node[3])
                threshold = float(node[4])

                # dot.node("{}".format(node_id), "{} <= {}".format(feature, threshold),shape="box")
                dot.node("{}".format(node_id), "{}".format(node_id), shape="box")
                dot.node("dummy1_{}".format(node_id), "", shape="oval", width="0.2", height="0.2")
                dot.node("dummy2_{}".format(node_id), "", shape="oval", width="0.2", height="0.2")
                dot.edge("{}".format(node_id), "dummy1_{}".format(node_id))
                dot.edge("{}".format(node_id), "dummy2_{}".format(node_id), style='dotted')

        dot.render(directory="output", view=True)


def subscript_number(n):
    sub_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(sub_digits)


def visualize_diagram():
    dot = graphviz.Digraph("diagram", comment="Testing Diagram")

    with open("diagram") as f:
        lines = f.readlines()
        number_decision_nodes = int(lines[0])
        number_classes = int(len(lines[2]))
        colors = _color_brew(number_classes)

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

                color = get_color(total, colors)

                dot.node("{}".format(node_id),
                         f"x{subscript_number(feature)} <= {threshold} \n [{', '.join(map(str, total))}]", shape="box",
                         style="filled", fillcolor=color)
                dot.edge("{}".format(node_id), "{}".format(left_child))
                dot.edge("{}".format(node_id), "{}".format(right_child), style='dotted')

            elif line.startswith("final"):
                node = line.split(',')

                node_id = int(node[1])
                feature = int(node[3])
                threshold = float(node[4])

                left = np.array(list(map(int, lines[i + 1].split(','))))
                right = np.array(list(map(int, lines[i + 2].split(','))))
                total = left + right

                color = get_color(total, colors)
                color_left = get_color(left, colors)
                color_right = get_color(right, colors)


                # dot.node("{}".format(node_id), "{} <= {}".format(feature, threshold),shape="box")
                dot.node("{}".format(node_id),
                         f"x{subscript_number(feature)} <= {threshold} \n [{', '.join(map(str, total))}]", shape="box",
                         style="filled", fillcolor=color)
                dot.node("dummy1_{}".format(node_id), f"[{', '.join(map(str, left))}]", shape="oval", style="filled",
                         fillcolor=color_left)
                dot.node("dummy2_{}".format(node_id), f"[{', '.join(map(str, right))}]", shape="oval", style="filled",
                         fillcolor=color_right)
                dot.edge("{}".format(node_id), "dummy1_{}".format(node_id))
                dot.edge("{}".format(node_id), "dummy2_{}".format(node_id), style='dotted')

        dot.render(directory="output", view=True)

def visualize_diagram2():
    dot = graphviz.Digraph("diagram", comment="Testing Diagram")

    with open("diagram") as f:
        lines = f.readlines()
        number_decision_nodes = int(lines[0])
        number_classes = int(len(lines[2]))
        colors = _color_brew(number_classes)

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

                color = get_color(total, colors)

                if threshold < -1000000:
                    dot.node("{}".format(node_id),
                             f"x{subscript_number(feature)} <= -∞ \n [{', '.join(map(str, total))}]",
                             shape="box",
                             style="filled", fillcolor=color)
                    dot.edge("{}".format(node_id), "{}".format(left_child))
                    dot.edge("{}".format(node_id), "{}".format(right_child), style='dotted')
                elif threshold > 100000:
                    dot.node("{}".format(node_id),
                             f"x{subscript_number(feature)} <= ∞ \n [{', '.join(map(str, total))}]",
                             shape="box",
                             style="filled", fillcolor=color)
                    dot.edge("{}".format(node_id), "{}".format(left_child))
                    dot.edge("{}".format(node_id), "{}".format(right_child), style='dotted')
                else:
                    dot.node("{}".format(node_id),
                             f"x{subscript_number(feature)} <= {threshold} \n [{', '.join(map(str, total))}]", shape="box",
                             style="filled", fillcolor=color)
                    dot.edge("{}".format(node_id), "{}".format(left_child))
                    dot.edge("{}".format(node_id), "{}".format(right_child), style='dotted')

            elif line.startswith("final"):
                node = line.split(',')

                node_id = int(node[1])
                feature = int(node[3])
                threshold = float(node[4])

                left = np.array(list(map(int, lines[i + 1].split(','))))
                right = np.array(list(map(int, lines[i + 2].split(','))))
                total = left + right

                color = get_color(total, colors)
                color_left = get_color(left, colors)
                color_right = get_color(right, colors)

                if threshold < -1000000:
                    dot.node("{}".format(node_id),
                             f"x{subscript_number(feature)} <= -∞ \n [{', '.join(map(str, total))}]",
                             shape="box",
                             style="filled", fillcolor=color)
                    dot.node("dummy1_{}".format(node_id), f"[{', '.join(map(str, left))}]", shape="box",
                             style="rounded,filled",
                             fillcolor=color_left)
                    dot.node("dummy2_{}".format(node_id), f"[{', '.join(map(str, right))}]", shape="box",
                             style="rounded,filled",
                             fillcolor=color_right)
                    dot.edge("{}".format(node_id), "dummy1_{}".format(node_id))
                    dot.edge("{}".format(node_id), "dummy2_{}".format(node_id), style='dotted')

                elif threshold > 100000:
                    dot.node("{}".format(node_id),
                             f"x{subscript_number(feature)} <= ∞ \n [{', '.join(map(str, total))}]",
                             shape="box",
                             style="filled", fillcolor=color)
                    dot.node("dummy1_{}".format(node_id), f"[{', '.join(map(str, left))}]", shape="box",
                             style="rounded,filled",
                             fillcolor=color_left)
                    dot.node("dummy2_{}".format(node_id), f"[{', '.join(map(str, right))}]", shape="box",
                             style="rounded,filled",
                             fillcolor=color_right)
                    dot.edge("{}".format(node_id), "dummy1_{}".format(node_id))
                    dot.edge("{}".format(node_id), "dummy2_{}".format(node_id), style='dotted')
                else:

                    # dot.node("{}".format(node_id), "{} <= {}".format(feature, threshold),shape="box")
                    dot.node("{}".format(node_id),
                             f"x{subscript_number(feature)} <= {threshold} \n [{', '.join(map(str, total))}]", shape="box",
                             style="filled", fillcolor=color)
                    dot.node("dummy1_{}".format(node_id), f"[{', '.join(map(str, left))}]", shape="box", style="rounded,filled",
                             fillcolor=color_left)
                    dot.node("dummy2_{}".format(node_id), f"[{', '.join(map(str, right))}]", shape="box", style="rounded,filled",
                             fillcolor=color_right)
                    dot.edge("{}".format(node_id), "dummy1_{}".format(node_id))
                    dot.edge("{}".format(node_id), "dummy2_{}".format(node_id), style='dotted')

        dot.render(directory="output", view=True)

def visualize_diagram_empty():
    dot = graphviz.Digraph("diagram", comment="Testing Diagram")

    with open("diagram") as f:
        lines = f.readlines()
        number_decision_nodes = int(lines[0])
        number_classes = int(len(lines[2]))
        colors = _color_brew(number_classes)

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

                color = get_color(total, colors)

                dot.node("{}".format(node_id),
                         shape="box",
                         )
                dot.edge("{}".format(node_id), "{}".format(left_child))
                dot.edge("{}".format(node_id), "{}".format(right_child), style='dotted')

            elif line.startswith("final"):
                node = line.split(',')

                node_id = int(node[1])
                feature = int(node[3])
                threshold = float(node[4])

                left = np.array(list(map(int, lines[i + 1].split(','))))
                right = np.array(list(map(int, lines[i + 2].split(','))))
                total = left + right

                color = get_color(total, colors)
                color_left = get_color(left, colors)
                color_right = get_color(right, colors)


                # dot.node("{}".format(node_id), "{} <= {}".format(feature, threshold),shape="box")
                dot.node("{}".format(node_id),
                         shape="box",
                         )
                dot.node("dummy1_{}".format(node_id), "", shape="oval"
                         )
                dot.node("dummy2_{}".format(node_id), "",  shape="oval")
                dot.edge("{}".format(node_id), "dummy1_{}".format(node_id))
                dot.edge("{}".format(node_id), "dummy2_{}".format(node_id), style='dotted')

        dot.render(directory="output", view=True)

def visualize_diagram_improved():
    dot = graphviz.Digraph("diagram", comment="Testing Diagram")

    with open("diagram") as f:
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
                print("TEST")
                print(threshold)
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


            if node.node_id == 33:
                test = 5
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
            print(f"ID {node.node_id}, left {node.leftChild}, right {node.rightChild}")

            # dot.node("{}".format(node.node_id), "{}".format(node.feature, node.threshold), shape="box")
            # dot.node("{}".format(node.node_id), "{}".format(node.node_id), shape="box")

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