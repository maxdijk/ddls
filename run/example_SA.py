import os
import glob
import subprocess
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import heuristic
import visualize
import re


filename = "breast-cancer-diagnostic.csv"
file_path = "../instances/" + filename

df = pd.read_csv(file_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

X_train = np.column_stack((X_train, y_train))
X_test = np.column_stack((X_test, y_test))

np.savetxt(f"../temp/example_train.csv", X_train, delimiter=",")
np.savetxt(f"../temp/example_test.csv", X_test, delimiter=",")

structure = heuristic.get_structure(4, 5)           # Get empty structure of width 4 and depth 5
heuristic.initialize_diagram(X_train, structure)    # Initialize the empty diagram
heuristic.save_diagram(structure, True)             # Save diagram to disk

command = [
    "../bin/bdd.exe",
    '--SA1::cooling_rate', "0.9",
    '--SA1::max_evaluations', '50000',
    '--SA1::start_temperature', "80",
    '--SA1::min_temperature', '0.0001',
    '--main::method', 'SA',                         # SA for simulated annealing
    '--main::initialization', "greedy",             # greedy for information gain initialization or random for random
    '--main::alpha', '0.01',
    '--main::filename_train', "example_train.csv",
    '--main::filename_test', "example_test.csv",
    '--main::seed', str(random.randint(0, 1000)),
    '--main::random_move', '0.35',
    '--main::optimal_move', '0.6',
    '--main::edge_move', '0.05'
]

try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output_lines = result.stdout.strip().split("\n")
    target_line = output_lines[-1]
    values = target_line.split(",")


    print(target_line)
    print(f"Objective: {values[3]}")
    print(f"Train Accuracy: {values[9]}")
    print(f"Test Accuracy: {values[10]}")


    # Next part is  for getting getting part of the output and saving it for visualization
    lines = result.stdout.splitlines()
    result = []

    for i, line in enumerate(lines):
        if re.fullmatch(r"\d+", line.strip()):
            count = int(line.strip()) * 3
            result = lines[i:i + 1 + count]
            break

    with open("diagram_output.txt", "w") as f:
        f.write("\n".join(result))

    visualize.visualize_diagram_improved("diagram_output.txt")
except subprocess.CalledProcessError as e:
    print("Command failed with error:", e)