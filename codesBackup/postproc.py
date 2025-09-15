import numpy as np
import pickle

# load a .dat file
file_path = "solverInfo.dat"

data = {}
with open(file_path, "r") as file:
    for line in file:
        print(line.strip())
