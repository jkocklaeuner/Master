#!/usr/bin/python3

import json
import numpy as np
import sys

name = sys.argv[1]
log = json.load(open(f"{name}",'r'))
e = log['Energy']['Mean']['real']
print(f"Current number of iterations: {len(e)} \nAverage energy of the last 50 steps: {np.mean(e[-50:])} \nLowest variational energy: {min(e)}")
print("Variational energies obtained in the last 10 steps:")
print(e[-10:])
