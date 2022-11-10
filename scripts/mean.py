#!/usr/bin/python3

import json
import numpy as np
import sys

name = sys.argv[1]
min_e = np.zeros(5)
for i,seed in enumerate([111,222,333,444,555]):    
    log = json.load(open(f"{name}_{seed}.log",'r'))
    e = log['Energy']['Mean']['real']
    min_e[i] = min(e)
print("energies: ",min_e)
print("Mean: ", min_e.mean())
print("Std: ", min_e.std())
  
