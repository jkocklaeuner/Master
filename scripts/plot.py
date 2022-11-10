#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import argparse

parser = argparse.ArgumentParser(description='Plot optimization relative to a reference value with a sliding average of 50 steps')
parser.add_argument('-f','--filename', type=str,
                    help='filename of a .log file')
parser.add_argument('-r','--reference', type = float,
                     default=0.0, help = "FCI reference value, if available given in .out file"
                    )
parser.add_argument('-s','--scale', type = str, default = "linear", help = "y-Axis scale, choose between log and linear")
parser.add_argument('-v','--values', default = None, help = 'Reference values displayed in the plot, should be given as list [[value 1, name 1], [value 2 , name 2],...]',nargs="+",action = "append") 


args = parser.parse_args()
name , ref, scale = args.filename, args.reference, args.scale
vals = args.values
log = json.load(open(f"{name}.log",'r'))
energy = log['Energy']['Mean']['real']
iterations = log['Energy']['iters']
colors = ['g','r','y','c','m','k']
energy = abs(np.array(energy) - ref)


out = np.zeros_like(energy)
for i,el in enumerate(energy):
    if i > 50:
        a = i-50
        out[i] = abs(np.mean(energy[a:i]))
    else:
        if i > 0:
            out[i] = abs(np.mean(energy[:i]))
        else:
            out[i] = energy[i]
#    out[i] = np.min(energy[:i])

#out=np.round(out,6)
if vals:
    vals = vals[0]
    for counter in range(int(len(vals)/2)):
        val, label = vals[2*counter],vals[ 2*counter + 1]
        val = float(val) - ref
        plt.hlines(val,iterations[0],iterations[-1],linestyle='dashed',label = label,color=colors[counter])
plt.plot(iterations[::10],out[::10],label="NNQS",lw = 2)
plt.yscale(args.scale) 
plt.ylabel("$\Delta E / \ E_H$",color = "black")
plt.xlabel("Iterations",color = "black")
plt.xticks(color = "black")
plt.yticks(color = "black")
plt.legend()
plt.title(f"{name}")
plt.tight_layout()

plt.savefig(f"{args.filename}.png")
plt.savefig(f"{args.filename}.svg")

