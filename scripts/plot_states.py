import jax.numpy as jnp
import flax
from arnn.coupled_arnn import AutoregressiveNN,DenseOrbital
import netket as nk
import netket.experimental as nkx
import openfermion as of
import pickle
import numpy as np
import time
from netket.experimental.hilbert import SpinOrbitalFermions
from arnn.coupled_sampler import ARWeightedSampler
from arnn.weighted_state import WeightedMCState
from arnn.numba_operator import HBOperator
from arnn.weighted_qgt import QGTJacobianPyTree
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Run experimental script.', allow_abbrev=True)
parser.add_argument('-n','--name', nargs='?', required = True,
                        help='Molecule Name, should be equal to the hamiltonian file name')
parser.add_argument('-lr', '--learningrate',default = "0.001",type=float)
parser.add_argument('-l', '--loadstate', default = False,type=bool)
parser.add_argument('-eps','--epsilon', default = 0,type=float)
parser.add_argument('-s','--samples',default = 100,type=int)
parser.add_argument('-i','--iterations',default = 10000,type=int)
parser.add_argument('-m','--magnetization',type=int)
parser.add_argument('-e','--electrons',type=int)
parser.add_argument('-o','--orbitals',type=int)
args = parser.parse_args()
name = args.name
load_state = 1
number_of_orbitals = args.orbitals
alpha = int(args.electrons / 2 + args.magnetization/2)
beta = int(args.electrons / 2 - args.magnetization/2)
n_alpha_beta =(alpha,beta)
learning_rate = args.learningrate
n_layers = 2
n_features = 64
samples = args.samples
iterations = args.iterations
eps = args.epsilon

###############################
#Define Hilbert space
##############################

hi = nkx.hilbert.SpinOrbitalFermions(number_of_orbitals,1/2,n_alpha_beta)  # Number of electrons can be a list with [alpha,beta] spin!

#############################
#Define Hamiltonian, use prepared OpenFermion operator!
############################
ha =HBOperator.from_openfermion(hi,f"/data/jkocklaeuner/Master/NAQS/data/{name}.hdf5",epsilon = eps)
# Autoregressive neural network
subspaces = 2
ma = AutoregressiveNN(hilbert=hi, layers=n_layers, features=n_features,use_phase=True,dtype=jnp.float32)

#Sampler

sa = ARWeightedSampler(hi)

# Optimizer

op = nk.optimizer.Adam(learning_rate= learning_rate,b2=0.99)

# Variational state
# With direct sampling, we don't need many samples in each step to form a
# Markov chain, and we don't need to discard samples
vs = WeightedMCState(sa,ma,n_samples=samples)
if load_state:
    with open(f"{name}.mpack", 'rb') as file:
        vs.variables = flax.serialization.from_bytes(vs.variables, file.read())

exact_states = np.array([hi.numbers_to_states(i) for i in range(hi.n_states)],dtype=np.int8)
amplis = np.zeros(hi.n_states,dtype=np.complex64)
for i,state in enumerate(exact_states):    
    ampli = vs.log_value(state)
    amplis[i] = np.exp(ampli)
x = np.arange(hi.n_states)
mask = np.round(amplis,6)
mask = np.nonzero(mask)
np.savetxt("test.txt",amplis)
np.savetxt("states.txt",exact_states)
#plt.scatter(x[mask],amplis[mask])
#plt.yscale('log')
#plt.savefig("states.png")

#print("Total norm is " , amplis)
#print("Total sampled norm is " , np.sum(np.exp(sample_amplis.real * 2)))
