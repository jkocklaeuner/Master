# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import optax
import flax
from arnn.models import NADE, RNN
import netket as nk
import netket.experimental as nkx
import openfermion as of
import pickle
import numpy as np
import time
from netket.experimental.hilbert import SpinOrbitalFermions
from arnn.sampler import NADEWeightedSampler, RNNWeightedSampler
from arnn.weighted_state import WeightedMCState
from arnn.fermion_operator import ElectronicOperator
from arnn.weighted_qgt import QGTJacobianPyTree
from arnn.supervised import pretraining
import argparse
from scipy.special import binom
import os

loc = os.path.dirname(__file__)


parser = argparse.ArgumentParser(description='Run a single VMC optimization with a RNN network', allow_abbrev=True)
parser.add_argument('-n','--name', nargs='?', required = True,
                        help='Molecule Name, should be equal to the hamiltonian file name')
parser.add_argument('-lr', '--learningrate',default = "0.001",type=float,help = "Learning rate of the optimizer, check optax or NetKet for details")
parser.add_argument('-ls', '--loadstate', default = False,type=bool,help = "Boolean, load a state from a previous calculation with the same settings")
parser.add_argument('-s','--samples',default = 100,type=int,help="Number of unique samples per iteration")
parser.add_argument('-i','--iterations',default = 10000,type=int, help = "Total number of optimization iterations")
parser.add_argument('-m','--magnetization',type=int,default = 0, help="Multiplicity")
parser.add_argument('-e','--electrons',type=int, help="Number of electrons")
parser.add_argument('-o','--orbitals',type=int, help = "Number of spatial orbitals")
parser.add_argument('-se','--seed',type = int,default = 0,  help = "Seed for the random initialization procedure")
parser.add_argument('-sub','--subsize',default = 1, type = int, help = "Size of each subnetwork, integer number of spatial orbitals. Should be an integer divisor of the spatial orbitals")
parser.add_argument('-p','--pretrain',type = bool, default = False, help = "Perform a pre-optimization with a CISD wavefunction, different options can be set directly in the script. Only use this option for CISD spaces < 10**5. Preoptimization is continued for 20000 steps or until an overlap of exp(-0.001) is reached.  ")
parser.add_argument('-f','--features',type=int,default = 64, help = "Number of hidden layer neurons for Dense layers, phase layers are multiplied by a factor 8" )
parser.add_argument('-l','--layers',type=int,default = 2, help = "Number of Dense layers of the phase network (+ 1 output layer of size 4) and each subnetwork (1 layer = output layer)")
parser.add_argument('-b','--buffer',type=int,default = 4, help = "Size of the sampling buffer multiplied with the number of samples, #buffer * #samples unique states are sampled and the #samples states with the largest amplitudes are used in the calculation" ) 
args = parser.parse_args()


name = args.name
load_state = args.loadstate
number_of_orbitals = args.orbitals
alpha = int(args.electrons / 2 + args.magnetization)
beta = int(args.electrons / 2 - args.magnetization)
n_alpha_beta =(alpha,beta)
learning_rate = args.learningrate
n_layers = args.layers
n_features = args.features
samples = args.samples
iterations = args.iterations
seed = args.seed
pretrain = args.pretrain
fci=0
mp2 = 0
cisd = 1
pre_steps = 20000
rescale = 1
subsize = args.subsize
s_buffer = args.buffer
n_dets = int(binom(number_of_orbitals,alpha) * binom(number_of_orbitals,beta))

if n_dets < samples:
    print("Number of samples exceeds hilbert space size! Samples set to hilbert space size")
    samples = n_dets 
if (number_of_orbitals % subsize) != 0:
    raise ValueError("Subnetwork size has to be an integer divisor of the number of spatial orbitals")
 


print(f"Running script nade.py \n"
f"Check options via scripts/rnn.py --help \n"
f"General setting:\n" \
f"Molecule {name} \n" \
f"Restart calculation: {load_state} \n"
f"Spin Orbitals {2* number_of_orbitals} \n"
f"Electrons (alpha beta) {n_alpha_beta} \n"
f"Multiplicity {args.magnetization} \n"
f"Seed {seed} \n"
f"Iterations {iterations} \n"
f"Learning rate {learning_rate} \n"
f"Samples per step {samples} \n"
f"Sampling buffer factor {s_buffer} \n"
f"Pretraining {bool(pretrain)} \n"
f"Valid Hilbert space size {binom(number_of_orbitals,alpha) * binom(number_of_orbitals,beta)} \n"
f"\n"
f"Model parameters: \n" 
f"System NADE \n"
f"Number of Dense layers 2 \n"
f"Number of Input features (Dense) {n_features} \n"
f"Number of Output Features (Dense) {2 ** (2*subsize)} \n"
f"Spatial orbtials per Subnetwork {subsize} \n"
f"Number of Phase Network layers {n_layers + 1} \n"
f"Number of Phase network features {n_features * 8}\n"
)





###############################
#Define Hilbert space
##############################

hi = nkx.hilbert.SpinOrbitalFermions(number_of_orbitals,1/2,n_alpha_beta)  # Number of electrons can be a list with [alpha,beta] spin!

#############################
#Define Hamiltonian, use prepared OpenFermion operator!
############################
file_name = os.path.join(loc, f'../data/{name}.hdf5')
ha =ElectronicOperator.from_openfermion(hi,file_name)

# Autoregressive neural network

ma = NADE(hilbert=hi, layers=n_layers, features=n_features,subsize=subsize,use_phase=True,dtype=jnp.float32)

#Sampler

sa = NADEWeightedSampler(hi)

# Optimizer

op = nk.optimizer.Adam(learning_rate= learning_rate,b2=0.99)

# Variational state
# With direct sampling, we don't need many samples in each step to form a
# Markov chain, and we don't need to discard samples

vs = WeightedMCState(sa,ma,n_samples=samples,seed=seed,sampling_buffer = s_buffer)

print(f"Model has {vs.n_parameters} parameters")


if load_state:
    with open(f"{name}.mpack", 'rb') as file:
        vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
if pretrain:
    if fci:
        vals, vecs, states = ha.FCI()
    elif mp2:
        vals, vecs, states = ha.MP2()
    elif cisd:
        vals, vecs, states = ha.CISD()
    else:
        print("No pretraining data available")
    if rescale:
        vecs = vecs * rescale
    data = np.column_stack((states,vecs))
    vecs = vecs ** 2
    mask = np.nonzero(np.round(vecs,16))    
    #data = data[mask]
    ha.h_init = False
    variables =  pretraining(
        vs,
        ma,
        data[mask],
        n_steps = pre_steps,
        norm = True)
    
    vs.parameters = variables
    ha.h_init = False
    states = jnp.asarray(states[mask],dtype=jnp.int8)
    
gs = nk.VMC(ha, op, variational_state=vs)
start = time.time()

print(f"Output is written to {name}_{seed}.log")
print(f"Model parameters are saved to {name}_{seed}.mpack")
print(f"Starting calculation at {time.ctime(start)}")

gs.run(n_iter=iterations, out = f"{name}_{seed}", show_progress = False)

end = time.time()

print("Sampling time  ", vs.sample_time)
print("Energy calculation time ", ha.time)
print("Total time ", end-start)

print("Final energy is : ", vs.expect(ha))
