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
import netket as nk
import netket.experimental as nkx
import openfermion as of
import pickle
import numpy as np
import time
from netket.experimental.hilbert import SpinOrbitalFermions
from arnn.FermionOperator2nd import FermionOperator2nd
import argparse
from scipy.special import binom
import os

loc = os.path.dirname(__file__)


parser = argparse.ArgumentParser(description='Run a single VMC optimization with a RBM network', allow_abbrev=True)
parser.add_argument('-n','--name', nargs='?', required = True,
                        help='Molecule Name, should be equal to the hamiltonian file name')
parser.add_argument('-lr', '--learningrate',default = "0.05",type=float,help = "Learning rate of the optimizer, check optax or NetKet for details")
parser.add_argument('-ls', '--loadstate', default = False,type=bool,help = "Boolean, load a state from a previous calculation with the same settings")
parser.add_argument('-s','--samples',default = 100,type=int,help="Number of unique samples per iteration")
parser.add_argument('-i','--iterations',default = 10000,type=int, help = "Total number of optimization iterations")
parser.add_argument('-m','--magnetization',type=int,default = 0, help="Multiplicity")
parser.add_argument('-e','--electrons',type=int, help="Number of electrons")
parser.add_argument('-o','--orbitals',type=int, help = "Number of spatial orbitals")
parser.add_argument('-se','--seed',type = int,default = 0,  help = "Seed for the random initialization procedure")
parser.add_argument('-f','--features',type=int,default = 1, help = "Hidden layer density, number of hidden layer neurons = #spin orbitals * density" )

args = parser.parse_args()


name = args.name
load_state = args.loadstate
number_of_orbitals = args.orbitals
alpha = int(args.electrons / 2 + args.magnetization)
beta = int(args.electrons / 2 - args.magnetization)
n_alpha_beta =(alpha,beta)
learning_rate = args.learningrate
n_features = args.features
samples = args.samples
iterations = args.iterations
seed = args.seed


print(f"Running script rbm.py \n"
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
f"Valid Hilbert space size {binom(number_of_orbitals,alpha) * binom(number_of_orbitals,beta)} \n"
f"\n"
f"Model parameters: \n" 
f"System RBM \n"
"Dtype Complex \n"
f"Hidden layer density {n_features}"
)





###############################
#Define Hilbert space
##############################

hi = nkx.hilbert.SpinOrbitalFermions(number_of_orbitals,1/2,n_alpha_beta)  # Number of electrons can be a list with [alpha,beta] spin!

g= nk.graph.Hypercube(length=2*number_of_orbitals, n_dim=1, pbc=False)

#############################
#Define Hamiltonian, use prepared OpenFermion operator!
############################
file_name = os.path.join(loc, f'../data/{name}_fermion_hamiltonian.pkl')

with open(file_name, "rb") as input_file:
        OF_hamiltonian = pickle.load(input_file)

ha = FermionOperator2nd.from_openfermion(hi,OF_hamiltonian,convert_spin_blocks=True)
ma = nk.models.RBM(alpha=n_features,dtype=jnp.complex64)
sa = nk.sampler.MetropolisExchange(hi,graph=g,dtype=jnp.int8,n_chains=1024)
op = nk.optimizer.Sgd(learning_rate= learning_rate)
sr = nk.optimizer.SR(diag_shift=0.01)
vs = nk.vqs.MCState(sa,ma,n_samples=samples,seed =seed,chunk_size = 1024,n_discard_per_chain = 10 * number_of_orbitals)

print(f"Model has {vs.n_parameters} parameters")


if load_state:
    with open(f"{name}.mpack", 'rb') as file:
        vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
    
gs = nk.VMC(ha, op, variational_state=vs)
start = time.time()

print(f"Output is written to {name}_{seed}.log")
print(f"Model parameters are saved to {name}_{seed}.mpack")
print(f"Starting calculation at {time.ctime(start)}")

gs.run(n_iter=iterations, out = f"{name}_{seed}", show_progress = False)

end = time.time()

print("Total time ", end-start)

print("Final energy is : ", vs.expect(ha))
