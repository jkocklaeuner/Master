####################################
# Implementation of an fermionic ARNN following Barett 2022
# Modification of the NetKet.nn class: MasekdDense1D
# Includes symmetries directly, requires the use of a different sampler!
######################################

from typing import Any, Tuple
import flax
import jax
from flax import linen as nn
from jax import lax
from jax import numpy as jnp
from jax.nn.initializers import lecun_normal, zeros, ones
import netket as nk
from netket.utils.types import Array, DType, NNInitFunc
from netket.sampler import Sampler,SamplerState
from netket.experimental.hilbert import SpinOrbitalFermions
from functools import partial
from netket.utils import struct
from netket.utils.types import PRNGKeyT


default_kernel_init = lecun_normal()


def wrap_kernel_init(kernel_init, mask):
    """Correction to LeCun normal init."""

    corr = jnp.sqrt(mask.size / mask.sum())

    def wrapped_kernel_init(*args):
        return corr * mask * kernel_init(*args)

    return wrapped_kernel_init


class AutoregressiveNN(nn.Module):
   
    hilbert: SpinOrbitalFermions 
    """ Only SpinOrbitalfermions Supported yet"""
    features: int
    """number of features in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    layers: int
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float32
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = ones
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""
    use_phase: bool = True
    """ Use a dense layer to apply a phase factor"""

    ##################
# Fix Setup
    #################### 
    def setup(self):
        n_orbitals = self.hilbert.size    
        self._orbitals = [
                DenseOrbital(
                hilbert = self.hilbert,
                index = i, 
                features = self.features,
                layers = self.layers,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )
            for i in range(n_orbitals)
        ]
       
        if self.use_phase is True:
            phase_features = [self.features] * (self.layers) + [self.hilbert.local_size]
            self.phase = [ nk.nn.Dense(features=phase_features[i], dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init) for i in range(self.layers + 1 ) ]
    
    def _log_psi(self,inputs):
        batchsize = len(inputs)
        output = jnp.zeros(batchsize)
        for i,orb in enumerate(self._orbitals):
            x = orb(inputs[:,:i])
            index = jnp.asarray(inputs[:,i],dtype=jnp.int8)
            output += x[jnp.arange(batchsize),index]
    #    x = x.reshape((x.shape[0], -1, x.shape[-1]))
        return output

    def __call__(self,inputs):
        log_psi = self._log_psi(inputs)
        phase = jnp.zeros(log_psi.shape[0])
        if self.use_phase:
            x = jnp.asarray(inputs,dtype=jnp.int8)
            log_psi = jnp.asarray(log_psi) 
            for i,lyr in enumerate(self.phase):
                x = lyr(x)
                if i > 0:
                    x = nk.nn.log_cosh(x)
            x = jnp.sum(x,axis=-1)
            phase = jnp.reshape(x,log_psi.shape)
           # log_psi += 1j *  phase 
        return log_psi + 1j * phase
    """
    def __call__(self,inputs):
        batchsize = len(inputs)
        output = jnp.zeros(batchsize)
        for i,orb in enumerate(self._orbitals):
            x = orb(inputs[:,:i])
            index = jnp.asarray(inputs[:,i],dtype=jnp.int8)
            output += x[jnp.arange(batchsize),index]
            if i > 0:
                phase = states_to_numbers(inputs[:,:i])
                phase = jnp.reshape(phase,(batchsize,-1))
                for j,lyr in enumerate(self.phase):
                    #phase = lyr(phase)
                    if j > 0:
                        phase = nk.nn.log_cosh(phase)
                    phase = lyr(phase)
                output += 1j * phase[jnp.arange(batchsize),index]
        return output
    """
    def conditionals(self, inputs: Array) -> Array:
        output = jnp.zeros((inputs.shape[0],2,0))
        for i,orbital in enumerate(self._orbitals):
             lyr_output = jnp.reshape(orbital(inputs[:,:i]),(inputs.shape[0],2,1))
             output = jnp.concatenate((output,lyr_output),axis=2)
        #p = jnp.reshape(output,(output.shape[0],1,output.shape[-1]))
        p = (jnp.exp(2*output.real))
        return p 

    def _conditional(self, inputs: Array, index) -> Array:
        
        #output = self._orbitals[index](inputs)
        #output = jnp.absolute(jnp.exp(output))**2
        return self.conditionals(inputs)[:,:,index]

class DenseOrbital(nn.Module):
    """
    Feed forward with dense layers representing a single spin orbital
    hilbert: Hilbert space, contains all restrictions, has to be SpinOrbitalFermions at the moment 
    index: input dimension
    batch_size: Number of samples, will be used for truncating the output
    features: Number of neurons per layer
    layers: Number of layers
    dtype: DataType for the calculation
    precision: jax precision
    kernel_init: Initilization of weights
    bias_init: Initialization of bias
    
    Output: We'll see
    """
    
    hilbert: SpinOrbitalFermions
    features: int
    layers: int
    index: int
    dtype: DType = jnp.float64
    precision: Any = None
    kernel_init: NNInitFunc = default_kernel_init
    bias_init: NNInitFunc = zeros
    exclusive: bool = False

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        self._layers = [ nk.nn.Dense(features=features[i], dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init) for i in range(self.layers) ]
    def __call__(self,states):
        """
        Function: Applies the orbital network to partially sampled states, appends another orbital occupation number to states
        inputs: [previously sampled states, count, log(psi)]
        outputs:[currently sampled stated, count, log(psi) ] 
        """
        #x = states
        if self.index == 0:
            #key = jax.random.PRNGKey(0)
            x = jnp.zeros((len(states),1),dtype=jnp.int8)
            #x = jax.random.bernoulli(key,shape=(len(states),1)) 
        else:
            x = jnp.asarray(states,dtype=jnp.int8)
        for i,lyr in enumerate(self._layers):
            if i > 0:
                x = nk.nn.relu(x)
            x = lyr(x) 
        x = self.new_mask_output(states,x)
        x = _normalize(x,2)
        return x
 
    def new_mask_output(self,states,probs):
        p0 = probs[:,0]
        p1 = probs[:,1]
        i = self.index 
        batchsize = probs.shape[0]
        n_states = jnp.sum(states,axis=-1)
        alpha,beta = self.hilbert.n_fermions
        n_orbitals = int(self.hilbert.size/2)
        spin_i = i % n_orbitals 
        min_alpha = jax.nn.sigmoid( (-n_states - n_orbitals + spin_i + alpha) * 1000 )   #mask the probability to not occupy the orbital if it violates the minimal number of alpha electrons
        max_alpha = jax.nn.sigmoid((n_states - alpha - (i - spin_i)) * 1000)            #mask the probability to occupy the orbital if it violates the maximum number of alpha electrons
        min_beta = jax.nn.sigmoid( (-n_states + alpha + beta - (self.hilbert.size - i)) * 1000 ) 
        max_beta = jax.nn.sigmoid( (n_states - alpha - beta ) * 1000 ) 
        p0 += (min_alpha + min_beta) * -1000 #= exp(-100) = 0
        p1 += (max_alpha + max_beta) * -1000 
        return jnp.vstack((p0,p1)).T

    


def _normalize(log_psi: Array, machine_pow: int) -> Array:
    """
    Normalizes log_psi to have L2-norm 1 along the last axis.
    """
    psi = jnp.exp(log_psi)
    norm = jnp.linalg.norm(psi,axis=-1,keepdims=True)
    return jnp.log(psi/norm)     

#    return log_psi - 1 / machine_pow * jax.scipy.special.logsumexp(
#        machine_pow * jnp.absolute(log_psi), axis=-1, keepdims=True )


