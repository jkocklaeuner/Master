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
from jax.nn.initializers import normal,uniform ,he_normal,lecun_normal, zeros, ones
import netket as nk
from netket.utils.types import Array, DType, NNInitFunc
from netket.sampler import Sampler,SamplerState
from netket.experimental.hilbert import SpinOrbitalFermions
from functools import partial
from netket.utils import struct
from netket.utils.types import PRNGKeyT
from ._utils import _mask, _normalize

default_kernel_init = he_normal()
default_bias_init = zeros

def wrap_kernel_init(kernel_init, mask):
    """Correction to LeCun normal init."""

    corr = jnp.sqrt(mask.size / mask.sum())

    def wrapped_kernel_init(*args):
        return corr * mask * kernel_init(*args)

    return wrapped_kernel_init


class NADE(nn.Module):
   
    hilbert: SpinOrbitalFermions 
    """ Only SpinOrbitalfermions Supported yet"""
    features: int
    """number of features in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    layers: int
    subsize: int = 2 
    """ Specify which orbitals should be treated within one subnetwork, should be a 2d array with size (Number of subnetworks,spatial orbitals per subnetwork)"""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float32
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = default_bias_init
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""
    use_phase: bool = True
    """ Use a dense layer to apply a phase factor"""

    def setup(self):
        """
        Specify Network structure
        """

        n_orbitals = int(self.hilbert.size/2)
        self.subspaces = jnp.arange(n_orbitals)
        self.subspaces = jnp.reshape(self.subspaces,(-1,self.subsize))    
        self._orbitals = [
                Subnetwork(
                hilbert = self.hilbert,
                index = i, 
                features = self.features,
                layers = self.layers,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                subsize = self.subsize
            )
            for i,sub in enumerate(self.subspaces)
        ]  
       
        if self.use_phase is True:
            phase_features = [self.features * 8] * (self.layers) + [4] 
            self.phase = [ nk.nn.Dense(features=phase_features[i], dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init) for i in range(self.layers+1) ]
    
    def _log_psi(self,inputs):
        batchsize,n_sites = inputs.shape
        n_orbitals = int(n_sites/2)
        output = jnp.zeros(batchsize)
        alpha = jnp.empty((batchsize,0))
        beta = jnp.empty((batchsize,0))
        for i,orb in enumerate(self._orbitals):
            if i == 0:
                x = jnp.zeros((batchsize,1))
            else:
                sub = self.subspaces[i-1]
                for spatial in sub:
                    alpha = jnp.hstack((alpha,jnp.expand_dims(inputs[:,spatial],axis=-1)))
                    beta = jnp.hstack((beta,jnp.expand_dims(inputs[:,n_orbitals + spatial],axis=-1)))
                x = jnp.hstack((alpha,beta)) 
            x = orb(x)
            sub = self.subspaces[i]
            sub_idx = i * self.subsize
            index = inputs[:,sub_idx:(sub_idx + self.subsize)] * 2**(2*jnp.arange(self.subsize)) + inputs[:,(n_orbitals + sub_idx):(n_orbitals + sub_idx + self.subsize)] * 2**(1 + 2*jnp.arange(self.subsize))
            index = jnp.sum(index, axis = -1)
            output += x[jnp.arange(batchsize),index]
        return output

    def __call__(self,inputs):
        log_psi = self._log_psi(inputs)
        phase = jnp.zeros(log_psi.shape[0])
        if self.use_phase:
            n_orbitals = int(inputs.shape[1] / 2)
            x = jnp.hstack((inputs[:,:(n_orbitals - 1)],inputs[:,n_orbitals:(2*n_orbitals-1)]))
            log_psi = jnp.asarray(log_psi) 
            for i,lyr in enumerate(self.phase):
                x = nk.nn.relu(x)
                x = lyr(x)
            index = (inputs[:,(n_orbitals -1)] * 1 + inputs[:,(2*n_orbitals-1)] * 2) 
            x = x[jnp.arange(inputs.shape[0]),index]
            phase += jnp.reshape(x,log_psi.shape)
        return log_psi + 1j * phase * jnp.pi
    

    def conditionals(self, inputs: Array) -> Array:
        
        batchsize = inputs.shape[0]
        output = jnp.zeros((batchsize,2**(2*self.subsize),0)) #make this more general
        n_orbitals = int(inputs.shape[1] / 2)
        alpha = jnp.empty((batchsize,0))
        beta = jnp.empty((batchsize,0))
        
        for i,orbital in enumerate(self._orbitals):
             if i == 0:
                 x = jnp.zeros((inputs.shape[0],1))
             
             else:
                 sub = self.subspaces[i-1]
                 
                 for spatial in sub:
                     alpha = jnp.hstack((alpha,jnp.expand_dims(inputs[:,spatial],axis=-1)))
                     beta = jnp.hstack((beta,jnp.expand_dims(inputs[:,n_orbitals + spatial],axis=-1)))
                 
                 x = jnp.hstack((alpha,beta))
             
             sub = self.subspaces[i]
             lyr_output = jnp.reshape(orbital(x),(inputs.shape[0],2**(2*self.subsize),1))
             output = jnp.concatenate((output,lyr_output),axis=2)
        
        p = (jnp.exp(2*output.real))
        return p 

    def _conditional(self, inputs: Array, index) -> Array:
        
        return self.conditionals(inputs)[:,:,index]


class Subnetwork(nn.Module):
    """
    Feed forward with dense layers representing 2 * # Subsize Spin Orbitals

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
    bias_init: NNInitFunc = default_bias_init
    subsize: int = 2

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [2**(2*self.subsize)]
        self._layers = [ nk.nn.Dense(features=features[i], dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init) for i in range(self.layers) ]
    

    def __call__(self,states):
        """
        Function: Applies the orbital network to partially sampled states, appends another orbital occupation number to states
        inputs: [previously sampled states, count, log(psi)]
        outputs:[currently sampled stated, count, log(psi) ] 
        """
        x = jnp.asarray(states,dtype=jnp.int8)
        for i,lyr in enumerate(self._layers):
            x = nk.nn.relu(x)
            x = lyr(x) 
        m =  _mask(states,self.hilbert,self.index,self.subsize)
        x += m
        x = _normalize(x,2)
        return x


class RNN(nn.Module):

    """
    ToDo: 
            Include option to use unshared weights
            Add flexible RNN structure (# of recurrent layers, type of recurrent cell etc.) --> At the moment: Always 3 GRU cells
            Add flexible dilation option (Dilation: See Hibbat-Allah 2020 on recurrent network wave functions ) --> at the moment: dialation = 1,2,3
    """

    hilbert: SpinOrbitalFermions
    """ Only SpinOrbitalfermions Supported yet"""
    features: int
    """number of features in the first dense layer"""
    layers: int
    """number of recurrent layers"""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float32
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = default_bias_init
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""
    use_phase: bool = True
    """ Use a dense layer to apply a phase factor"""
    carry_size: int = 64  
    """Number of memory units (equals the carry size) for each recurrent layer"""
    subsize: int = 1
    """Number of subnetworks combined in a single Network layer"""

    def setup(self):
        n_orbitals = self.hilbert.size // 2
        self.subspaces = jnp.arange(n_orbitals)
        self.subspaces = jnp.reshape(self.subspaces,(-1,self.subsize))  
        self.dense1 = nk.nn.Dense(features = self.features,  dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init) 
        self.rnn1 = nn.GRUCell(bias_init=self.bias_init,dtype = self.dtype,kernel_init=self.kernel_init)  
        self.rnn2 = nn.GRUCell(bias_init=self.bias_init,dtype = self.dtype,kernel_init=self.kernel_init)  
        self.rnn3 = nn.GRUCell(kernel_init=self.kernel_init,bias_init=self.bias_init,dtype = self.dtype)
        self.dense2 = nk.nn.Dense(features = (2**(2*self.subsize)),  dtype=self.dtype, kernel_init=self.kernel_init, bias_init=self.bias_init) 
        
        if self.use_phase is True:
            phase_features = [self.features * 8] * (self.layers) + [4]
            self.phase = [ nk.nn.Dense(features=phase_features[i], dtype=self.dtype, kernel_init=self.kernel_init, bias_init=zeros) for i in range(self.layers+1) ]

    def init_carry(self,batchsize):
        key = nk.jax.PRNGKey()
        carry1 = self.rnn1.initialize_carry(key, (batchsize,), self.carry_size, init_fn=zeros)
        carry2 = jnp.asarray([self.rnn2.initialize_carry(key, (batchsize,), self.carry_size, init_fn=zeros) for i in range(2)] )
        carry3 = jnp.asarray([self.rnn3.initialize_carry(key, (batchsize,), self.carry_size, init_fn=zeros) for i in range(4) ] )
        return carry1,carry2,carry3

    def _p_kernel(self,states,i,carry1,carry2,carry3):
        x =  self.dense1(states)
        x = nk.nn.relu(x)
        carry1,x = self.rnn1(carry1,states)
        carry2,x = self.rnn2(carry2,x)
        carry3,x = self.rnn3(carry3,x)
        x = nk.nn.relu(x)
        x = self.dense2(x)
        m = _mask(states,self.hilbert,i,self.subsize)
        x = _normalize(x,2)
        x = x + m
        x = _normalize(x,2) 
        return x, carry1, carry2, carry3       
    
    def __call__(self,inputs):
        batchsize,n_sites = inputs.shape
        phase = jnp.zeros(batchsize)
        n_orbitals = n_sites // 2
        n_subspaces = n_orbitals // self.subsize
        output = jnp.zeros(batchsize)
        carry1, carry2, carry3 = self.init_carry(batchsize)
        states = jnp.zeros_like(inputs,dtype=jnp.int8)
        for i in range(n_subspaces):
            if i > 0:
                sub = self.subspaces[i-1]
                for j in sub:
                    states=states.at[:,j].set(inputs[:,j])
                    states=states.at[:,(n_orbitals + j)].set(inputs[:,(n_orbitals + j)])
            idx2 = i % 1
            idx3 = i % 1
            x, carry1, new_carry2, new_carry3 = self._p_kernel(states,i,carry1,carry2[idx2],carry3[idx3]) 
            carry2 = carry2.at[idx2].set(new_carry2)
            carry3 = carry3.at[idx3].set(new_carry3)
            sub_idx = i * self.subsize
            index = inputs[:,sub_idx:(sub_idx + self.subsize)] * 2**(2*jnp.arange(self.subsize)) + inputs[:,(n_orbitals + sub_idx):(n_orbitals + sub_idx + self.subsize)] * 2**(1 + 2*jnp.arange(self.subsize)) 
            index = jnp.sum(index, axis = -1)
            output += x[jnp.arange(batchsize),index]

        phase = jnp.zeros(batchsize)
        if self.use_phase:
           x = jnp.hstack((inputs[:,:(n_orbitals - 1)],inputs[:,n_orbitals:(2*n_orbitals-1)]))
           for i,lyr in enumerate(self.phase):
               x = nk.nn.relu(x)
               x = lyr(x)
           index = (inputs[:,(n_orbitals -1)] * 1 + inputs[:,(2*n_orbitals-1)] * 2) #use binary number for indexing
           
           x = x[jnp.arange(batchsize),index]
           phase += jnp.reshape(x,output.shape)
    
        return output + 1j * phase * jnp.pi

    def _conditional(self, inputs: Array,i,carry1,carry2,carry3) -> Array:
        p ,carry1,carry2,carry3= self._p_kernel(inputs,i,carry1,carry2,carry3)
        return jnp.exp(2 * p.real) , carry1 , carry2 , carry3
