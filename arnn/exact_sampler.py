from functools import partial
import netket as nk
import jax
from jax import numpy as jnp
from netket import nn
from netket.sampler import Sampler, SamplerState
from netket.utils import struct
from netket.utils.deprecation import warn_deprecation
from netket.utils.types import PRNGKeyT
from netket.experimental.hilbert import SpinOrbitalFermions


@struct.dataclass
class ARWeightedSampler:
    '''
    ToDo: Combine with Sampler Baseclass for compatibility or include functions if necessary
    Samples individual configuration from a normalized autoregressive network
    Should be used with a weighted MC State, since it is not compatible with non-weighted states
    Input: hilbert, dtype (for electronic structure calculations: int8 sufficient!
    Output: Sampled states, probabilities, maybe amplitudes ? 
    '''
    hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    """The Hilbert space to sample."""
    dtype: jnp.dtype = struct.field(pytree_node=False, default=jnp.int8)
    """The dtype of the states sampled."""
    machine_pow: int = 2
    #hilbert: SpinOrbitalFermions
    #dtype: jnp.dtype=jnp.int8
    #def __init__(self,hilbert,dtype=jnp.int8):
    #    self.hilbert =  hilbert
    #    self.dtype = dtype
    
    @property
    def is_exact(sampler):
        """
        Returns `True` because the sampler is exact.
        The sampler is exact if all the samples are exactly distributed according to the
        chosen power of the variational state, and there is no correlation among them.
        """
        return True


    @partial(jax.jit, static_argnums=(1, 3))
    def sample(self, model, variables, chain_length):
        states = self.hilbert.all_states() 
        ampli = model.apply(variables,states) 
 
        p = abs(jnp.exp(ampli))**2
        return states, p/jnp.sum(p) , jnp.exp(ampli)/jnp.sum(p)#count/jnp.sum(count), jnp.exp(ampli)#/jnp.sum(count)

