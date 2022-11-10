from functools import partial
import netket as nk
import jax
from jax import numpy as jnp
from ._utils import _unpackbits
from netket.sampler import Sampler, SamplerState
from netket.utils import struct
from netket.utils.deprecation import warn_deprecation
from netket.utils.types import PRNGKeyT
from netket.experimental.hilbert import SpinOrbitalFermions


def add_states(states,count,p,i,subspace):
    '''
    Function for autoregressive sampling: Given a set of states with an conditional probability p, it returns the states with the highest conditional probability for a local degree of freedom i
    states: Array of size (batchsize,hilbert.size), partially sampled configurations
    count: conditinal probabilities of states
    p : conditional probabilities of the local degree of freedom i, only implemented for 2 local states 0 and 1

    '''

    batchsize,sites = states.shape
    n_orbitals = int(sites/2)
    subsize = 2**(2*subspace.shape[0])
    count = jnp.resize(count,batchsize * subsize)
    bin_rep = jnp.resize(jnp.arange(subsize,dtype=jnp.int16),(subsize * batchsize))
    bin_rep = jnp.reshape(bin_rep,(batchsize,subsize))
    bin_rep = jnp.ravel(bin_rep.T)
    state_idx = jnp.resize(jnp.arange(batchsize,dtype=jnp.int32),batchsize * subsize) # prepare state idx for sorting
    p = jnp.ravel(p.T)
    count *= p
    order = jnp.argsort(count)
    order = order[-batchsize:]
    state_idx = state_idx[order]
    states = states[state_idx]
    bin_rep = bin_rep[order]
    count = count[order]
    bin_rep = _unpackbits(bin_rep,16)
    #bin_rep = jnp.reshape(bin_rep,(batchsize,16))
    for k,spatial in enumerate(subspace):
            alpha =  jnp.asarray(bin_rep[:,2*k],dtype = jnp.int8)
            beta = jnp.asarray(bin_rep[:,2*k+1], dtype = jnp.int8)
            states = states.at[:,spatial].set(alpha)
            states = states.at[:,n_orbitals+spatial].set(beta)

    return states,count,state_idx


@struct.dataclass
class NADEWeightedSampler:
    '''
    ToDo: 
          - Combine with Sampler Baseclass for compatibility or include functions if necessary
          - Rename variable chain_length for readability
          - Combine recurrent sampler with NADE sampler to reduce redundancy

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
      

 
    @property
    def is_exact(sampler):
        """
        Returns `True` because the sampler is exact.
        The sampler is exact if all the samples are exactly distributed according to the
        chosen power of the variational state, and there is no correlation among them.
        """
        return True

    def _init_cache(self, model, σ, key):
        variables = model.init(key, σ, 0, method=model._conditional)
        if "cache" in variables:
            cache = variables["cache"]
        else:
            cache = None
        return cache

    @partial(jax.jit, static_argnums=(1, 3, 4))
    def sample(self, model, variables, chain_length, sampling_buffer):
        
        if "cache" in variables:
            variables, _ = variables.pop("cache")

        def scan_fun(carry, index):
            σ, count, subspace = carry
            sub = subspace[index]
            p = model.apply(
                variables,
                σ,
                index,
                method=model._conditional,
            )

            σ, count, _ = add_states(σ,count,p,index,sub)

            return (σ, count,subspace), None

        σ = jnp.zeros(
            ( sampling_buffer * chain_length , self.hilbert.size),
            dtype=self.dtype,
        )

        count = jnp.zeros(sampling_buffer * chain_length)
        count = count.at[0].set(1.0)
        subspace = jnp.arange(self.hilbert.size // 2,dtype=jnp.int16)
        subspace = jnp.reshape(subspace,(-1,model.subsize))

        indices = jnp.arange(subspace.shape[0])
        (σ, count,subspace), _ = jax.lax.scan(scan_fun, (σ,count,subspace), indices)
        σ = σ[-chain_length:]
        
        ampli = model.apply(
                variables,
                σ
            )
        
        p = jnp.exp(2*ampli.real)
        n = jnp.sum(p)
        
        return σ, p/n , jnp.exp(ampli)#count/jnp.sum(count), jnp.exp(ampli)#/jnp.sum(count)


@struct.dataclass
class RNNWeightedSampler:
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

    def _init_cache(self, model, σ, key):
        variables = model.init(key, σ, 0, method=model._conditional)
        if "cache" in variables:
            cache = variables["cache"]
        else:
            cache = None
        return cache

    @partial(jax.jit, static_argnums=(1, 3,4))
    def sample(self, model, variables, chain_length, sampling_buffer):
        if "cache" in variables:
            variables, _ = variables.pop("cache")

        def scan_fun(carry, index):
            σ, count, subspace, carry1, carry2, carry3 = carry
            sub = subspace[index]
            idx2 = index % 2
            idx3 = index % 3
            p,carry1,new_carry2,new_carry3 = model.apply(
                variables,
                σ,
                index,
                carry1,
                carry2[idx2],
                carry3[idx3],
                method=model._conditional,
            )

            σ, count, order = add_states(σ,count,p,index,sub)
            carry1 = carry1[order]
            carry2 = carry2[:,order]
            carry3 = carry3[:,order]
            carry2=carry2.at[idx2].set(new_carry2[order])
            carry3=carry3.at[idx3].set(new_carry3[order])
            return (σ, count,subspace,carry1,carry2,carry3), None
        
        σ = jnp.zeros(
            ( sampling_buffer * chain_length , self.hilbert.size),
            dtype=self.dtype,
        )
        count = jnp.zeros(sampling_buffer * chain_length)
        count = count.at[0].set(1.0)
        subspace = jnp.arange(self.hilbert.size // 2,dtype=jnp.int16)
        subspace = jnp.reshape(subspace,(-1,model.subsize))
        carry1,carry2,carry3 = model.apply(variables,sampling_buffer * chain_length, method = model.init_carry)
        indices = jnp.arange(subspace.shape[0])
        (σ, count,subspace,carry1,carry2,carry3), _ = jax.lax.scan(scan_fun, (σ,count,subspace,carry1,carry2,carry3), indices)
        σ = σ[-chain_length:]
        ampli = model.apply(
                variables,
                σ
            )
        p = jnp.exp(2*ampli.real)
        n = jnp.sum(p)
        return σ, p / n  , jnp.exp(ampli)



