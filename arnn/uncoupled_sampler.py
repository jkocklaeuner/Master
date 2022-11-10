from functools import partial
import netket as nk
import jax
from jax import numpy as jnp

from netket.sampler import Sampler, SamplerState
from netket.utils import struct
from netket.utils.deprecation import warn_deprecation
from netket.utils.types import PRNGKeyT
from netket.experimental.hilbert import SpinOrbitalFermions

def add_states(states,count,p,i):
    '''
    Function for autoregressive sampling: Given a set of states with an conditional probability p, it returns the states with the highest conditional probability for a local degree of freedom i
    states: Array of size (batchsize,hilbert.size), partially sampled configurations
    count: conditinal probabilities of states
    p : conditional probabilities of the local degree of freedom i, only implemented for 2 local states 0 and 1 

    '''
    
    batchsize = states.shape[0]
    states = jnp.vstack((states,states))
    count = jnp.hstack((count,count))
    p = jnp.hstack((p[:,0],p[:,1]))
    
    local = jnp.hstack((jnp.zeros(batchsize,dtype=jnp.int8),jnp.ones(batchsize,dtype=jnp.int8)))
    new_states = states.at[:,i].set(local)
    count *= p 
    order = jnp.argsort(count)
    count = count[order]
    new_states = new_states[order]
    return new_states[batchsize:,:],count[batchsize:]

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

    def _init_cache(self, model, σ, key):
        variables = model.init(key, σ, 0, method=model._conditional)
        if "cache" in variables:
            cache = variables["cache"]
        else:
            cache = None
        return cache

    @partial(jax.jit, static_argnums=(1, 3))
    def sample(self, model, variables, chain_length, sampling_buffer):
        if "cache" in variables:
            variables, _ = variables.pop("cache")

        def scan_fun(carry, index):
            σ, count,cache = carry
            if cache:
                _variables = {**variables, "cache": cache}
            else:
                _variables = variables

            p,mutables = model.apply(
                _variables,
                σ,
                index,
                method=model._conditional,
                mutable =["cache"]
            )

            #local_states = jnp.asarray(
            #    sampler.hilbert.local_states, dtype=sampler.dtype
            #)
            #new_σ = batch_choice(key, local_states, p)
           # log_p = jnp.log(p/2)
           # log_p = new_mask_output(index,σ,log_p)
           # log_p = normalize(log_p, 2) 
           # p = jnp.exp(2*log_p)
            σ, count = add_states(σ,count,p,index)
            if "cache" in mutables:
                cache = mutables["cache"]
            else:
                cache = None
            return (σ, count,cache), None

        σ = jnp.zeros(
            (sampling_buffer * chain_length , self.hilbert.size),
            dtype=self.dtype,
        )
        key_init = nk.jax.PRNGKey()
        cache = self._init_cache(model, σ, key_init)
        count = jnp.zeros(sampling_buffer * chain_length)
        count = count.at[0].set(1.0)
        # Initialize `cache` before generating each sample,
        # even if `variables` is not changed and `reset` is not called

        indices = jnp.arange(self.hilbert.size)
        (σ, count,cache), _ = jax.lax.scan(scan_fun, (σ,count,cache), indices)
        σ = σ[-chain_length:]
        count = count[-chain_length:]
        ampli = model.apply(
                variables,
                σ
            )
        n = jnp.sum(count)
        return σ, count / n , jnp.exp(ampli)

