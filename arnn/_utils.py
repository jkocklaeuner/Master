import jax
import jax.numpy as jnp
from netket.utils.types import Array


def _mask(states,hilbert,index,subspace_size):
        batchsize, n_local = states.shape
        n_subspace = 2**(2*subspace_size)
        alpha,beta = hilbert.n_fermions
        n_orbitals = hilbert.size // 2
        i = (n_orbitals // subspace_size - index - 1) * subspace_size
        n_alpha = jnp.sum(states[:,:(n_local // 2)],axis=-1)
        n_beta = jnp.sum(states[:,(n_local // 2):],axis=-1)
        n_alpha = jnp.resize(n_alpha,(n_subspace,batchsize))
        n_beta = jnp.resize(n_beta, (n_subspace,batchsize))
        bin_rep = jnp.arange(n_subspace)
        bin_rep = _unpackbits(bin_rep,16)
        alpha_occ = jnp.sum(bin_rep[:,::2],axis = -1)
        beta_occ = jnp.sum(bin_rep[:,1::2],axis = -1)
        alpha_occ = jnp.resize(alpha_occ, (batchsize,n_subspace))
        beta_occ = jnp.resize(beta_occ, (batchsize,n_subspace))
        dummy = jnp.zeros((batchsize,n_subspace))
        min_alpha = jnp.heaviside( (  alpha - n_alpha.T - alpha_occ - i), dummy ) # Example: i = 2, n = 6, n_occ = 1, a = 4: n_a = 0 --> masking, n_a = 1 -> masking 
        max_alpha = jnp.heaviside( ( n_alpha.T - alpha + alpha_occ), dummy) # Example: i = 2, n=6, n_occ = 1, a = 2
        min_beta =  jnp.heaviside( (  beta - n_beta.T - beta_occ - i), dummy)
        max_beta =  jnp.heaviside( (n_beta.T - beta + beta_occ), dummy)
        out = (min_alpha + min_beta + max_alpha + max_beta) * -200
        return out

def _normalize(log_psi: Array, machine_pow: int) -> Array:
    """
    Normalizes log_psi to have L2-norm 1 along the last axis.
    """
    return log_psi - 1 / machine_pow * jax.scipy.special.logsumexp(
        machine_pow * log_psi.real, axis=-1, keepdims=True )

def states_to_numbers(states):
    #converts state with binary local states to number which can be used as hash values
    numbers = jnp.sum(states*(2**(jnp.arange(states.shape[-1]))),axis=-1)
    numbers = numbers.flatten()
    return numbers

def _unpackbits(x, num_bits):
    xshape = x.shape
    x = jnp.reshape(x,(-1, 1))
    mask = 2**jnp.reshape(jnp.arange(num_bits, dtype=x.dtype),(1, num_bits))
    return jnp.reshape(jnp.asarray(x & mask, dtype = bool),(xshape[0],num_bits))

