import abc
from math import sqrt
from typing import Any, Callable, Iterable, Tuple, Union
import optax
import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.nn.initializers import zeros
from plum import dispatch
from functools import partial
import netket as nk
import netket.experimental as nkx
import numpy as np

################
#Pretraining
################

def pretraining(
vs,
model,
training_data,
n_steps: int = 1000,
norm: bool = False,
is_holomorphic: bool = False,
lr: float = 0.0005,
threshold: float=0.001,
batch_size: int = False,
optimizer = optax.adam):
#Model: netket model
#t_data: file with Inputdata /Targets as array
#n_steps: Is the model normalized? --> True for ARNN
#is_holomorphic: True for complex functions i guess
#lr: Learning rate
#batchsize: If only a subset of amplitudes should be used for each step
#optimizer: optimizer 

    if isinstance(training_data,str):
        training_data = np.loadtxt(training_data,dtype=np.complex128)
    init_data = td = jnp.asarray(training_data[:,:-1],dtype=jnp.int8)
    #initial_params = model.init(jax.random.PRNGKey(10),init_data) 
    #state = vs.parameters
    params = vs.parameters
    state = vs.model_state 
    if not batch_size:
        batch_size = len(training_data)
    
    @partial(jax.jit,static_argnums=(0))
    def update_step(apply_fn, states,amplis,opt_state, params, state):
        def loss(params,x,y):
            z,updated_state = apply_fn({'params': params, **state},
                                x, mutable=list(state.keys()))
            l=-jnp.log((jnp.exp(z)*jnp.conjugate(y)).sum()*(jnp.conjugate(jnp.exp(z))*y).sum())  
            #if not norm:
            #    l = -jnp.log(jnp.exp(-l)/((abs(z)**2).sum()*(abs(y)**2).sum()))
            #if not is_holomorphic:
            #l = jnp.sum((jnp.exp(2* z.real) - y**2)**2) 
            l = abs(l)
            return l, updated_state
        (l,updated_state), grads = jax.value_and_grad(loss,has_aux=True,allow_int=True)(params,states,amplis) # add stuff for holomorphic = complex models
        updates, opt_state = tx.update(grads, opt_state)  # Defined below.
        params = optax.apply_updates(params, updates)
        return opt_state, params, state, l

    #state, _ = params.pop('params')
    tx = optimizer(learning_rate=lr,b2=0.99)
    opt_state = tx.init(params)

    new_key = jax.random.PRNGKey(9)
    for i in range(n_steps):
        new_key = jax.random.split(new_key)
        train_batch = jax.random.choice(new_key,training_data,shape=(batch_size,),axis = 0,replace=False)
        td = jnp.asarray(train_batch[:,:-1],dtype=jnp.int8)
        labels = jnp.asarray(train_batch[:,-1],dtype=jnp.complex64)
        opt_state, params, state, loss = update_step(
            model.apply,td,labels, opt_state, params, state)
        if i % 1000 == 0:
            print(f'step {i} loss {loss}')
            if loss < threshold:
                break
    print(f'step {i} loss {loss}')
    return params

