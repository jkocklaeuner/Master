import warnings
from functools import partial
from typing import Any , Callable, Dict, Optional, Tuple
import numpy as np
from netket.utils.dispatch import dispatch, TrueT, Bool
from netket import jax as nkjax
import jax
from jax import numpy as jnp
from netket.vqs import expect
from netket.vqs.mc.common import check_hilbert, get_local_kernel_arguments, get_local_kernel
from netket.vqs.mc import (
    kernels,
    get_local_kernel,
    get_local_kernel_arguments,
)
import flax
from flax import serialization
from arnn.weighted_qgt import QGTJacobianPyTree
import time
from netket import jax as nkjax
from netket import nn
#from netket.stats import Stats,statistics
from netket.stats import statistics as mpi_statistics, mean as mpi_mean, Stats
from netket.operator import AbstractOperator
from netket.sampler import Sampler, SamplerState
from netket.utils import (
    maybe_wrap_module,
    deprecated,
    warn_deprecation,
    mpi,
    wrap_afun,
    wrap_to_support_scalar,
)
from netket.utils.types import PyTree, SeedT, NNInitFunc, Array
from netket.optimizer import LinearOperator
from netket.optimizer.qgt import QGTAuto

from netket.vqs.base import VariationalState, expect, expect_and_grad
#from netket.vqs.mc import get_local_kernel, get_local_kernel_arguments
from arnn.fermion_operator import ElectronicOperator
from netket.jax import vjp as nkvjp
from netket.jax import HashablePartial
from jax.tree_util import tree_map
from netket.jax.utils import (
    tree_ravel,
    is_complex,
    is_complex_dtype,
    tree_size,
    eval_shape,
    tree_leaf_iscomplex,
    tree_ishomogeneous,
    dtype_complex,
    dtype_real,
    maybe_promote_to_complex,
    tree_conj,
    tree_dot,
    tree_cast,
    tree_axpy,
    tree_to_real,
    compose,
    mpi_split,
    PRNGKey,
    PRNGSeq,
)



def check_chunk_size(n_samples, chunk_size):
    n_samples_per_rank = n_samples // mpi.n_nodes

    if chunk_size is not None:
        if chunk_size < n_samples_per_rank and n_samples_per_rank % chunk_size != 0:
            raise ValueError(
                f"chunk_size={chunk_size}`<`n_samples_per_rank={n_samples_per_rank}, "
                "chunk_size is not an integer fraction of `n_samples_per rank`. This is"
                "unsupported. Please change `chunk_size` so that it divides evenly the"
                "number of samples per rank or set it to `None` to disable chunking."
            )


def _is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n - 1) == 0)


@partial(jax.jit, static_argnums=0)
def jit_evaluate(fun: Callable, *args):
    """
    call `fun(*args)` inside of a `jax.jit` frame.
    Args:
        fun: the hashable callable to be evaluated.
        args: the arguments to the function.
    """
    return fun(*args)


class WeightedMCState(VariationalState):
    """Variational State for a Variational Neural Quantum State.
    The state is sampled according to the provided sampler.
    """

    # model: Any
    # """The model"""
    model_state: Optional[PyTree]
    """An Optional PyTree encoding a mutable state of the model that is not trained."""

    _sampler: Any
    """The sampler used to sample the Hilbert space."""

    _samples: Optional[jnp.ndarray] = None
    """Cached samples obtained with the last sampling."""

    _init_fun: Callable = None
    """The function used to initialise the parameters and model_state."""
    _apply_fun: Callable = None
    """The function used to evaluate the model."""

    _chunk_size: Optional[int] = None
    

    def __init__(
        self,
        sampler,
        model,
        *,
        n_samples: int = 1,
        sampling_buffer: int  = 1,
        chunk_size: Optional[int] = None,
        variables: Optional[PyTree] = None,
        init_fun: NNInitFunc = None,
        apply_fun: Callable = None,
        sample_fun: Callable = None,
        seed: Optional[SeedT] = None,
        sampler_seed: Optional[SeedT] = None,
        mutable: bool = False,
        training_kwargs: Dict = {},
    ):
        """
        Constructs the MCState.
        Args:
            sampler: The sampler
            model: (Optional) The model. If not provided, you must provide init_fun and apply_fun.
            n_samples: the total number of samples across chains and processes when sampling (default=1000).
            parameters: Optional PyTree of weights from which to start.
            seed: rng seed used to generate a set of parameters (only if parameters is not passed). Defaults to a random one.
            sampler_seed: rng seed used to initialise the sampler. Defaults to a random one.
            mutable: Dict specifying mutable arguments. Use it to specify if the model has a state that can change
                during evaluation, but that should not be optimised. See also flax.linen.module.apply documentation
                (default=False)
            init_fun: Function of the signature f(model, shape, rng_key, dtype) -> Optional_state, parameters used to
                initialise the parameters. Defaults to the standard flax initialiser. Only specify if your network has
                a non-standard init method.
            variables: Optional initial value for the variables (parameters and model state) of the model.
            apply_fun: Function of the signature f(model, variables, σ) that should evaluate the model. Defaults to
                `model.apply(variables, σ)`. specify only if your network has a non-standard apply method.
            sample_fun: Optional function used to sample the state, if it is not the same as `apply_fun`.
            training_kwargs: a dict containing the optional keyword arguments to be passed to the apply_fun during training.
                Useful for example when you have a batchnorm layer that constructs the average/mean only during training.
            n_discard: DEPRECATED. Please use `n_discard_per_chain` which has the same behaviour.
        """
        super().__init__(sampler.hilbert)

        # Init type 1: pass in a model
        if model is not None:
            # extract init and apply functions
            # Wrap it in an HashablePartial because if two instances of the same model are provided,
            # model.apply and model2.apply will be different methods forcing recompilation, but
            # model and model2 will have the same hash.
            _, model = maybe_wrap_module(model)

            self._model = model

            self._init_fun = nkjax.HashablePartial(
                lambda model, *args, **kwargs: model.init(*args, **kwargs), model
            )
            self._apply_fun = wrap_to_support_scalar(
                nkjax.HashablePartial(
                    lambda model, pars, x, **kwargs: model.apply(pars, x, **kwargs),
                    model,
                )
            )

        elif apply_fun is not None:
            self._apply_fun = wrap_to_support_scalar(apply_fun)

            if init_fun is not None:
                self._init_fun = init_fun
            elif variables is None:
                raise ValueError(
                    "If you don't provide variables, you must pass a valid init_fun."
                )

            self._model = wrap_afun(apply_fun)

        else:
            raise ValueError(
                "Must either pass the model or apply_fun, otherwise how do you think we"
                "gonna evaluate the model?"
            )

        # default argument for n_samples/n_samples_per_rank

        if sample_fun is not None:
            self._sample_fun = sample_fun
        else:
            self._sample_fun = self._apply_fun

        self.mutable = mutable
        self.training_kwargs = flax.core.freeze(training_kwargs)

        if variables is not None:
            self.variables = variables
        else:
            self.init(seed, dtype=sampler.dtype)

        if sampler_seed is None and seed is not None:
            key, key2 = jax.random.split(nkjax.PRNGKey(seed), 2)
            sampler_seed = key2

        self._sampler_seed = sampler_seed
        self.sampler = sampler
        self._n_samples = n_samples
        self.sampling_buffer = sampling_buffer
        self.chunk_size = chunk_size
        self.sample_time = 0
        self.grad_time = 0


    def init(self, seed=None, dtype=None):
        """
        Initialises the variational parameters of the variational state.
        """
        if self._init_fun is None:
            raise RuntimeError(
                "Cannot initialise the parameters of this state"
                "because you did not supply a valid init_function."
            )

        if dtype is None:
            dtype = self.sampler.dtype

        key = nkjax.PRNGKey(seed)

        dummy_input = jnp.zeros((1, self.hilbert.size), dtype=dtype)

        variables = jit_evaluate(self._init_fun, {"params": key}, dummy_input)
        self.variables = variables

    @property
    def model(self) -> Optional[Any]:
        """Returns the model definition of this variational state.
        This field is optional, and is set to `None` if the variational state has
        been initialized using a custom function.
        """
        return self._model

    @property
    def sampler(self) :
        """The Monte Carlo sampler used by this Monte Carlo variational state."""
        return self._sampler
    @property
    def n_samples(self) -> int:
        return self._n_samples
    @sampler.setter
    def sampler(self, sampler):
        # Save the old `n_samples` before the new `sampler` is set.
        # `_chain_length == 0` means that this `MCState` is being constructed.
        self._sampler = sampler

        #This might be necessary for sampling with a distribution --> At the moment, only exact sampling is considered
        #self.sampler_state = self.sampler.init_state(
        #    self.model, self.variables, seed=self._sampler_seed
        #)
        #self._sampler_state_previous = self.sampler_state

        # Update `n_samples`, `n_samples_per_rank`, and `chain_length` according
        # to the new `sampler.n_chains`.
        # If `n_samples` is divisible by the new `sampler.n_chains`, it will be
        # unchanged; otherwise it will be rounded up.
        # If the new `n_samples_per_rank` is not divisible by `chunk_size`, a
        # `ValueError` will be raised.
        # `_chain_length == 0` means that this `MCState` is being constructed.
        self.reset()

    @property
    def chunk_size(self) -> int:
        """
        Suggested *maximum size* of the chunks used in forward and backward evaluations
        of the Neural Network model. If your inputs are smaller than the chunk size
        this setting is ignored.
        This can be used to lower the memory required to run a computation with a very
        high number of samples or on a very large lattice. Notice that inputs and
        outputs must still fit in memory, but the intermediate computations will now
        require less memory.
        This option comes at an increased computational cost. While this cost should
        be negligible for large-enough chunk sizes, don't use it unless you are memory
        bound!
        This option is an hint: only some operations support chunking. If you perform
        an operation that is not implemented with chunking support, it will fall back
        to no chunking. To check if this happened, set the environment variable
        `NETKET_DEBUG=1`.
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: Optional[int]):
        # disable chunks if it is None
        if chunk_size is None:
            self._chunk_size = None
            return

        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer. ")

        if not _is_power_of_two(chunk_size):
            warnings.warn(
                "For performance reasons, we suggest to use a power-of-two chunk size."
            )

        check_chunk_size(self.n_samples, chunk_size)

        self._chunk_size = chunk_size

    def reset(self):
        """
        Resets the sampled states. This method is called automatically every time
        that the parameters/state is updated.
        """
        self._samples = None
        self._weights = None

    def sample(
        self,
        *,
        n_samples: Optional[int] = None,
    ) :
        """
        Sample a certain number of configurations.
        If one among chain_length or n_samples is defined, that number of samples
        are generated. Otherwise the value set internally is used.
        Args:
            chain_length: The length of the markov chains.
            n_samples: The total number of samples across all MPI ranks.
        """
        start = time.time()
        if n_samples is None:
            n_samples = self.n_samples

        # Store the previous sampler state, for serialization purposes
       # self._sampler_state_previous = self.sampler_state

        self._samples, self._weights, self._amplitudes = self.sampler.sample(
            self.model,
            self.variables,
            chain_length=n_samples,
            sampling_buffer = self.sampling_buffer
        )
        end = time.time()
        self.sample_time += end-start
        return self._samples, self._weights, self._amplitudes

    @property
    def samples(self) -> jnp.ndarray:
        """
        Returns the set of cached samples.
        The samples returned are guaranteed valid for the current state of
        the variational state. If no cached parameters are available, then
        they are sampled first and then cached.
        To obtain a new set of samples either use
        :meth:`~MCState.reset` or :meth:`~MCState.sample`.
        """
        if self._samples is None:
            self.sample()
        return (self._samples, self._weights, self._amplitudes)

    def log_value(self, σ: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the variational state for a batch of states and returns
        the logarithm of the amplitude of the quantum state. For pure states,
        this is :math:`log(<σ|ψ>)`, whereas for mixed states this is
        :math:`log(<σr|ρ|σc>)`, where ψ and ρ are respectively a pure state
        (wavefunction) and a mixed state (density matrix).
        For the density matrix, the left and right-acting states (row and column)
        are obtained as :code:`σr=σ[::,0:N]` and :code:`σc=σ[::,N:]`.
        Given a batch of inputs (Nb, N), returns a batch of outputs (Nb,).
        """
        return jit_evaluate(self._apply_fun, self.variables, σ)

    #Check whether this is necessary for QGT
    def local_estimators(
        self, op: AbstractOperator, *, chunk_size: Optional[int] = None
    ):
        r"""
        Compute the local estimators for the operator :code:`op` (also known as local energies
        when :code:`op` is the Hamiltonian) at the current configuration samples :code:`self.samples`.
        .. math::
            O_\mathrm{loc}(s) = \frac{\langle s | \mathtt{op} | \psi \rangle}{\langle s | \psi \rangle}
        .. warning::
            The samples differ between MPI processes, so returned the local estimators will
            also take different values on each process. To compute sample averages and similar
            quantities, you will need to perform explicit operations over all MPI ranks.
            (Use functions like :code:`self.expect` to get process-independent quantities without
            manual reductions.)
        Args:
            op: The operator.
            chunk_size: Suggested maximum size of the chunks used in forward and backward evaluations
                of the model. (Default: :code:`self.chunk_size`)
        """
        return local_estimators(self, op, chunk_size=chunk_size)

    # override to use chunks
    def expect(self, Ô: AbstractOperator) -> Stats:
        r"""Estimates the quantum expectation value for a given operator O.
            In the case of a pure state $\psi$, this is $<O>= <Psi|O|Psi>/<Psi|Psi>$
            otherwise for a mixed state $\rho$, this is $<O> = \Tr[\rho \hat{O}/\Tr[\rho]$.
        Args:
            Ô: the operator O.
        Returns:
            An estimation of the quantum expectation value <O>.
        """
        return expect(self, Ô, self.chunk_size)

    # override to use chunks
    def expect_and_grad(
        self,
        Ô: AbstractOperator,
        *,
        mutable: Optional[Any] = None,
        use_covariance: Optional[bool] = True,
    ) -> Tuple[Stats, PyTree]:
        r"""Estimates both the gradient of the quantum expectation value of a given operator O.
        Args:
            Ô: the operator Ô for which we compute the expectation value and it's gradient
            mutable: Can be bool, str, or list. Specifies which collections in the model_state should
                     be treated as  mutable: bool: all/no collections are mutable. str: The name of a
                     single mutable  collection. list: A list of names of mutable collections.
                     This is used to mutate the state of the model while you train it (for example
                     to implement BatchNorm. Consult
                     `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                     for a more in-depth explanation).
            use_covariance: whether to use the covariance formula, usually reserved for
                hermitian operators, ⟨∂logψ Oˡᵒᶜ⟩ - ⟨∂logψ⟩⟨Oˡᵒᶜ⟩
        Returns:
            An estimation of the quantum expectation value <O>.
            An estimation of the average gradient of the quantum expectation value <O>.
        """
        if mutable is None:
            mutable = self.mutable

        return expect_and_grad(
            self, Ô, use_covariance, self.chunk_size, mutable=mutable
        )

    def quantum_geometric_tensor(
        self, qgt_T: LinearOperator = QGTAuto()
    ) -> LinearOperator:
        r"""Computes an estimate of the quantum geometric tensor G_ij.
        This function returns a linear operator that can be used to apply G_ij to a given vector
        or can be converted to a full matrix.
        Args:
            qgt_T: the optional type of the quantum geometric tensor. By default it's automatically selected.
        Returns:
            nk.optimizer.LinearOperator: A linear operator representing the quantum geometric tensor.
        """
        return partial(QGTJacobianPyTree)

    def to_array(self, normalize: bool = True) -> jnp.ndarray:
        return nn.to_array(
            self.hilbert, self._apply_fun, self.variables, normalize=normalize
        )

    def __repr__(self):
        return (
            "MCState("
            + "\n  hilbert = {},".format(self.hilbert)
            + "\n  sampler = {},".format(self.sampler)
            + "\n  n_samples = {},".format(self.n_samples)
            + "\n  n_parameters = {})".format(self.n_parameters)
        )

    def __str__(self):
        return (
            "MCState("
            + "hilbert = {}, ".format(self.hilbert)
            + "sampler = {}, ".format(self.sampler)
            + "n_samples = {})".format(self.n_samples)
        )


@partial(jax.jit, static_argnames=("kernel", "apply_fun", "shape"))
def _local_estimators_kernel(kernel, apply_fun, shape, variables, samples, extra_args):
    O_loc = kernel(apply_fun, variables, samples, extra_args)

    # transpose O_loc so it matches the (n_chains, n_samples_per_chain) shape
    # expected by netket.stats.statistics.
    return O_loc.reshape(shape).T


def local_estimators(
    state: WeightedMCState, op: ElectronicOperator, *, chunk_size: Optional[int]
):
    s, extra_args = get_local_kernel_arguments(state, op)

    shape = s.shape
    if jnp.ndim(s) != 2:
        s = s.reshape((-1, shape[-1]))

    if chunk_size is None:
        chunk_size = state.chunk_size  # state.chunk_size can still be None

    if chunk_size is None:
        kernel = get_local_kernel(state, op)
    else:
        kernel = get_local_kernel(state, op, chunk_size)

    return _local_estimators_kernel(
        kernel, state._apply_fun, shape[:-1], state.variables, s, extra_args
    )


# serialization
def serialize_MCState(vstate):
    state_dict = {
        "variables": serialization.to_state_dict(vstate.variables),
        "n_samples": vstate.n_samples,
        "chunk_size": vstate.chunk_size,
    }
    return state_dict


def deserialize_MCState(vstate, state_dict):
    import copy

    new_vstate = copy.copy(vstate)
    new_vstate.reset()

    new_vstate.variables = serialization.from_state_dict(
        vstate.variables, state_dict["variables"]
    )
    new_vstate.n_samples = state_dict["n_samples"]
    new_vstate.chunk_size = state_dict["chunk_size"]

    return new_vstate


serialization.register_serialization_state(
    WeightedMCState,
    serialize_MCState,
    deserialize_MCState,
)

@dispatch
def get_local_kernel_arguments(vstate: WeightedMCState, Ô: ElectronicOperator):  # noqa: F811
    #check_hilbert(vstate.hilbert, Ô.hilbert) #Check this 

    σ , p , a= vstate.samples
    mels, weights = Ô.get_weighted_mels(σ , p , a) #Implement this: mels = weighted matrix elements per state, weights = sum of the weigths of each state 
    return σ, (weights, mels)

@dispatch
def get_local_kernel(vstate: WeightedMCState, Ô: ElectronicOperator):  # noqa: F811
    return weighted_value_kernel #Define kernel function

def batch_discrete_kernel(kernel):
    """
    Batch a kernel that only works with 1 sample so that it works with a
    batch of samples.
    Works only for discrete-kernels who take two args as inputs
    """

    def vmapped_kernel(logpsi, pars, σ, args):
        """
        local_value kernel for MCState and generic operators
        """
        weights,mels = args
   
      #  if jnp.ndim(σ) == 2:
           # σp = σp.reshape((-1,σ.shape[0]))
      #      mels = mels.reshape(σ.shape[:-1])
      #      weights =  mels.reshape(σ.shape[:-1])
        vkernel = jax.vmap(kernel, in_axes=(None, None, 0, (0, 0)), out_axes=0)
        return vkernel(logpsi, pars, σ, (weights, mels))

    return vmapped_kernel


@batch_discrete_kernel
def weighted_value_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    weights, mels = args

    return mels #/ jnp.exp(logpsi(pars, σ)) #/ weights#,keepdims=True 

@expect.dispatch
def expect_nochunking(vstate: WeightedMCState, operator: ElectronicOperator, chunk_size: None):
    return expect(vstate, operator)

@dispatch
def expect(vstate: WeightedMCState, Ô: ElectronicOperator) -> Stats:  # noqa: F811
    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô)

    return _expect(
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )

@partial(jax.jit, static_argnums=(0, 1))
def _expect(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Stats:
    σ_shape = σ.shape

    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    _, Ō_stats = expect_call(
        log_pdf,
        partial(local_value_kernel, logpsi),
        parameters,
        σ,
        local_value_args,
        n_chains=None,
    )
    #print(Ō_stats)
    return Ō_stats

def expect_call(
    log_pdf: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars: PyTree,
    σ: jnp.ndarray,
    *expected_fun_args,
    n_chains: int = None,
) -> Tuple[jnp.ndarray, Stats]:
    """
    Computes the expectation value over a log-pdf.
    Args:
        log_pdf:
        expected_ffun
    """
    return _expect_kernel(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args)

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _expect_kernel(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    weights,_ = expected_fun_args[0]
    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))
    L_σ = jnp.sum(jnp.multiply(L_σ , weights),keepdims=True)
    L̄_σ = mpi_statistics(L_σ)
    #L̄_σ = L_σ.mean(axis=0)

    return L̄_σ.mean, L̄_σ

def _expect_fwd(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    weights = expected_fun_args[0] 
    if n_chains is not None:
        L_σ_r = L_σ.reshape((n_chains, -1))
    else:
        L_σ_r = L_σ
    L_σ_r = jnp.sum(jnp.multiply(L_σ_r , weights),keepdims=True)
    L̄_stat = mpi_statistics(L_σ_r.T)

    L̄_σ = L̄_stat.mean
    # L̄_σ = L_σ.mean(axis=0)

    # Use the baseline trick to reduce the variance
    ΔL_σ = L_σ - L̄_σ   #Check this!!!

    return (L̄_σ, L̄_stat), (pars, σ, expected_fun_args, ΔL_σ)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the chunk dimension and without a loop through the chunk dimension
def _expect_bwd(n_chains, log_pdf, expected_fun, residuals, dout):
    pars, σ, cost_args, ΔL_σ = residuals
    dL̄, dL̄_stats = dout

    def f(pars, σ, *cost_args):
        log_p = log_pdf(pars, σ)
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        term2 = expected_fun(pars, σ, *cost_args)
        out = mpi_mean(term1 + term2, axis=0)
        out = out.sum()
        return out

    _, pb = nkvjp(f, pars, σ, *cost_args)
    grad_f = pb(dL̄)
    return grad_f

_expect_kernel.defvjp(_expect_fwd, _expect_bwd)

@expect_and_grad.dispatch
def expect_and_grad_nochunking(  # noqa: F811
    vstate: WeightedMCState,
    operator: ElectronicOperator,
    use_covariance: Bool,
    chunk_size: None,
    *args,
    **kwargs,
):
    return expect_and_grad(vstate, operator, use_covariance, *args, **kwargs)


@dispatch
def expect_and_grad(  # noqa: F811
    vstate: WeightedMCState,
    Ô: ElectronicOperator,
    use_covariance: TrueT,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô)

    Ō, Ō_grad, new_model_state = grad_expect_hermitian(
        local_estimator_fun,
        vstate._apply_fun,
        mutable,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad

@partial(jax.jit, static_argnums=(0, 1, 2))
def grad_expect_hermitian(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Tuple[PyTree, PyTree]:

    σ_shape = σ.shape
    weights,_ = local_value_args
    log_psi = model_apply_fun({"params": parameters, **model_state}, σ)
    O_loc = local_value_kernel(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
    )
    Ō = jnp.sum(jnp.multiply(O_loc,weights),keepdims=True)
    Ō = mpi_statistics(Ō)
    #Ō.variance = jnp.sum(weights * (O_loc**2) - (Ō.mean)**2 )
    O_loc -= (Ō.mean)  
    O_loc =  jnp.conjugate((jnp.multiply(weights , O_loc)))
    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(O_loc)[0] #multiply with weights
    Ō_grad = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )
    new_model_state = new_model_state[0] if is_mutable else None
        
    return Ō, jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), new_model_state

