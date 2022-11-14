from typing import List, Union, Optional
import re
from collections import defaultdict
from functools import partial
import numpy as np
from numba import jit
import numbers
import copy
import numba as nb
from scipy import sparse
from netket.utils.types import DType, Array
from netket.operator._abstract_operator import AbstractOperator
from netket.operator._pauli_strings import _count_of_locations
from netket.hilbert.abstract_hilbert import AbstractHilbert
from netket.utils.numbers import is_scalar
import time
from netket.experimental.hilbert import SpinOrbitalFermions
from scipy.sparse.linalg import eigsh

class ElectronicOperator(AbstractOperator):

    r"""
    A fermionic operator in :math:`2^{nd}` quantization.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        one_ints,
        two_ints,
        constant: Union[float, complex] = 0.0,
        dtype: DType = None,
        epsilon : float = 0.0,
        ref: float = 0.0,
        order =  False,
        diag = False
    ):
        super().__init__(hilbert)
        self._dtype = dtype

        self.onee_matrix = one_ints
        self.twoe_matrix = two_ints
        self._constant = constant
        self._initialized = False
        self._is_hermitian = True  # set when requested
        self.epsilon = epsilon
        self.ref = ref
        self.time = 0
        self.indices = np.arange(hilbert.size // 2, dtype = int)[::-1]
        if order:
            if order == "random":
                np.random.shuffle(self.indices)
            elif order == "reverse":
                self.indices = self.indices[::-1]
            else:
                self.indices = order
        #idx = np.reshape(self.indices,(2,hilbert.size // 4))
        #idx[1] = idx[1][::-1]
        #self.indices = np.ravel(idx.T)
        print("Orbital Indices: ",self.indices)
        self.diag = diag
        if diag:
            self.eig_vals = np.empty(0)

    def _reset_caches(self):
        """
        Cleans the internal caches built on the operator.
        """
        self._initialized = False
        self._is_hermitian = None

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:

            # following lists will be used to compute matrix elements
            # they are filled in _add_term
            self._initialized = True
            #print(self.hb_singles)
            self.h_init = False
    

    @staticmethod
    @jit(nopython=True,parallel = True)
    def _flattened_kernel(
        x,
        probs,
        amplis,
        constant,
        singles,  #matrix with all one electron matrix elements, is needed for the efficient calculation of single excitations --> sparse matrix?
        doubles,  #matrix with all two electron matrix elements, only needed for the calculation of the single excitations! --> sparse matrix?
        h,
        old_bin,
        idx_dict,
        idx_order
 ):
        #Method for determinning all connected determinatns efficiently following Heat-Bath CI --> scaling N^2, requires integral matrices at the moment
        #Consider truncating the singles and doubles matrices or the use of sparse representations

        def idx(a,size):
            idx1 = a % size
            #idx1 = size - 1 - idx1
            return idx_order[int(idx1)]
        
        def non_excited(x):
            x1 = x.copy()
            indices = np.arange(x1.shape[0])
            occ = indices[np.nonzero(x1)]
            val = constant
            n_o = x1.shape[0] / 2
            for i in occ:
                idx1 = idx(i,n_o)
                val += singles[idx1,idx1]
                spin1 = 1 * (i < n_o)
                for j in occ:
                    if j < i:
                        pass
                    else:
                        idx2 = idx(j,n_o)
                        spin2 = 1 *(j < n_o)
                        val += doubles[idx1,idx2,idx2,idx1] -  doubles[idx1,idx2,idx1,idx2] * (spin1 == spin2)
            return val

        def get_double_sign(xb,i,j,b,a): #get the proper sign of double excitation matrix element
            x2 = xb.copy()
            n_o = x2.shape[0] / 2
            sign = np.sum(x2[:i])
            x2[i] = 0
            sign = sign + np.sum(x2[:j])
            x2[j] = 0
            sign = sign + np.sum(x2[:b])
            x2[b] = 1
            sign = sign + np.sum(x2[:a])
            x2[a] = 1
            idx1 = idx(i , n_o)
            idx2 = idx(j , n_o)
            idx3 = idx(a , n_o)
            idx4 = idx(b , n_o)
            spin1 = 1 * (i < n_o)
            spin2 = 1 * (j < n_o)
            spin3 = 1 * (a < n_o)
            spin4 = 1 * (b < n_o)
            mel = doubles[idx1,idx2,idx4,idx3] * (spin1 == spin3) * (spin2 == spin4) - doubles[idx1,idx2,idx3,idx4] * (spin1 == spin4) * (spin2 == spin3)
            return mel * (-1)**sign 

        def single_excitation(xa,i,a): #Has to be adapted --> get the correct double sign, 
            x3 = xa.copy()
            indices = np.arange(x3.shape[0])
            occ = indices[np.nonzero(x3)]
            val = 0
            n_o = x3.shape[0] / 2
            idx1 = idx(i , n_o)
            idx2 = idx(a , n_o)
            spin1 = 1 * (i < n_o) 
            spin2 = 1 * (a < n_o)
            for j in occ:
                 if j != i:
                     d_mel =  get_double_sign(x3,i,j,a,j)
                     val += d_mel 
            sign = np.sum(x3[:i])
            x3[i] = 0
            sign = sign + np.sum(x3[:a])
            val += singles[idx1,idx2] * ( (-1) ** sign) * (spin1 == spin2)
            x3[a] = 1
            return val


        def states_to_numbers(states):
        #converts state with binary local states to number which can be used as hash values
            numbers = np.sum(states*(2**(np.arange(states.shape[1]))),axis=1)
            numbers = numbers.flatten()
            return numbers
 
        def state_to_number(states):
            number = np.sum(states*(2**(np.arange(states.shape[0]))))
            return int(number)
        
        def get_new_states(states,bin_states,state_dict,new_dict,h): #Get new and old states as well as h_indices
            batchsize = len(bin_states)
            h_idx = np.zeros(batchsize,dtype=nb.types.int64) 
            for i,bin_idx in enumerate(bin_states):
                dummy = -batchsize +(i-1)
                val = state_dict.pop(bin_idx,dummy)
                h_idx[i] = val
                new_dict[bin_idx] = val
            order = np.argsort(h_idx)
            h_idx = h_idx[order]
            reorder = np.argsort(order)
            states = states[order]
            bin_states = bin_states[order]
            items = state_dict.items()
            i = 0
            for item in items:
                key, val = item
                h[:,val] = 0.0
                h[val,:] = 0.0
                h_idx[i] = val
                new_key = bin_states[i]
                new_dict[new_key] = val
                i = i+1 
               
            
            new_states = states[:i,:]
            new_idx = h_idx[:i]
            return new_dict, new_states,new_idx, order, reorder, h_idx, h      
                 
    
        batchsize,orbitals = x.shape
        indices = np.arange(orbitals)
        new_bin = states_to_numbers(x)
        new_dict = nb.typed.Dict.empty(
                                        key_type=nb.types.int64,
                                        value_type=nb.types.int64)
        new_dict,new_states,new_idx,order,reorder,h_idx,h = get_new_states(x,new_bin,idx_dict,new_dict,h) 
        h_idx = h_idx[reorder]
        h_order = np.argsort(h_idx)
        amplis = amplis[h_order]
        reorder = np.argsort(h_order)
        n_new = len(new_idx)
        for iters in nb.prange(n_new): #loop only over new states, to be implemented
            state1_index = new_idx[iters]
            state1 = new_states[iters]
            h[state1_index,state1_index] = non_excited(state1)
            diffs = np.abs((x - state1))
            exc = np.sum(diffs,axis=1)
            order = np.argsort(exc)
            x_index = np.arange(batchsize)
            x_index = x_index[order]
            for index in x_index: #Loop only over states which satisfy this!
                state2 = x[index]
                exc_lvl = exc[index]
                state2_index = h_idx[index]
                if exc_lvl ==  2:
                    occ1 = indices[state1 > state2]
                    ucc1 = indices[state2 > state1]
                    mel = single_excitation(state1,occ1[0],ucc1[0]) 
                    h[state1_index,state2_index] = mel
                    h[state2_index,state1_index] = mel
                elif exc_lvl == 4:
                    occ1,occ2 = indices[state1 > state2]
                    ucc1,ucc2 = indices[state2 > state1]
                    mel = get_double_sign(state1,occ1,occ2,ucc1,ucc2) 
                    h[state1_index,state2_index],h[state2_index,state1_index] = mel, mel
                elif exc_lvl == 0:
                    pass
                else: 
                    break 
        mels = np.dot(h,amplis)
        mels = mels / amplis
        mels = mels[reorder]    
        return probs,mels, h, new_bin, new_dict

    
    def get_weighted_mels(self , x , p, amplis):
        self._setup()
        start = time.time()
        if not self.h_init:
            self.h_init = True
            self.h = np.zeros((x.shape[0],x.shape[0]),dtype=np.complex64)
            self.idx_dict = nb.typed.Dict.empty(
                                                      key_type=nb.types.int64,
                                                      value_type=nb.types.int64)
            self.state_cache = (-1) * np.arange(x.shape[0])
            for i,el in enumerate(self.state_cache):
                self.idx_dict[el] = i
        x = np.array(x,dtype=np.int8)
        p = np.array(p)
        amplis = np.array(amplis,dtype=np.complex64) 
        weights,mels, self.h, self.state_cache, self.idx_dict = self._flattened_kernel(
        x,
        p,
        amplis,
        self._constant,
        self.onee_matrix,  #matrix with all one electron matrix elements, is needed for the efficient calculation of single excitations --> sparse matrix?
        self.twoe_matrix,  #matrix with all two electron matrix elements, only needed for the calculation of the single excitations! --> sparse matrix?
        self.h,
        self.state_cache,
        self.idx_dict,
        self.indices
)       

        if self.diag:
            val = eigsh(self.h,k=1)
            self.eig_vals = np.append(self.eig_vals,val[0]) 
        end = time.time()
        self.time += end - start
        #print(f"Energy calculation took {end - start} seconds")
        return mels , weights
   

    def gs(self,state1,state2):
        self._setup()
       
        e = self._matrix_element(np.array(state1),np.array(state2),self._orb_idxs,
            self._daggers,
            self._weights,
            self._term_ends)
        return e + self._constant  
                        
    @staticmethod
    @jit(nopython=True)
    def _matrix_element(
        xa,xb, # xa,xb: states
        orb_idxs,
        daggers,
        weights,
        term_ends,
        abs_val = False,
        n_ops = False
    ):
        '''x_prime[n_c,a]
        Returns a single matrix element  <xa|Ô|xb>
        '''
        def is_empty(site):
            return _isclose(site, 0)

        def flip(site):
            return 1 - site

        new_term=True
      #  x_prime = np.zeros(x.shape[0])
      #  x_prime[a]
        matrix_elements = 0
        # loop over all terms,return <ij|ab> - <ij|ba>
        for orb_idx, dagger, weight, term_end in zip(
                orb_idxs, daggers, weights, term_ends
            ):
            if new_term:
                mel = weight
                new_term = False
                has_xp = True
                xt = np.copy(xa)
                op = 0
            op += 1    
            empty_site = is_empty(xt[orb_idx])
            if dagger:
                if not empty_site:
                    has_xp = False
                else:
                    mel *= (-1) ** np.sum(xt[:orb_idx])  # jordan wigner sign
                    xt[orb_idx] = flip(xt[orb_idx])
            else:
                if empty_site:
                    has_xp = False
                else:
                    mel *= (-1) ** np.sum(xt[:orb_idx])  # jordan wigner sign
                    xt[orb_idx] = flip(xt[orb_idx])

                # if this is the end of the term, we collect things
           
            
            if term_end:
                if n_ops and n_ops != op:
                    mel = 0
                if has_xp and (xt == xb).all():
                    if abs_val:
                        matrix_elements += abs(mel)
                    else: 
                        matrix_elements += mel

                new_term = True

        return matrix_elements

      

    @staticmethod
    def from_openfermion(
        hilbert: AbstractHilbert,
        of_file: str = "",  # noqa: F821
        *,
        n_orbitals: Optional[int] = None,
        frozen: int = 0,
        diag: bool = False
    ):
        r"""
        Converts an openfermion FermionOperator into a netket FermionOperator2nd.

        The hilbert first argument can be dropped, see __init__ for details and default
        value.
        Warning: convention of openfermion.hamiltonians is different from ours: instead
        of strong spin components as subsequent hilbert state outputs (i.e. the 1/2 spin
        components of spin-orbit i are stored in locations (2*i, 2*i+1)), we concatenate
        blocks of definite spin (i.e. locations (i, n_orbitals+i)).

        Args:
            hilbert: (optional) hilbert of the resulting FermionOperator2nd object
            of_fermion_operator: openfermion.ops.FermionOperator object
            n_orbitals: (optional) total number of orbitals in the system, default
                None means inferring it from the FermionOperator2nd. Argument is
                ignored when hilbert is given.
            convert_spin_blocks: whether or not we need to convert the FermionOperator
                to our convention. Only works if hilbert is provided and if it has
                spin != 0

        Returns:
            A FermionOperator2nd object.
        """
        from openfermion.chem import MolecularData

        if hilbert is None:
            raise ValueError(
                "The first argument `from_openfermion` must either be an"
                "openfermion operator or an Hilbert space, followed by"
                "an openfermion operator"
            )
        print(f"Loading molecule integrals from {of_file}")
        molecule = MolecularData(filename=of_file)
        molecule.load()
        print(f"HF energy {molecule.hf_energy:.10f}")
        hf = molecule.hf_energy
        cisd = molecule.cisd_energy
        ccsd = molecule.ccsd_energy
        fci = molecule.fci_energy
        if cisd:
            print(f"CISD energy {molecule.cisd_energy:.10f}")
        if ccsd:
            print(f"CCSD energy {molecule.ccsd_energy:.10f}")
        if fci:
            print(f"FCI energy {molecule.fci_energy:.10f}")
        
        e = molecule.orbital_energies
        epsilon = np.zeros(e.shape[0] * 2)
        
        for i,el in enumerate(e):
            index1 = e.shape[0] - i - 1
            index2 = index1 + e.shape[0]
            epsilon[index1],epsilon[index2] = el,el
        spatial = np.arange(molecule.n_orbitals,dtype=int)
        occ = spatial[:frozen]
        active = spatial[frozen:]
        core,one_ints,two_ints = molecule.get_active_space_integrals(occupied_indices=occ,active_indices=active)
        constant =  molecule.nuclear_repulsion + core
        operator = ElectronicOperator(hilbert, one_ints, two_ints, constant=constant, epsilon = epsilon,ref =  hf, diag = diag)
        return operator


    def __repr__(self):
        return (
            f"FermionOperator2nd(hilbert={self.hilbert}, "
            f"n_operators={len(self._operators)}, dtype={self.dtype})"
        )

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    @staticmethod
    @jit(nopython=True) 
    def GenerateSinglesDoubles(ref_state):

        orbitals = ref_state.shape[0]
        spatial = int(orbitals/2)
        
        #size = alpha * (orbitals -alpha) + beta * (orbitals - beta) + alpha * (alpha - 1) * (orbitals - alpha) * (orbitals - alpha -1) + beta * (beta - 1) * (orbitals - beta) * (orbitals - beta -1) + alpha * beta * (orbitals - alpha) * (orbitals - beta)
        states = np.reshape(ref_state.copy(),(1,orbitals))
        idx = np.arange(orbitals)
        occ = idx[ref_state != 0]
        unocc = idx[ref_state == 0]
        n_occ = occ.shape[0]
        n_unocc = unocc.shape[0]
        for idx1,i in enumerate(occ):
            for idx2,a in enumerate(unocc):
                spin1,spin2 = 1*(i < spatial), 1*(a < spatial)
                if spin1 == spin2:
                    state = ref_state.copy()
                    state[i],state[a] = 0,1
                    state = np.reshape(state,(1,orbitals))
                    states = np.row_stack((states,state))
                for idx3 in range(idx1+1,n_occ):
                    for idx4 in range(idx2+1,n_unocc):
                        j = occ[idx3]
                        b = unocc[idx4]
                        spin3,spin4 = 1 * (j < spatial), 1 * (b < spatial)
                        if (spin1+spin3) == (spin2+spin4):
                            state = ref_state.copy()
                            state[i],state[j],state[a],state[b] = 0,0,1,1
                            state = np.reshape(state,(1,orbitals))
                            states = np.row_stack((states,state))
        return states     

    def CISD(self):   
        alpha,beta = self.hilbert.n_fermions
        size = self.hilbert.size
        ref_state = np.zeros(size,dtype=np.int8)
        for i in range(alpha):
            index = int(size/2 - i - 1)
            ref_state[index] = 1
        for i in range(beta):
            index = int(size - i - 1) 
            ref_state[index] = 1
        states = self.GenerateSinglesDoubles(ref_state) 
        p = np.ones(states.shape[0])
        ampli = np.ones(states.shape[0])
        self.h_init = False
        _,_ = self.get_weighted_mels(states , p, ampli)
        #indices = np.zeros(states.shape[0],dtype=np.int64)
        #for i,key in enumerate(self.state_cache):
        #    indices[i] = self.idx_dict[key]
        #states = states[np.argsort(indices)]
        val,vecs = np.linalg.eigh(self.h)
        return val, vecs[:,0], states   
        

    def MP2(self): #Implement: MP2 Amplitudes, States, energies, maybe include even singles for arbitrary basis sets
        
        def state_to_number(states):
            number = np.sum(states*(2**(np.arange(states.shape[0]))))
            return int(number)

        alpha,beta = self.hilbert.n_fermions
        size = self.hilbert.size
        ref_state = np.zeros(size,dtype=np.int8)
        for i in range(alpha):
            index = int(size/2 - i - 1)
            ref_state[index] = 1
        for i in range(beta):
            index = int(size - i - 1)
            ref_state[index] = 1
        val,vecs,states = self.CISD()
        #e = np.diagonal(self.h) 
        e = np.dot(states,self.epsilon)
        
        ref_bin = state_to_number(ref_state) 
        h_idx = self.idx_dict[ref_bin] 
        mels = self.h[h_idx,:]
        hf = mels[h_idx] 
        mask = np.ones(mels.shape[0])
        mask[h_idx] = 0
        ref = e[0] #mels[h_idx]
        mels = mels[mask != 0]
        e = ref - e[1:]  
        amplis = mels / e 
        states = states[mask != 0]
        mp2_energy = np.sum(amplis * mels) + hf 
        states = np.vstack((states,np.reshape(ref_state,(-1,size))))     
        amplis = np.append(amplis,1)
        amplis = amplis / np.sqrt(np.sum(abs(amplis)**2))
        return mp2_energy, amplis, states

    def FCI(self):
        n_states = np.arange(self.hilbert.n_states) 
        states = self.hilbert.numbers_to_states(n_states)
        p = np.ones(states.shape[0])
        ampli = np.ones(states.shape[0])
        self.h_init = False
        _,_ = self.get_weighted_mels(states , p, ampli)
        val,vecs = np.linalg.eigh(self.h)
        self.h_init = False
        return val[0], vecs[:,0], states


    def RotateBasis(self,rot_matrix):
        self.setup()
        self.onee_matrix = np.einsum("ij,jk,kl -> il",rot_matrix,self.onee_matrix,rot_matrix)
        self.twoe_matrix = np.einsum("im,jn,kq,lr,ijkl -> mnqr",rot_matrix,rot_matrix,rot_matrix,rot_matrix,self.twoe_matrix)
