from collections.abc import Callable
import numpy as np
import jax
import jax.numpy as jnp
from qiskit.quantum_info import Pauli, SparsePauliOp


def apply_pauli(pauli: Pauli, state: jax.Array) -> jax.Array:
    nq = pauli.num_qubits
    out = state
    # Apply X and Z in contiguous blocks
    for ptype, arr in [('x', pauli.x), ('z', pauli.z)]:
        pos = np.nonzero(arr)[0]
        if pos.shape[0]:
            npos = np.nonzero(~arr)[0]
            start = pos[0]
            while True:
                iend = np.searchsorted(npos, start)
                if iend == npos.shape[0]:
                    end = nq
                else:
                    end = npos[iend]
                nsites = end - start
                out = jnp.reshape(out, (2 ** (nq - nsites - start), 2 ** nsites, 2 ** start))
                if ptype == 'x':
                    out = jnp.flip(out, axis=1)
                else:
                    binary = (np.arange(2 ** nsites)[:, None] >> np.arange(nsites)[None, ::-1]) % 2
                    signs = jnp.array([1, -1])[np.sum(binary, axis=1) % 2]
                    out *= signs[None, :, None]
                istart = np.searchsorted(pos, end)
                if istart == pos.shape[0]:
                    break
                else:
                    start = pos[istart]

    out = jnp.reshape(out, (-1,))
    if (iphase := np.count_nonzero(pauli.x & pauli.z) % 4) > 0:
        out *= (1., -1.j, -1., 1.j)[iphase]
    return out


def make_apply_h(hamiltonian: SparsePauliOp) -> Callable[[jax.Array], jax.Array]:
    @jax.jit
    def apply_h(state, result=None):
        if result is None:
            result = jnp.zeros_like(state)
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            result += coeff * apply_pauli(pauli, state)
        return result

    return apply_h
