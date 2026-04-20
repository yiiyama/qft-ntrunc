from collections.abc import Callable
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from qiskit.quantum_info import Pauli, SparsePauliOp


@partial(jax.jit, static_argnames=['num_qubits'])
def make_signs(num_qubits: int) -> jax.Array:
    filt = jnp.array(1, dtype=np.uint8)
    signs = (1 - 2 * (jnp.bitwise_count(jnp.arange(2 ** num_qubits)) & filt)).astype(np.complex128)
    return signs[None, :, None]


def apply_pauli(
    pauli: Pauli,
    state: jax.Array,
    coeff: complex,
    preset_signs: dict[int, jax.Array]
) -> jax.Array:
    nq = pauli.num_qubits
    out = state

    # Apply X and Z in contiguous blocks
    for ptype, arr in [('x', pauli.x), ('z', pauli.z)]:
        pos = np.nonzero(arr)[0]
        if pos.shape[0] == 0:
            continue
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
                if (signs := preset_signs.get(nsites)) is None:
                    signs = make_signs(nsites)
                    preset_signs[nsites] = signs
                out *= signs
            istart = np.searchsorted(pos, end)
            if istart == pos.shape[0]:
                break
            else:
                start = pos[istart]

    out = jnp.reshape(out, (-1,))
    out *= coeff * (1., -1.j, -1., 1.j)[np.count_nonzero(pauli.x & pauli.z) % 4]
    return out


def make_apply_h(hamiltonian: SparsePauliOp) -> Callable[[jax.Array], jax.Array]:
    @jax.jit
    def apply_h(state, result=None):
        preset_signs = {
            1: jnp.array([[[1.], [-1.]]], dtype=np.complex128),
            2: jnp.array([[[1.], [-1.], [-1.], [1.]]], dtype=np.complex128),
            3: jnp.array([[[1.], [-1.], [-1.], [1.],
                          [-1.], [1.], [1.], [-1.]]], dtype=np.complex128)
        }
        if result is None:
            result = jnp.zeros_like(state)
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            result += apply_pauli(pauli, state, coeff, preset_signs)
        return result

    return apply_h
