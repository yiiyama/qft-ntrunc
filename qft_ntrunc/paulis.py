from collections.abc import Callable
import numpy as np
import jax
import jax.numpy as jnp
from qiskit.quantum_info import Pauli, SparsePauliOp


def apply_pauli(pauli: Pauli, state: jax.Array) -> jax.Array:
    nq = pauli.num_qubits
    pauli_z = jnp.array([[[1.], [-1.]]], dtype=np.complex128)

    out = state
    for isite, (x, z) in enumerate(zip(pauli.x, pauli.z)):
        if not (x or z):
            continue
        out = jnp.reshape(out, (2 ** (nq - isite - 1), 2, 2 ** isite))
        if x:
            out = jnp.flip(out, 1)
        if z:
            out *= pauli_z

    out = jnp.reshape(out, (-1,))
    if (iphase := np.count_nonzero(pauli.x & pauli.z) % 4) > 0:
        out *= (1., -1.j, -1., 1.j)[iphase]
    return out


def make_apply_h(hamiltonian: SparsePauliOp) -> Callable[[jax.Array], jax.Array]:
    @jax.jit
    def apply_h(state):
        result = jnp.zeros_like(state)

        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            if not np.any(pauli.x | pauli.z):
                continue
            result += coeff * apply_pauli(pauli, state)

        return result

    return apply_h
