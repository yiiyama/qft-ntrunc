from collections.abc import Callable
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from qiskit.quantum_info import SparsePauliOp


def make_apply_h(
    hamiltonian: SparsePauliOp,
    by_parts: bool = False
) -> Callable[[jax.Array], jax.Array]:
    """Make the apply_h function.

    Pauli products are grouped by the X signature first.
    out = X @ state
    out = Z @ out * coeff * phase
    """
    xuniq, indices, counts = np.unique(hamiltonian.paulis.x, axis=0, return_inverse=True, return_counts=True)

    powers = 1 << np.arange(hamiltonian.num_qubits, dtype=np.int32)[None, :]
    # X signatures are interpreted as a binaries and converted to integer masks
    xmasks = np.sum(xuniq.astype(np.uint8) * powers, axis=1, dtype=np.int32)
    # Do the same for the Z signatures. Also precompute the coeff * phase factors
    shape = (xuniq.shape[0], np.max(counts))
    zmasks = np.zeros(shape, dtype=np.int32)
    coeffs = np.zeros(shape, dtype=np.complex128)
    for ipat, xmask in enumerate(xmasks):
        ipaulis = np.nonzero(indices == ipat)[0]
        zpatterns = hamiltonian.paulis.z[ipaulis]
        zm = np.sum(zpatterns.astype(np.int32) * powers, axis=1)
        zmasks[ipat, :counts[ipat]] = zm
        # Will multiply the entire op by (-i)^{n_zx}
        phases = np.array([1., -1.j, -1., 1.j])[np.bitwise_count(xmask & zm) % 4]
        coeffs[ipat, :counts[ipat]] = hamiltonian.coeffs[ipaulis] * phases

    def apply_pauli(xm, zms, cs, cnt, iota, state):
        idx = jnp.bitwise_xor(iota, xm)
        xstate = state[idx]

        def add_diag(val):
            ip, out = val
            signs = 1. - 2. * (jnp.bitwise_count(iota & zms[ip]) & 1)
            out += xstate * signs * cs[ip]
            return (ip + 1, out)

        return jax.lax.while_loop(
            lambda val: jnp.less(val[0], cnt),
            add_diag,
            (0, jnp.zeros_like(xstate))
        )[1]

    @jax.jit
    def apply_h(xmasks, zmasks, coeffs, counts, state):
        iota = jnp.arange(state.size, dtype=np.int32)
        return jax.lax.fori_loop(
            0, xmasks.shape[0],
            lambda i, r: r + apply_pauli(xmasks[i], zmasks[i], coeffs[i], counts[i], iota, state),
            jnp.zeros_like(state)
        )

    if by_parts:
        return apply_h, xmasks, zmasks, coeffs, counts
    return partial(apply_h, jnp.array(xmasks), jnp.array(zmasks), jnp.array(coeffs), jnp.array(counts))
