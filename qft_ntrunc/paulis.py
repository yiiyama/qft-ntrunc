from functools import partial
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from qiskit.quantum_info import SparsePauliOp


def make_apply_h_args(
    hamiltonian: SparsePauliOp,
    width: Optional[int] = None,
    mesh: Optional[Mesh] = None
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    xuniq, indices, counts = np.unique(hamiltonian.paulis.x, axis=0, return_inverse=True,
                                       return_counts=True)
    if width is None:
        width = np.max(counts)
    assert width >= np.max(counts)

    powers = 1 << np.arange(hamiltonian.num_qubits, dtype=np.int32)[None, :]
    # X signatures are interpreted as a binaries and converted to integer masks
    xmasks = np.sum(xuniq.astype(np.uint8) * powers, axis=1, dtype=np.int32)
    # Do the same for the Z signatures. Also precompute the coeff * phase factors
    shape = (xuniq.shape[0], width)
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

    if mesh:
        pad_len = mesh.shape['x'] - xmasks.shape[0] % mesh.shape['x']
        if pad_len:
            xmasks = np.pad(xmasks, [(0, pad_len)])
            zmasks = np.pad(zmasks, [(0, pad_len), (0, 0)])
            coeffs = np.pad(coeffs, [(0, pad_len), (0, 0)])
            counts = np.pad(counts, [(0, pad_len)])
        sharding = NamedSharding(mesh, P('x'))
        xmasks = jax.device_put(xmasks, sharding)
        zmasks = jax.device_put(zmasks, sharding)
        coeffs = jax.device_put(coeffs, sharding)
        counts = jax.device_put(counts, sharding)

    return xmasks, zmasks, coeffs, counts


@partial(jax.jit, static_argnames=['mult'])
def apply_h(state, xmasks, zmasks, coeffs, counts, mult=1):
    iota = jnp.arange(state.size, dtype=np.int32)

    def apply_pauli(xm, zms, cs, cnt):
        idx = jnp.bitwise_xor(iota, xm)
        xstate = state[idx]

        if mult == 1:
            def add_diag(val):
                ip, out = val
                signs = 1. - 2. * (jnp.bitwise_count(iota & zms[ip]) & 1)
                out += xstate * signs * cs[ip]
                return (ip + 1, out)
        else:
            def add_diag(val):
                ip, out = val
                idx = ip + jnp.arange(mult)
                zmslice = zms.at[idx].get(mode='fill', fill_value=0)
                signs = 1. - 2. * (jnp.bitwise_count(iota[None, :] & zmslice[:, None]) & 1)
                cslice = cs.at[idx].get(mode='fill', fill_value=0.j)
                out += jnp.sum(xstate[None, :] * signs * cslice[:, None], axis=0)
                return (ip + mult, out)

        return jax.lax.while_loop(
            lambda val: jnp.less(val[0], cnt),
            add_diag,
            (0, jnp.zeros_like(xstate))
        )[1]

    result = jnp.zeros_like(state)
    if (sharded := jax.typeof(xmasks).sharding.num_devices != 0):
        result = jax.lax.pcast(result, 'x', to='varying')

    result = jax.lax.fori_loop(
        0, xmasks.shape[0],
        lambda i, r: r + apply_pauli(xmasks[i], zmasks[i], coeffs[i], counts[i]),
        result
    )

    if sharded:
        return jax.lax.psum(result, 'x')
    return result


@partial(jax.jit, static_argnames=['mult', 'basis'])
def apply_h_truncated(state, xmasks, zmasks, coeffs, counts, basis, nmax, mult=1):
    if basis == 'fock_ab':
        npart = jnp.bitwise_count(jnp.arange(state.shape[0]))
        projector = jnp.less_equal(npart, nmax)
    else:
        raise NotImplementedError('for later')

    state *= projector
    out = apply_h(state, xmasks, zmasks, coeffs, counts, mult=mult)
    return out * projector
