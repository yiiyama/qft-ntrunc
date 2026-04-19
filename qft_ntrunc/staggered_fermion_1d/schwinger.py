"""Functions to construct the Schwinger model Hamiltonian."""
from functools import partial
from typing import Any
import numpy as np
from scipy.sparse import coo_array
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, bcoo_reduce_sum
from qiskit.quantum_info import SparsePauliOp
from .fermion import (
    get_rapidity,
    dagger,
    jw_annihilator_spo,
    ab_to_phi_sparse,
    staggered_hopping_term_spo,
    staggered_mass_term_spo,
    get_basis_indices,
    get_basis_change_matrix
)
from qft_ntrunc.utils import clean_array, identity, simplify


def get_h_free(num_sites, fock_indices, mu):
    """Return the free part of the Hamiltonian in Fock basis, projected onto the given indices."""
    rapidity = get_rapidity(num_sites, mu, npmod=jnp)
    binaries = (fock_indices[:, None] >> jnp.arange(num_sites)[None, :]) % 2
    energy = mu * jnp.cosh(rapidity)
    energy = jnp.tile(energy, 2)
    h_free = jnp.sum(energy[None, :] * binaries, axis=1)
    clean_array(h_free, inplace=True)
    return h_free


def get_h_elec(num_sites, position_indices, basis_change_matrix, l0):
    r"""Return the electric term of the Hamiltonian in Fock basis, projected onto the given position
    indices.

    Since we work in Gauss's law-solved space, :math:`L_n = \sum_{m=0}^{n} q_m + L_0`.
    """
    half_lat = num_sites // 2
    binaries = (position_indices[:, None] >> jnp.arange(num_sites)[None, :]) % 2
    charge = jnp.tile(np.array([0, 1]), half_lat)[None, :] - binaries
    electric_config = jnp.cumsum(charge, axis=1) + l0
    l2 = jnp.sum(jnp.square(electric_config), axis=1)
    h_elec = jnp.einsum('h,ih,jh->ij', l2, basis_change_matrix, basis_change_matrix.conjugate())
    clean_array(h_elec, inplace=True)
    return h_elec


def schwinger_electric_term_spo(
    num_sites: int,
    lin: int,
    bc: str = 'periodic'
) -> SparsePauliOp:
    """Electric term of the Schwinger model Hamiltonian.

    Args:
        num_sites: Number of staggered sites (multiple of 2).
        lin: Field value incident to site 0 (required if bc='periodic')
        bc: Boundary condition ('periodic' or else).

    Returns:
        Electric term as a Pauli sum.
    """
    term = 0
    field = SparsePauliOp('I' * num_sites, lin)
    if bc == 'periodic':
        # Assuming the operators act on the subspace with net charge 0
        term += field @ field
        smax = num_sites - 1
    else:
        smax = num_sites

    for isite in range(smax):
        paulis = ['I' * (num_sites - isite - 1) + 'Z' + 'I' * isite, 'I' * num_sites]
        charge = SparsePauliOp(paulis, [-0.5, (0.5, -0.5)[isite % 2]])
        field += charge
        term += field @ field
        term = term.simplify()

    return term

def schwinger_electric_term_sparse(
    phi: list[Any],
    lin: int,
    bc: str = 'periodic',
) -> Any:
    """Electric term of the Schwinger model Hamiltonian from site annihilation operators."""
    term = 0
    ident = identity(phi[0])
    field = lin * ident
    if bc == 'periodic':
        # Assuming the operators act on the subspace with net charge 0
        term += field @ field
        ops = phi[:-1]
    else:
        ops = phi

    for isite, op in enumerate(ops):
        charge = dagger(op) @ op
        if isite % 2:
            charge -= ident
        field += charge
        term += field @ field
        term = simplify(term)

    return term


def schwinger_hamiltonian_spo(num_sites, lsp, mass, coupling_j, bc='periodic'):
    hamiltonian = staggered_mass_term_spo(num_sites, mass)
    hamiltonian += staggered_hopping_term_spo(num_sites, lsp, bc=bc)
    if coupling_j != 0.:
        hamiltonian += schwinger_electric_term_spo(num_sites, coupling_j)
    return hamiltonian.simplify()


def setup(num_sites, mu, l0, sparse=True):
    print('Identifying Fock-space and position-space physical state indices')
    fock_indices, position_indices = get_basis_indices(num_sites)
    subdim = fock_indices.shape[0]

    print('Free Hamiltonian')
    h_free = get_h_free(num_sites, fock_indices, mu)

    print('Constructing position-space number operators')
    rapidity, wavenumber = get_rapidity(num_sites, mu, with_wn=True)
    fock_ab = [op.to_matrix(sparse=True) for op in jw_annihilator_spo(num_sites)]
    phi = ab_to_phi_sparse(fock_ab, rapidity, wavenumber)
    site_num_coords = []
    site_num_data = []
    for op in phi:
        op = op[:, fock_indices]
        arr = coo_array(dagger(op) @ op)
        site_num_coords.append(np.array(arr.coords).T)
        site_num_data.append(arr.data)
    site_num_op = BCOO((jnp.array(site_num_data), jnp.array(site_num_coords)),
                       shape=(num_sites, subdim, subdim))
    print('Computing basis change matrix')
    basis_change_matrix = get_basis_change_matrix(site_num_op)

    print('Computing the electric Hamiltonian')
    h_elec = get_h_elec(num_sites, position_indices, basis_change_matrix, l0)
    if sparse:
        h_elec = BCOO.fromdense(h_elec)

    return fock_indices, position_indices, site_num_op, basis_change_matrix, h_free, h_elec


@jax.jit
def _update_proj(isite, data):
    proj, bidx, site_num = data
    proj = jax.lax.cond(
        jnp.equal(bidx[isite], 1),
        lambda: site_num[isite] @ proj,
        lambda: proj - site_num[isite] @ proj
    )
    return (proj, bidx, site_num)


@jax.jit
def _position_as_fock(idx, site_num, proj_init):
    bidx = (idx >> jnp.arange(site_num.shape[0])) % 2
    proj = jax.lax.fori_loop(
        1, site_num.shape[0], _update_proj, (proj_init[bidx[0]], bidx, site_num)
    )[0]
    return jnp.linalg.eigh(proj)[1][:, -1]

    # absvals = jnp.sqrt(jnp.diagonal(proj).real)
    # ikey = jax.lax.while_loop(
    #     lambda i: jnp.isclose(absvals[i], 0.),
    #     lambda i: i + 1,
    #     0
    # )
    # return absvals * jnp.exp(1.j * jnp.angle(proj[:, ikey]))


@jax.jit
def _position_as_fock_i(icol, data):
    mat, pos_indices, site_num, proj_init = data
    mat = mat.at[:, icol].set(_position_as_fock(pos_indices[icol], site_num, proj_init))
    return (mat, pos_indices, site_num, proj_init)


# _position_as_fock_v = jax.jit(jax.vmap(_position_as_fock, in_axes=(0, None, None), out_axes=1))


@jax.jit
def position_states_as_fock_state_sums(pos_indices, site_num):
    pos_indices = jnp.asarray(pos_indices)
    subdim = pos_indices.shape[0]

    proj_init = jnp.array([
        jnp.eye(subdim, dtype=site_num.dtype) - site_num[0].todense(),
        site_num[0].todense()
    ])

    mat_init = jnp.empty((subdim, subdim), dtype=site_num.dtype)
    return jax.lax.fori_loop(0, subdim, _position_as_fock_i,
                             (mat_init, pos_indices, site_num, proj_init))[0]

    # batch_size = 32
    # num_batches = subdim // batch_size + int(subdim % batch_size != 0)
    # batched_subdim = batch_size * num_batches
    # pos_indices_batched = jnp.reshape(
    #     jnp.concatenate([
    #         pos_indices,
    #         jnp.zeros(batched_subdim - subdim, dtype=pos_indices.dtype)
    #     ]),
    #     (num_batches, batch_size)
    # )

    # @jax.jit
    # def _position_as_fock_vi(ibatch, mat):
    #     return mat.at[:, ibatch].set(
    #         _position_as_fock_v(pos_indices_batched[ibatch], site_num, proj_init)
    #     )

    # mat_init = jnp.empty((subdim, num_batches, batch_size), dtype=site_num.dtype)
    # mat = jax.lax.fori_loop(0, num_batches, _position_as_fock_vi, mat_init)
    # return mat.reshape((subdim, batched_subdim))[:, :subdim]
