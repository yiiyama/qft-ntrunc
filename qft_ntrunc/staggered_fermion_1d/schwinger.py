"""Functions to construct the Schwinger model Hamiltonian."""
from collections.abc import Callable
from typing import Any
import numpy as np
import jax
from qiskit.quantum_info import SparsePauliOp
from qft_ntrunc.staggered_fermion_1d.fermion import (
    dagger,
    staggered_hopping_term_spo,
    staggered_hopping_term_sparse,
    staggered_mass_term_spo,
    staggered_mass_term_sparse
)
from qft_ntrunc.paulis import make_apply_h_args
from qft_ntrunc.utils import identity, simplify


def schwinger_electric_term_spo(
    num_sites: int,
    lin: int = 0,
    bc: str = 'periodic'
) -> SparsePauliOp:
    r"""Electric term of the Schwinger model Hamiltonian.

    .. math::

        H_{\mathrm{elec}} = \frac{1}{2} \sum_{n} L_n^2

    The sum runs from 0 to :math:`N-2` for open boundary, and to :math:`N-1` for periodic. The term
    :math:`L_n` represents the electric field to the left of site :math:`n` (sites are indexed from
    right to left). Since we work in Gauss's law-resolved space,

    .. math::

        L_n = L_{\mathrm{in}} + \sum_{m=0}^{n} Q_{m}.

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

    return 0.5 * term

def schwinger_electric_term_sparse(
    phi: list[Any],
    lin: int = 0,
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
        field = simplify(field)
        term += field @ field
        term = simplify(term)

    return 0.5 * term


def schwinger_hamiltonian_spo(
    num_sites: int,
    lsp: float,
    mass: float,
    coupling_g: float,
    lin: int = 0,
    bc: str = 'periodic'
) -> SparsePauliOp:
    hamiltonian = 1. / lsp * staggered_hopping_term_spo(num_sites, bc=bc)
    if mass != 0.:
        hamiltonian += mass * staggered_mass_term_spo(num_sites)
    if coupling_g != 0.:
        hamiltonian += np.square(coupling_g) * lsp * schwinger_electric_term_spo(num_sites, lin=lin,
                                                                                 bc=bc)
    return hamiltonian.simplify()


def schwinger_hamiltonian_sparse(
    phi: list[Any],
    lsp: float,
    mass: float,
    coupling_g: float,
    lin: int = 0,
    bc: str = 'periodic'
) -> Any:
    hamiltonian = 1. / lsp * staggered_hopping_term_sparse(phi, bc=bc)
    if mass != 0.:
        hamiltonian += mass * staggered_mass_term_sparse(phi)
    if coupling_g != 0.:
        hamiltonian += np.square(coupling_g) * lsp * schwinger_electric_term_sparse(phi, lin=lin, bc=bc)
    return simplify(hamiltonian)


def make_param_apply_h_args(
    phi: list[Any],
    lsp: float,
    mass: float,
    lin: int = 0,
    bc: str = 'periodic'
) -> Callable[[jax.Array, jax.Array, float], jax.Array]:
    """Return apply_h arguments with a separation between free and electric terms (entry 0 is free).
    """
    args_elec = make_apply_h_args(schwinger_electric_term_sparse(phi, lin=lin, bc=bc))
    args_free = make_apply_h_args(schwinger_hamiltonian_sparse(phi, lsp, mass, 0., lin=lin, bc=bc),
                                  width=args_elec[1].shape[1])
    return tuple(np.concatenate([f, e], axis=0) for f, e in zip(args_free, args_elec))
