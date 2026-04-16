from typing import Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit.quantum_info import SparsePauliOp
from jax.experimental.sparse import BCOO
from qft_ntrunc.utils import cleaned, dagger, simplify

SIGMA_Z = np.diagflat([1.+0.j, -1.+0.j])
SIGMA_PLUS = np.array([[0., 1.], [0., 0.]], dtype=np.complex128)


def get_rapidity(
    num_sites: int,
    mu: float,
    wavenumber: Optional[ArrayLike] = None,
    with_wn: bool = False,
    npmod=np
) -> NDArray | tuple[NDArray, NDArray]:
    r"""Return an array of rapidity values for each wave number.

    The number of modes represented in a lattice of :math:`N` sites is :math:`N/2` due to staggering
    of the fermion. Therefore, the wave number :math:`k \in \{-N/4, \dots, N/4-1\}`.

    We use a discretization convention where the momentum :math:`p_k` corresponding to wave number
    :math:`k` is

    .. math::

        p_k = \mu \sin \left( \frac{2 \pi}{N} k \right).

    Rapidity :math:`w_k` is defined by

    .. math::

        w_k = \sinh^{-1} \frac{p_k}{\mu}.
    
    Args:
        num_sites: Number of sites.
        mu: Mass parameter.
        wavenumber: The wave number to compute the rapidity for. If None, set to all wave numbers.
        with_wn: Whether to return the wave number array in addition.

    Returns:
        Array of rapidity values (and wave numbers if with_wn=True).
    """
    if wavenumber is None:
        half_lat = num_sites // 2
        wavenumber = npmod.arange(-half_lat // 2, half_lat // 2)
    gamma_beta = npmod.sin(2 * npmod.pi / num_sites * wavenumber) / mu
    rapidity = npmod.arcsinh(gamma_beta)

    if with_wn:
        return rapidity, wavenumber
    return rapidity


def jw_annihilator_spo(num_sites):
    ops = []
    for isite in range(num_sites):
        paulis = ['I' * (num_sites - isite - 1) + 'X' + 'Z' * isite]
        paulis.append(paulis[0].replace('X', 'Y'))
        coeffs = [0.5 * (1., 1.j, -1., -1.j)[isite % 4]]
        coeffs.append(coeffs[0] * 1.j)
        op = SparsePauliOp(paulis, coeffs)
        ops.append(op)

    return ops


def jw_annihilator_dense(num_sites):
    ops = np.empty((num_sites, 2 ** num_sites, 2 ** num_sites), dtype=np.complex128)
    for isite in range(num_sites):
        op = np.eye(2 ** num_sites, dtype=np.complex128).reshape((2,) * (2 * num_sites))
        for jsite in range(isite):
            dim = num_sites - jsite - 1
            op = np.moveaxis(np.tensordot(SIGMA_Z, op, (1, dim)), 0, dim)
        op *= (1.j) ** isite
        dim = num_sites - isite - 1
        op = np.moveaxis(np.tensordot(SIGMA_PLUS, op, (1, dim)), 0, dim)
        ops[isite] = op.reshape((2 ** num_sites, 2 ** num_sites))

    return ops


def phi_to_ab_spo(phi, rapidity, wavenumber):
    num_sites = len(phi)
    half_lat = num_sites // 2
    twopii = 2.j * np.pi

    phase_k = np.exp(-twopii / num_sites * wavenumber)
    cosh = np.cosh(rapidity / 2.)
    sinh = np.sinh(rapidity / 2.)

    coeffs = [[cosh, phase_k * sinh], [sinh, phase_k * cosh]]
    eikl = np.exp(-twopii / half_lat * wavenumber[:, None] * np.arange(half_lat)[None, :])
    norm = np.sqrt(half_lat * np.cosh(rapidity))

    ops = []
    for ptype in [0, 1]:
        for ik in range(half_lat):
            op = 0
            for isite in range(num_sites):
                field_op = phi[isite]
                if ptype == 1:
                    field_op = field_op.adjoint()
                op += eikl[ik, isite // 2] * coeffs[ptype][isite % 2][ik] * field_op
                op = op.simplify()
            op /= norm[ik]
            ops.append(op)

    return ops


def phi_to_ab_dense(phi, rapidity, wavenumber, npmod=np):
    aop = npmod.empty_like(phi)
    num_sites = phi.shape[0]
    half_lat = num_sites // 2

    phidag = dagger(phi)

    twopii = 2.j * npmod.pi
    eikl = npmod.exp(-twopii / half_lat * wavenumber[:, None] * npmod.arange(half_lat)[None, :])
    eikl = npmod.expand_dims(eikl, (-2, -1))
    phase_k = npmod.expand_dims(npmod.exp(-twopii / num_sites * wavenumber), (-3, -2, -1))
    cosh = npmod.expand_dims(npmod.cosh(rapidity / 2.), (-3, -2, -1))
    sinh = npmod.expand_dims(npmod.sinh(rapidity / 2.), (-3, -2, -1))
    norm = npmod.expand_dims(npmod.sqrt(half_lat * npmod.cosh(rapidity)), (-2, -1))

    summand = cosh * phi[None, ::2] + phase_k * sinh * phi[None, 1::2]
    aop[:half_lat] = npmod.sum(eikl * summand, axis=1) / norm
    summand = sinh * phidag[None, ::2] + phase_k * cosh * phidag[None, 1::2]
    aop[half_lat:] = npmod.sum(eikl * summand, axis=1) / norm

    return cleaned(aop)


def ab_to_phi_sparse(fock_ab, rapidity, wavenumber):
    num_sites = len(fock_ab)
    half_lat = num_sites // 2
    twopii = 2.j * np.pi

    phase_k = np.exp(twopii / num_sites * wavenumber)
    cosh = np.cosh(rapidity / 2.)
    sinh = np.sinh(rapidity / 2.)

    a_coeffs = [cosh, phase_k * sinh]
    b_coeffs = [sinh, phase_k.conjugate() * cosh]
    norm = np.sqrt(half_lat * np.cosh(rapidity))

    ops = []
    for isite in range(num_sites):
        eikl = np.exp(twopii / half_lat * wavenumber * (isite // 2))

        op = 0
        for ik in range(half_lat):
            a_op = fock_ab[ik]
            bdag_op = dagger(fock_ab[half_lat + ik])

            op += (eikl[ik] * a_coeffs[isite % 2][ik] * a_op
                   + eikl[ik].conjugate() * b_coeffs[isite % 2][ik] * bdag_op) / norm[ik]
            op = simplify(op)
        ops.append(op)

    return ops


def ab_to_phi_dense(fock_ab, rapidity, wavenumber, npmod=np):
    phi = npmod.empty_like(fock_ab)
    num_sites = fock_ab.shape[0]
    half_lat = num_sites // 2

    fock_a = fock_ab[:half_lat]
    fock_bdag = dagger(fock_ab[half_lat:], npmod=npmod)

    twopii = 2.j * npmod.pi
    eikl = npmod.exp(twopii / half_lat * wavenumber[:, None] * npmod.arange(half_lat)[None, :])
    eikl = npmod.expand_dims(eikl, (-2, -1))
    phase_k = npmod.expand_dims(npmod.exp(twopii / num_sites * wavenumber), (-3, -2, -1))
    cosh = npmod.expand_dims(npmod.cosh(rapidity / 2.), (-3, -2, -1))
    sinh = npmod.expand_dims(npmod.sinh(rapidity / 2.), (-3, -2, -1))
    norm = npmod.expand_dims(npmod.sqrt(half_lat * npmod.cosh(rapidity)), (-3, -2, -1))

    summand = eikl * cosh * fock_a[:, None, ...]
    summand += eikl.conjugate() * sinh * fock_bdag[:, None, ...]
    phi[::2] = npmod.sum(summand / norm, axis=0)

    summand = eikl * phase_k * sinh * fock_a[:, None, ...]
    summand += eikl.conjugate() * phase_k.conjugate() * cosh * fock_bdag[:, None, ...]
    phi[1::2] = npmod.sum(summand / norm, axis=0)

    return cleaned(phi)


def staggered_mass_term_spo(num_sites, mass):
    paulis = ['I' * (num_sites - isite - 1) + 'Z' + 'I' * isite for isite in range(num_sites)]
    coeffs = np.tile([-1., 1.], num_sites // 2) * 0.5 * mass
    return SparsePauliOp(paulis, coeffs)


def staggered_mass_term_dense(num_sites, mass):
    phi = jw_annihilator_dense(num_sites)
    phidag = dagger(phi)
    signs = np.expand_dims(np.tile([1., -1.], num_sites // 2), (-2, -1))
    mass_term = np.sum(signs * mass * np.einsum('nij,njk->nik', phidag, phi), axis=0)
    return mass_term


def staggered_hopping_term_spo(num_sites, lsp, bc='periodic'):
    param_w = 0.5 / lsp
    paulis = ['I' * (num_sites - isite - 2) + 'XX' + 'I' * isite for isite in range(num_sites - 1)]
    coeffs = [0.5 * param_w] * (num_sites - 1)
    if bc == 'periodic':
        paulis.append('X' + 'Z' * (num_sites - 2) + 'X')
        coeffs.append((1 - 2 * ((num_sites // 2) % 2)) * 0.5 * param_w)
    term = SparsePauliOp(paulis, coeffs)

    paulis = ['I' * (num_sites - isite - 2) + 'YY' + 'I' * isite for isite in range(num_sites - 1)]
    coeffs = [0.5 * param_w] * (num_sites - 1)
    if bc == 'periodic':
        paulis.append('Y' + 'Z' * (num_sites - 2) + 'Y')
        coeffs.append((1 - 2 * ((num_sites // 2) % 2)) * 0.5 * param_w)
    term += SparsePauliOp(paulis, coeffs)

    return term


def staggered_hopping_term_dense(num_sites, lsp, bc='periodic'):
    phi = jw_annihilator_dense(num_sites)
    phidag = dagger(phi)
    if bc == 'periodic':
        phi = np.roll(phi, -1, axis=0)
    else:
        phidag = phidag[:-1]
        phi = phi[1:]

    hopping_term = -0.5j / lsp * np.sum(np.einsum('nij,njk->nik', phidag, phi), axis=0)
    hopping_term += hopping_term.conjugate().T
    return hopping_term
