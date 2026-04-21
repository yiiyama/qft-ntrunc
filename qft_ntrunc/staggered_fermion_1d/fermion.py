from typing import Any, Optional
from functools import partial
import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit.quantum_info import SparsePauliOp
import jax
import jax.numpy as jnp
from jax.experimental.sparse import bcoo_reduce_sum
from qft_ntrunc.utils import clean_array, dagger, simplify

SIGMA_Z = np.diagflat([1.+0.j, -1.+0.j])
SIGMA_PLUS = np.array([[0., 1.], [0., 0.]], dtype=np.complex128)


def get_wavenumbers(
    num_sites: int
) -> NDArray:
    r"""Return an array of wave numbers.

    The number of modes represented in a lattice of :math:`N` sites is :math:`N/2` due to staggering
    of the fermion. Therefore, the wave number is

    .. math::

        k \in \{-N/4, \dots, N/4-1\}

    if :math:`N/2` is even, and

    .. math::

        k \in \{-(N/2-1)/2, \dots, (N/2-1)/2\}

    otherwise.
    """
    half_lat = num_sites // 2
    if half_lat % 2 == 0:
        min, max = -half_lat // 2, half_lat // 2
    else:
        min, max = -(half_lat - 1) // 2, (half_lat + 1) // 2
    return np.arange(min, max)


def get_rapidity(
    num_sites: int,
    mu: float,
    wavenumber: Optional[ArrayLike] = None,
    with_wn: bool = False
) -> NDArray | tuple[NDArray, NDArray]:
    r"""Return an array of rapidity values for each wave number.

    We use a discretization convention where the momentum :math:`p_k` corresponding to wave number
    :math:`k` is

    .. math::

        p_k = \mu \sin \left( \frac{2 \pi}{N} k \right).

    Rapidity :math:`w_k` is defined by

    .. math::

        w_k = \sinh^{-1} \frac{p_k}{\mu}.

    Args:
        num_sites: Number of sites.
        mu: Mass * lattice spacing.
        wavenumber: The wave number to compute the rapidity for. If None, set to all wave numbers.
        with_wn: Whether to return the wave number array in addition.

    Returns:
        Array of rapidity values (and wave numbers if with_wn=True).
    """
    if wavenumber is None:
        wavenumber = get_wavenumbers(num_sites)
    gamma_beta = np.sin(2 * np.pi / num_sites * wavenumber) / mu
    rapidity = np.arcsinh(gamma_beta)

    if with_wn:
        return rapidity, wavenumber
    return rapidity


def jw_annihilator_spo(num_sites: int) -> list[SparsePauliOp]:
    r"""Jordan-Wigner representation of a chain of annihilators.

    As a convention, an annihilator :math:`A` acts as

    .. math::

        A \lvert 0 \rangle = 0 \\
        A \lvert 1 \rangle = e^{i \theta} \lvert 0 \rangle

    with :math:`\theta` some phase.

    The annihilator is mapped to Pauli products via the Jordan-Wigner transformation

    .. math::

        \Phi_n = \prod_{l<n} (i Z_l) \sigma^{-}_n

    with :math:`\sigma^{-} = 1/2 (X + iY)`.

    Sites are indexed from right to left.
    """
    ops = []
    for isite in range(num_sites):
        paulis = ['I' * (num_sites - isite - 1) + 'X' + 'Z' * isite]
        paulis.append(paulis[0].replace('X', 'Y'))
        coeffs = [0.5 * (1., 1.j, -1., -1.j)[isite % 4]]
        coeffs.append(coeffs[0] * 1.j)
        op = SparsePauliOp(paulis, coeffs)
        ops.append(op)

    return ops


def jw_annihilator_dense(num_sites: int) -> list[np.ndarray]:
    r"""Jordan-Wigner representation of :math:`\Phi_n` implemented in numpy."""
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


def phi_to_ab_sparse(phi: list[Any], rapidity: NDArray, wavenumber: NDArray) -> list[Any]:
    r"""Convert :math:`\Phi_n` to :math:`a_k` and :math:`b_k`.

    .. math::

        \Phi_{2m} = \sqrt{\frac{2}{N}} \sum_{k} \frac{1}{\sqrt{\cosh w_k}}
                  \left( e^{i \frac{2 \pi}{N} 2m k} \cosh \frac{w_k}{2} a_k
                         + e^{-i \frac{2 \pi}{N} 2m k} \sinh \frac{w_k}{2} b^{\dagger}_k \right) \\
        \Phi_{2m+1} = \sqrt{\frac{2}{N}} \sum_{k} \frac{1}{\sqrt{\cosh w_k}}
                  \left( e^{i \frac{2 \pi}{N} (2m+1) k} \sinh \frac{w_k}{2} a_k
                         + e^{-i \frac{2 \pi}{N} (2m+1) k} \cosh \frac{w_k}{2} b^{\dagger}_k \right)

    Therefore

        a_k = \frac{1}{\sqrt{M \cosh w_k}} \sum_{l=0}^{M-1} e^{-i\frac{2\pi}{M} kl} \left[
                \cosh \frac{w_k}{2} \Phi_{2l}
                + e^{-i \frac{2\pi}{N}k} \sinh \frac{w_k}{2} \Phi_{2l + 1}
                \right], \\
        b_k = \frac{1}{\sqrt{M \cosh w_k}} \sum_{l=0}^{M-1} e^{-i\frac{2\pi}{M} kl} \left[
                \sinh \frac{w_k}{2} \hc{\Phi}_{2l}
                + e^{-i \frac{2\pi}{N}k} \cosh \frac{w_k}{2} \hc{\Phi}_{2l + 1}
                \right].

    Args:
        phi: List of sparse ops representing :math:`\Phi_n`.
        rapidity: Rapidity array.
        wavenumber: Wave number array.

    Returns:
        List :math:`[a_{-N/4}, \dots, a_{N/4-1}, b_{-N/4}, \dots, b_{N/4-1}]`
    """
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
                    field_op = dagger(field_op)
                op += eikl[ik, isite // 2] * coeffs[ptype][isite % 2][ik] * field_op
                op = simplify(op)
            op /= norm[ik]
            clean_array(op.coeffs, inplace=True)
            ops.append(op)

    return ops


def phi_to_ab_dense(phi: NDArray, rapidity: NDArray, wavenumber: NDArray, npmod=np) -> np.ndarray:
    r"""Convert :math:`\Phi_n` to :math:`a_k` and :math:`b_k`.

    Args:
        phi: Array for :math:`\Phi_n`.
        rapidity: Rapidity array.
        wavenumber: Wave number array.

    Returns:
        Array :math:`[a_{-N/4}, \dots, a_{N/4-1}, b_{-N/4}, \dots, b_{N/4-1}]`
    """
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

    return clean_array(aop)


def ab_to_phi_sparse(fock_ab: list[Any], rapidity: NDArray, wavenumber: NDArray) -> list[Any]:
    r"""Convert :math:`a_k` and :math:`b_k` to :math:`\Phi_n`.

    .. math::

        \Phi_{2m} = \sqrt{\frac{2}{N}} \sum_{k} \frac{1}{\sqrt{\cosh w_k}}
                  \left( e^{i \frac{2 \pi}{N} 2m k} \cosh \frac{w_k}{2} a_k
                         + e^{-i \frac{2 \pi}{N} 2m k} \sinh \frac{w_k}{2} b^{\dagger}_k \right) \\
        \Phi_{2m+1} = \sqrt{\frac{2}{N}} \sum_{k} \frac{1}{\sqrt{\cosh w_k}}
                  \left( e^{i \frac{2 \pi}{N} (2m+1) k} \sinh \frac{w_k}{2} a_k
                         + e^{-i \frac{2 \pi}{N} (2m+1) k} \cosh \frac{w_k}{2} b^{\dagger}_k \right)

    Args:
        fock_ab: List of sparse ops representing :math:`a_k` and :math:`b_k`.
        rapidity: Rapidity array.
        wavenumber: Wave number array.

    Returns:
        List :math:`[\Phi_n]`.
    """
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
        clean_array(op.coeffs, inplace=True)
        ops.append(op)

    return ops


def ab_to_phi_dense(fock_ab: NDArray, rapidity: NDArray, wavenumber: NDArray, npmod=np) -> NDArray:
    r"""Convert :math:`a_k` and :math:`b_k` to :math:`\Phi_n`.

    Args:
        fock_ab: Array of ops representing :math:`a_k` and :math:`b_k`.
        rapidity: Rapidity array.
        wavenumber: Wave number array.

    Returns:
        Array :math:`[\Phi_n]`.
    """
    phi = npmod.empty_like(fock_ab)
    num_sites = fock_ab.shape[0]
    half_lat = num_sites // 2

    fock_a = fock_ab[:half_lat]
    fock_bdag = dagger(fock_ab[half_lat:])

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

    return clean_array(phi)


def staggered_mass_term_spo(num_sites: int) -> SparsePauliOp:
    r"""Staggered mass Hamiltonian :math:`\sum_n (-)^n \Phi^{\dagger}_n \Phi_n`.

    Our convention is that the even-indexed sites (indexed from right) represent positive charges
    (:math:`\lvert 0 \rangle` is lower mass than :math:`\lvert 1 \rangle`).

    Args:
        num_sites: Number of staggered sites (multiple of 2).

    Returns:
        Mass term represented by a Pauli sum.
    """
    paulis = ['I' * (num_sites - isite - 1) + 'Z' + 'I' * isite for isite in range(num_sites)]
    coeffs = np.tile([-1., 1.], num_sites // 2) * 0.5
    return SparsePauliOp(paulis, coeffs)


def staggered_mass_term_sparse(phi: list[Any]) -> Any:
    """Staggered mass Hamiltonian constructed from a set of site annihilation operators."""
    term = 0
    for op in phi[::2]:
        term += dagger(op) @ op
        term = simplify(term)
    for op in phi[1::2]:
        term -= dagger(op) @ op
        term = simplify(term)
    return term


def staggered_mass_term_dense(num_sites: int, npmod=np) -> NDArray:
    r"""Staggered mass Hamiltonian :math:`\sum_n (-)^n \Phi^{\dagger}_n \Phi_n`."""
    phi = jw_annihilator_dense(num_sites)
    phidag = dagger(phi)
    signs = npmod.expand_dims(npmod.tile([1., -1.], num_sites // 2), (-2, -1))
    mass_term = npmod.sum(signs * npmod.einsum('nij,njk->nik', phidag, phi), axis=0)
    return mass_term


def staggered_hopping_term_spo(num_sites: int, bc: str = 'periodic') -> SparsePauliOp:
    r"""Staggered hopping Hamiltonian.

    .. math::

        H_{\mathrm{hop}} = -i/2 \sum_n (\Phi^{\dagger}_n \Phi_{n+1} - \mathrm{h.c.})

    Args:
        num_sites: Number of staggered sites (multiple of 2).
        bc: Boundary condition ('periodic' or else).
    Returns:
        Hopping term represented by a Pauli sum.
    """
    paulis = ['I' * (num_sites - isite - 2) + 'XX' + 'I' * isite for isite in range(num_sites - 1)]
    coeffs = [0.5] * (num_sites - 1)
    if bc == 'periodic':
        paulis.append('X' + 'Z' * (num_sites - 2) + 'X')
        coeffs.append((1 - 2 * ((num_sites // 2) % 2)) * 0.5)
    term = SparsePauliOp(paulis, coeffs)

    paulis = ['I' * (num_sites - isite - 2) + 'YY' + 'I' * isite for isite in range(num_sites - 1)]
    coeffs = [0.5] * (num_sites - 1)
    if bc == 'periodic':
        paulis.append('Y' + 'Z' * (num_sites - 2) + 'Y')
        coeffs.append((1 - 2 * ((num_sites // 2) % 2)) * 0.5)
    term += SparsePauliOp(paulis, coeffs)

    return term.simplify()


def staggered_hopping_term_sparse(phi: list[Any], bc: str = 'periodic') -> Any:
    """Staggered hopping Hamiltonian constructed from a set of site annihilation operators."""
    term = 0
    for opl, opr in zip(phi[:-1], phi[1:]):
        term += dagger(opl) @ opr
        term = simplify(term)
    if bc == 'periodic':
        term += dagger(phi[-1]) @ phi[0]
        term = simplify(term)
    term *= -0.5j
    term += dagger(term)
    return simplify(term)


def staggered_hopping_term_dense(
    num_sites: int,
    bc: str = 'periodic',
    npmod=np
) -> NDArray:
    """Staggered hopping Hamiltonian."""
    phi = jw_annihilator_dense(num_sites)
    phidag = dagger(phi)
    if bc == 'periodic':
        phi = np.roll(phi, -1, axis=0)
    else:
        phidag = phidag[:-1]
        phi = phi[1:]

    hopping_term = -1.j * npmod.sum(npmod.einsum('nij,njk->nik', phidag, phi), axis=0)
    hopping_term += hopping_term.conjugate().T
    return hopping_term


def _get_basis_indices(num_sites, basis, npmod=np):
    half_lat = num_sites // 2
    hdim = 2 ** num_sites
    # zero-charge subspace is N_C_(N/2) dim
    subdim = np.round(np.prod(np.arange(half_lat + 1, num_sites + 1) / np.arange(1, half_lat + 1)))
    subdim = int(subdim)
    binaries = (npmod.arange(hdim)[:, None] >> np.arange(num_sites)[None, :]) % 2
    if npmod is np:
        size_arg = {}
    else:
        size_arg = {'size': subdim}

    if basis == 'fock_ab':
        # In the Fock representation, first N/2 slots are for positrons
        sign = npmod.repeat(np.array([1, -1]), half_lat)
        charge = npmod.sum(binaries * sign[None, :], axis=1)
        indices = npmod.nonzero(npmod.equal(charge, 0), **size_arg)[0]
    else:
        # Position rep = staggered fermions
        total_excitations = npmod.sum(binaries, axis=1)
        indices = npmod.nonzero(npmod.equal(total_excitations, half_lat), **size_arg)[0]

    return indices


_jit_get_basis_indices = jax.jit(
    partial(_get_basis_indices, npmod=jnp),
    static_argnames=['num_sites', 'basis']
)

def get_basis_indices(num_sites: int, basis: str = 'site_phi', npmod=np) -> NDArray:
    """
    Get the computational basis indices of zero-charge states in the Fock and position reps.
    """
    if npmod is np:
        return _get_basis_indices(num_sites, basis)
    if npmod is jnp:
        return _jit_get_basis_indices(num_sites, basis)
