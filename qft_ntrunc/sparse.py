import numpy as np
from qiskit.quantum_info import SparsePauliOp
from jax.experimental.sparse import BCOO


def dagger(op):
    if isinstance(op, SparsePauliOp):
        return op.adjoint()
    return op.conjugate().T


def simplify(op):
    if isinstance(op, SparsePauliOp):
        return op.adjoint()
    if isinstance(op, BCOO):
        return op.sum_duplicates()
    op.eliminate_zeros()
    return op


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


def staggered_mass_term_spo(num_sites, mass):
    paulis = ['I' * (num_sites - isite - 1) + 'Z' + 'I' * isite for isite in range(num_sites)]
    coeffs = np.tile([-1., 1.], num_sites // 2) * 0.5 * mass
    return SparsePauliOp(paulis, coeffs)


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


def schwinger_interaction_term_spo(num_sites, coupling_j):
    term = 0
    for bound in range(1, num_sites):
        paulis = ['I' * num_sites] * bound
        coeffs = [0.5 * (1. - 2. * (isite % 2)) for isite in range(bound)]
        op = SparsePauliOp(paulis, coeffs)
        paulis = ['I' * (num_sites - isite - 1) + 'Z' + 'I' * isite for isite in range(bound)]
        op += SparsePauliOp(paulis, 0.5)
        term += coupling_j * (op @ op).simplify()

    return term


def schwinger_hamiltonian_spo(num_sites, lsp, mass, coupling_j, bc='periodic'):
    hamiltonian = staggered_mass_term_spo(num_sites, mass)
    hamiltonian += staggered_hopping_term_spo(num_sites, lsp, bc=bc)
    if coupling_j != 0.:
        hamiltonian += schwinger_interaction_term_spo(num_sites, coupling_j)
    return hamiltonian.simplify()
