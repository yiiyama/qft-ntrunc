import numpy as np


sigma_z = np.diagflat([1.+0.j, -1.+0.j])
sigma_plus = np.array([[0., 1.], [0., 0.]], dtype=np.complex128)


def cleaned(op, by_abs=True, npmod=np):
    if by_abs:
        return npmod.where(npmod.isclose(op, 0.), 0., op)
    real = npmod.where(npmod.isclose(op.real, 0.), 0., op.real)
    return real + npmod.where(npmod.isclose(op.imag, 0.), 0., op.imag) * 1.j


def dagger(op, npmod=np):
    return npmod.moveaxis(op.conjugate(), -2, -1)


def jw_annihilator_dense(num_sites):
    ops = np.empty((num_sites, 2 ** num_sites, 2 ** num_sites), dtype=np.complex128)
    for isite in range(num_sites):
        op = np.eye(2 ** num_sites, dtype=np.complex128).reshape((2,) * (2 * num_sites))
        for jsite in range(isite):
            dim = num_sites - jsite - 1
            op = np.moveaxis(np.tensordot(sigma_z, op, (1, dim)), 0, dim)
        op *= (1.j) ** isite
        dim = num_sites - isite - 1
        op = np.moveaxis(np.tensordot(sigma_plus, op, (1, dim)), 0, dim)
        ops[isite] = op.reshape((2 ** num_sites, 2 ** num_sites))

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


def staggered_mass_term_dense(num_sites, mass):
    phi = jw_annihilator_dense(num_sites)
    phidag = dagger(phi)
    signs = np.expand_dims(np.tile([1., -1.], num_sites // 2), (-2, -1))
    mass_term = np.sum(signs * mass * np.einsum('nij,njk->nik', phidag, phi), axis=0)
    return mass_term
