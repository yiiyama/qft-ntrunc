import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import time
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from qft_ntrunc.staggered_fermion_1d.fermion import (
    jw_annihilator_spo,
    get_wavenumbers,
    get_rapidity,
    ab_to_phi_sparse,
    get_basis_indices
)
from qft_ntrunc.staggered_fermion_1d.schwinger import make_param_apply_h_args
from qft_ntrunc.paulis import apply_h, apply_h_truncated


def make_dpsidt(num_sites, lsp, mass, coupling, mult=1, nmax=None):
    fock_ab = jw_annihilator_spo(num_sites)
    rapidity, wavenumber = get_rapidity(num_sites, mass * lsp, with_wn=True)
    phi = ab_to_phi_sparse(fock_ab, rapidity, wavenumber)
    xmasks, zmasks, coeffs, counts = make_param_apply_h_args(phi, lsp, mass)

    # Run the simulation for 5 sigma duration, with 1 sigma = 1/Emax
    sigma = 1. / (mass * np.cosh(rapidity[0]))
    tmax = 5. * sigma
    t0 = tmax * 0.5
    dt = tmax / 100

    @jax.jit
    def dpsidt(state, tm, args):
        g2lsp = (coupling ** 2) * lsp * jnp.exp(-jnp.square((tm - t0) / sigma))
        xmasks, zmasks, coeffs, counts = args
        # Scale coeffs[1:] (electric terms)
        coeffs *= jnp.where(jnp.arange(xmasks.shape[0]) == 0, 1., g2lsp)[:, None]
        if nmax is None:
            return -1.j * apply_h(state, xmasks, zmasks, coeffs, counts, mult=mult)
        return -1.j * apply_h_truncated(state, xmasks, zmasks, coeffs, counts, 'fock_ab', nmax,
                                        mult=mult)

    return dpsidt, dt, (xmasks, zmasks, coeffs, counts)


def profile_trace(dpsidt, dt, args, vinit, steps, trace_name):
    tpoints = np.linspace(0., dt * steps, steps + 1)
    with jax.profiler.trace(trace_name):
        odeint(dpsidt, vinit, tpoints, args)


def integrate_and_write(dpsidt, dt, args, vinit, steps, out):
    num_sites = int(np.round(np.log2(vinit.shape[0])))
    indices = get_basis_indices(num_sites, 'fock_ab')
    dataset = out.create_dataset('states', shape=(101, indices.shape[0]), dtype=np.complex128)

    logging.info('Starting integration..')
    start_clock = time.time()

    state = vinit
    for start in range(0, 100, steps):
        end = start + steps
        tpoints = np.linspace(dt * start, dt * end, steps + 1)
        states = odeint(dpsidt, state, tpoints, args)
        logging.info('Integrated from %f to %f in %d steps. Elapsed time %f s',
                     dt * start, dt * end, options.steps, time.time() - start_clock)
        nout = min(steps, 101 - start)
        dataset[start:start + nout] = states[:nout, indices]
        state = states[nout]
    dataset[-1] = state[indices]

    logging.info('Integration completed in %f s', time.time() - start_clock)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-sites', type=int, required=True)
    parser.add_argument('--lsp', type=float, required=True)
    parser.add_argument('--mass', type=float, required=True)
    parser.add_argument('--coupling', type=float, required=True)
    parser.add_argument('--truncate', type=int, default=None)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--out', default='/data/iiyama/qft-ntrunk')
    parser.add_argument('--comp-cache')
    parser.add_argument('--mult', type=int, default=16)
    parser.add_argument('--gpu')
    parser.add_argument('--profile')
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    jax.config.update('jax_enable_x64', True)
    if options.comp_cache:
        jax.config.update("jax_compilation_cache_dir", options.comp_cache)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    dpsidt, dt, args = make_dpsidt(options.num_sites, options.lsp, options.mass, options.coupling,
                                   mult=options.mult, nmax=options.truncate)

    # Make initial state vector
    wavenumbers = get_wavenumbers(options.num_sites)
    # Give particle the highest positive momentum
    init_a = wavenumbers.shape[0] - 1
    # Give antiparticle the negative of that momentum
    init_b = np.argwhere(wavenumbers == -wavenumbers[-1])[0][0]
    init = (1 << init_a) + (1 << (options.num_sites // 2 + init_b))
    # idx = np.searchsorted(indices, init)
    # assert indices[idx] == init
    # vinit = jax.nn.one_hot(idx, indices.shape[0], dtype=np.complex128)
    vinit = jax.nn.one_hot(init, (2 ** options.num_sites), dtype=np.complex128)

    if options.profile:
        profile_trace(dpsidt, dt, args, vinit, options.steps, options.profile)
        sys.exit(0)

    filename = f'integrate_{options.num_sites}sites_a{options.lsp}_m{options.mass}_g{options.coupling}'
    if options.truncate:
        filename += f'_trunc{options.truncate}'
    filename += '.h5'
    try:
        os.makedirs(options.out)
    except FileExistsError:
        pass

    with h5py.File(Path(options.out) / filename, 'w', libver='latest') as out:
        integrate_and_write(dpsidt, dt, args, vinit, options.steps, out)
