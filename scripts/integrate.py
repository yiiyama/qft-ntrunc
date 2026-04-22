import os
import sys
import re
from argparse import ArgumentParser
from pathlib import Path
import time
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.sharding import PartitionSpec as P, AxisType
from qft_ntrunc.staggered_fermion_1d.fermion import (
    jw_annihilator_spo,
    get_wavenumbers,
    get_rapidity,
    ab_to_phi_sparse,
    get_basis_indices
)
from qft_ntrunc.staggered_fermion_1d.schwinger import make_param_apply_h_args
from qft_ntrunc.paulis import apply_h, apply_h_truncated


def make_dpsidt(num_sites, lsp, mass, coupling, profile, mult=1, nmax=None, mesh=None):
    fock_ab = jw_annihilator_spo(num_sites)
    rapidity, wavenumber = get_rapidity(num_sites, mass * lsp, with_wn=True)
    phi = ab_to_phi_sparse(fock_ab, rapidity, wavenumber)
    xmasks, zmasks, coeffs, counts = make_param_apply_h_args(phi, lsp, mass, mesh=mesh)
    coupling_j = np.zeros(xmasks.shape[0])
    coupling_j[1:] = (coupling ** 2) * lsp
    if mesh:
        coupling_j = jax.device_put(coupling_j, xmasks.device)

    if profile is None:
        # Run the simulation for +-3 sigma duration, with 1 sigma = 1/Emax
        sigma = 1. / (mass * np.cosh(rapidity[0]))
        tmax = 6. * sigma
        prof_fn = gaus
        prof_args = (tmax, sigma)
    else:
        name, prof_args = parse_profile(profile)
        tmax = prof_args[0]
        prof_fn = profile_fns[name]
        logging.info('Using profile fn %s with args %s', name, prof_args)

    dt = tmax / 100

    @jax.jit
    def dpsidt(state, tm, xmasks, zmasks, coeffs, counts, coupling_j, prof_args):
        # Scale coeffs[1:] (electric terms)
        coeffs *= coupling_j[:, None] * prof_fn(tm, *prof_args)
        if nmax is None:
            return -1.j * apply_h(state, xmasks, zmasks, coeffs, counts, mult=mult)
        return -1.j * apply_h_truncated(state, xmasks, zmasks, coeffs, counts, 'fock_ab', nmax,
                                        mult=mult)

    if mesh:
        in_specs = (P(None), P(), P('x'), P('x', None), P('x', None), P('x'), P('x'))
        in_specs += ((P(),) * len(prof_args),)
        dpsidt = jax.shard_map(
            dpsidt,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P(None)
        )

    return dpsidt, dt, (xmasks, zmasks, coeffs, counts, coupling_j, prof_args)


def parse_profile(profile):
    match = re.match(r'([a-z0-9]+)\(([0-9.]+)(?:, *([0-9.]+))?(?:, *([0-9.]+))?\)', profile)
    args = tuple(float(v) for v in match.groups()[1:] if v is not None)
    return match.group(1), args


def gaus(t, tmax, sigma):
    t0 = tmax * 0.5
    return jnp.exp(-jnp.square((t - t0) / sigma))


def turnon(t, tmax, sigma):
    t0 = tmax * 0.5
    return jax.lax.select(
        t < t0,
        jnp.exp(-jnp.square((t - t0) / sigma)),
        1.
    )


def erf(t, tmax, sigma):
    t0 = tmax * 0.25
    exponent = (t - t0) / sigma / np.sqrt(2.)
    return 0.5 + 0.5 * jax.scipy.special.erf(exponent)


def erfsymm(t, tmax, sigma):
    t0 = tmax * 0.25
    tcent = tmax * 0.5
    right = jnp.asarray(t > tcent, dtype=np.float64)
    sign = 1. - 2. * right
    offset = tcent * right
    exponent = (t - t0 - offset) * sign / sigma / np.sqrt(2.)
    return 0.5 + 0.5 * jax.scipy.special.erf(exponent)


profile_fns = {'gaus': gaus, 'turnon': turnon, 'erf': erf}


def make_vinit(num_sites, config):
    wavenumbers = get_wavenumbers(num_sites)
    if config == 'collision':
        # Give particle the highest positive momentum
        init_a = wavenumbers.shape[0] - 1
        # Give antiparticle the negative of that momentum
        init_b = np.argwhere(wavenumbers == -wavenumbers[-1])[0][0]
        init = (1 << init_a) + (1 << (num_sites // 2 + init_b))
    elif config == 'vacuum':
        init = 0

    return jax.nn.one_hot(init, (2 ** num_sites), dtype=np.complex128)


def profile_trace(dpsidt, dt, args, vinit, steps, trace_name):
    tpoints = np.linspace(0., dt * steps, steps + 1)
    with jax.profiler.trace(trace_name):
        odeint(dpsidt, vinit, tpoints, *args)


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
        states = odeint(dpsidt, state, tpoints, *args)
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
    parser.add_argument('--config', default='collision')
    parser.add_argument('--lsp', type=float, required=True)
    parser.add_argument('--mass', type=float, required=True)
    parser.add_argument('--coupling', type=float, required=True)
    parser.add_argument('--truncate', type=int, default=None)
    parser.add_argument('--profile')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--out', default='/data/iiyama/qft-ntrunk')
    parser.add_argument('--comp-cache')
    parser.add_argument('--mult', type=int, default=16)
    parser.add_argument('--gpu')
    parser.add_argument('--trace')
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    mesh = None
    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
        if (ndev := len(options.gpu.split(','))) > 1:
            mesh = jax.make_mesh((ndev,), ('x',), (AxisType.Explicit,))

    jax.config.update('jax_enable_x64', True)
    if options.comp_cache:
        jax.config.update("jax_compilation_cache_dir", options.comp_cache)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    dpsidt, dt, args = make_dpsidt(options.num_sites, options.lsp, options.mass, options.coupling,
                                   options.profile, mult=options.mult, nmax=options.truncate,
                                   mesh=mesh)
    vinit = make_vinit(options.num_sites, options.config)

    if options.trace:
        profile_trace(dpsidt, dt, args, vinit, options.steps, options.trace)
        sys.exit(0)

    filename = f'integrate_{options.num_sites}sites_{options.config}'
    if options.profile:
        prof_name, prof_args = parse_profile(options.profile)
        filename += f'_{prof_name}_' + '_'.join(f'{v:.1f}' for v in prof_args)
    filename += f'_a{options.lsp:.1f}_m{options.mass:.1f}_g{options.coupling:.1f}'
    if options.truncate:
        filename += f'_trunc{options.truncate}'
    filename += '.h5'
    try:
        os.makedirs(options.out)
    except FileExistsError:
        pass

    with h5py.File(Path(options.out) / filename, 'w', libver='latest') as out:
        integrate_and_write(dpsidt, dt, args, vinit, options.steps, out)
