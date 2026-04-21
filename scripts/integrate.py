import os
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
    get_rapidity,
    ab_to_phi_sparse,
    get_basis_indices
)
from qft_ntrunc.staggered_fermion_1d.schwinger import (
    make_compressed_param_apply_h,
)
from qft_ntrunc.utils import *
from qft_ntrunc.paulis import make_apply_h

parser = ArgumentParser()
parser.add_argument('--num-sites', type=int, required=True)
parser.add_argument('--lsp', type=float, required=True)
parser.add_argument('--mass', type=float, required=True)
parser.add_argument('--coupling', type=float, required=True)
parser.add_argument('--truncate', type=int)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--out', default='/data/iiyama/qft-ntrunk')
parser.add_argument('--comp-cache')
parser.add_argument('--gpu')
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

fock_ab = jw_annihilator_spo(options.num_sites)
rapidity, wavenumber = get_rapidity(options.num_sites, options.mass * options.lsp, with_wn=True)
phi = ab_to_phi_sparse(fock_ab, rapidity, wavenumber)
truncation = None
if options.truncate:
    truncation = ('fock_ab', options.truncate)
apply_h = make_compressed_param_apply_h(phi, options.lsp, options.mass, options.coupling,
                                        truncation=truncation)
indices = get_basis_indices(options.num_sites, 'fock_ab')

# Initial state = |0..010 1..000>
init = (1 << (options.num_sites // 2 + 1)) + (1 << (options.num_sites // 2 - 1))
idx = np.searchsorted(indices, init)
assert indices[idx] == init
vinit = jax.nn.one_hot(idx, indices.shape[0], dtype=np.complex128)

# Run the simulation for 5 sigma duration, with 1 sigma = 1/Emax
sigma = 1. / (options.mass * np.cosh(rapidity[0]))
tmax = 5. * sigma
t0 = tmax * 0.5
dt = tmax / 100

@jax.jit
def integrate(state, tstart):
    def dpsidt(state, t):
        g = options.coupling * jnp.exp(-0.5 * jnp.square((t - t0) / sigma))
        return -1.j * apply_h(state, indices, g)

    tpoints = jnp.linspace(tstart, tstart + dt * (options.steps + 1), options.steps + 1)
    return odeint(dpsidt, state, tpoints)

filename = f'integrate_{options.num_sites}sites_a{options.lsp}_m{options.mass}_g{options.coupling}'
if options.truncate:
    filename += f'_trunc{options.truncate}'
filename += '.h5'
try:
    os.makedirs(options.out)
except FileExistsError:
    pass

logging.info('Starting integration..')
start_clock = time.time()

with h5py.File(Path(options.out) / filename, 'w', libver='latest') as out:
    dataset = out.create_dataset('states', shape=(101, indices.shape[0]), dtype=np.complex128)
    state = vinit
    for start in range(0, 100, options.steps):
        states = integrate(state, dt * start)
        logging.info('Integrated from %f to %f in %d steps. Elapsed time %f s', dt * start, dt * (start + options.steps + 1), options.steps, time.time() - start_clock)
        nst = min(options.steps, 101 - start)
        dataset[start:start + nst] = states[:nst]
        state = states[nst]
    dataset[-1] = state

logging.info('Integration completed in %f s', time.time() - start_clock)
