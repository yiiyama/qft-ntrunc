import numpy as np


def get_rapidity(num_sites, mu, wavenumber=None, with_wn=False, npmod=np):
    if wavenumber is None:
        half_lat = num_sites // 2
        wavenumber = npmod.arange(-half_lat // 2, half_lat // 2)
    gamma_beta = npmod.sin(2 * npmod.pi / num_sites * wavenumber) / mu
    rapidity = npmod.arcsinh(gamma_beta)

    if with_wn:
        return rapidity, wavenumber
    return rapidity
