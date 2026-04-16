from typing import Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray


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
