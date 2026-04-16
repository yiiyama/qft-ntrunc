"""Utility functions."""
import numpy as np


def cleaned(op, by_abs=True, npmod=np):
    if by_abs:
        return npmod.where(npmod.isclose(op, 0.), 0., op)
    real = npmod.where(npmod.isclose(op.real, 0.), 0., op.real)
    return real + npmod.where(npmod.isclose(op.imag, 0.), 0., op.imag) * 1.j
