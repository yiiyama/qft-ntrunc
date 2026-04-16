"""Utility functions."""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from jax.experimental.sparse import BCOO


def cleaned(op, by_abs=True, npmod=np):
    if by_abs:
        return npmod.where(npmod.isclose(op, 0.), 0., op)
    real = npmod.where(npmod.isclose(op.real, 0.), 0., op.real)
    return real + npmod.where(npmod.isclose(op.imag, 0.), 0., op.imag) * 1.j


def dagger(op):
    if isinstance(op, SparsePauliOp):
        return op.adjoint()
    return op.transpose(tuple(range(op.ndim - 2)) + (op.ndim - 1, op.ndim - 2)).conjugate()


def simplify(op):
    if isinstance(op, SparsePauliOp):
        return op.simplify()
    if isinstance(op, BCOO):
        return op.sum_duplicates()
    op.eliminate_zeros()
    return op