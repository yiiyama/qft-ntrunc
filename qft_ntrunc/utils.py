"""Utility functions."""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from jax.experimental.sparse import BCOO


def clean_array(op, by_abs=False, inplace=False, npmod=np):
    if by_abs:
        return npmod.where(npmod.isclose(op, 0.), 0., op)
    arr = npmod.where(npmod.isclose(op.real, 0.), 0., op.real).astype(op.dtype)
    if op.dtype == np.complex128:
        arr += npmod.where(npmod.isclose(op.imag, 0.), 0., op.imag) * 1.j
    if inplace:
        # Valid only for npmod=np
        op[...] = arr
    else:
        return arr


def dagger(op):
    if isinstance(op, SparsePauliOp):
        return op.adjoint()
    return op.transpose(tuple(range(op.ndim - 2)) + (op.ndim - 1, op.ndim - 2)).conjugate()


def simplify(op):
    if isinstance(op, SparsePauliOp):
        op = op.simplify()
        clean_array(op._coeffs, inplace=True)
        return op
    if isinstance(op, BCOO):
        return op.sum_duplicates()
    op.eliminate_zeros()
    return op


def identity(op):
    if isinstance(op, SparsePauliOp):
        return SparsePauliOp('I' * op.num_qubits)
    # if isinstance(op, BCOO):
    #     return bcoo_eye
