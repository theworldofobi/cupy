from __future__ import annotations

import copy

import numpy
import scipy.sparse
from scipy.io.matlab import mat_struct

import cupy
import cupyx.scipy.sparse


def _walk(obj, convert):
    ret = convert(obj)
    if ret is not NotImplemented:
        return ret
    if isinstance(obj, numpy.ndarray) and obj.dtype.hasobject:
        # Copying keeps ndarray subclasses such as MatlabObject.
        out = obj.copy()
        fields = [out] if out.dtype.names is None else [
            out[name] for name in out.dtype.names
            if out.dtype[name].kind == 'O']
        for field in fields:
            for idx in numpy.ndindex(field.shape):
                field[idx] = _walk(field[idx], convert)
        return out
    if isinstance(obj, mat_struct):
        out = copy.copy(obj)
        for name, value in vars(obj).items():
            if name != '_fieldnames':
                setattr(out, name, _walk(value, convert))
        return out
    if isinstance(obj, dict):
        return {k: _walk(v, convert) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_walk(v, convert) for v in obj)
    return obj


def _leaf_to_device(obj):
    if isinstance(obj, numpy.ndarray) and obj.dtype.kind in 'biufc':
        if not obj.dtype.isnative:
            # CuPy does not support non-native byte order (e.g. RIFX files).
            obj = obj.astype(obj.dtype.newbyteorder('='))
        return cupy.asarray(obj)
    if scipy.sparse.issparse(obj):
        # e.g. csc_matrix -> cupyx.scipy.sparse.csc_matrix
        cls = getattr(cupyx.scipy.sparse, type(obj).__name__, None)
        if cls is None:
            return cupyx.scipy.sparse.csc_matrix(obj.tocsc())
        return cls(obj)
    return NotImplemented


def _leaf_to_host(obj):
    if isinstance(obj, cupy.ndarray) or cupyx.scipy.sparse.issparse(obj):
        return obj.get()
    return NotImplemented


def _to_device(obj):
    return _walk(obj, _leaf_to_device)


def _to_host(obj):
    return _walk(obj, _leaf_to_host)

