from __future__ import annotations

import scipy.io.matlab
from scipy.io.matlab import (  # NOQA
    MatlabFunction, MatlabObject, MatlabOpaque, MatReadError, MatReadWarning,
    MatWriteError, mat_struct)

from cupyx.scipy.io._convert import _to_device, _to_host


def loadmat(file_name, mdict=None, appendmat=True, **kwargs):
    variables = _to_device(
        scipy.io.matlab.loadmat(file_name, None, appendmat, **kwargs))
    if mdict is None:
        return variables
    mdict.update(variables)
    return mdict


def savemat(file_name, mdict, appendmat=True, format='5',
            long_field_names=False, do_compression=False, oned_as='row'):
    
    scipy.io.matlab.savemat(
        file_name, _to_host(mdict), appendmat=appendmat, format=format,
        long_field_names=long_field_names, do_compression=do_compression,
        oned_as=oned_as)


def whosmat(file_name, appendmat=True, **kwargs):
    return scipy.io.matlab.whosmat(file_name, appendmat, **kwargs)

