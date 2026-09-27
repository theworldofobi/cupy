from __future__ import annotations

import scipy.io.wavfile
from scipy.io.wavfile import WavFileWarning  # NOQA

import cupy
from cupyx.scipy.io._convert import _to_device


def read(filename, mmap=False):
    rate, data = scipy.io.wavfile.read(filename, mmap=mmap)
    return rate, _to_device(data)


def write(filename, rate, data):
    scipy.io.wavfile.write(filename, rate, cupy.asnumpy(data))

