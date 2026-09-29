from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing

scipy_io_wavfile = pytest.importorskip('scipy.io.wavfile')
import cupyx.scipy.io.wavfile  # NOQA: E402
from cupyx.scipy.io._convert import _to_device  # NOQA: E402


_dtypes = [numpy.uint8, numpy.int16, numpy.int32, numpy.float32, numpy.float64]


@pytest.mark.parametrize('dtype', _dtypes)
@pytest.mark.parametrize('shape', [(100,), (100, 2)])
class TestWavfile:

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_read(self, xp, scp, tmp_path, dtype, shape):
        path = tmp_path / 'test.wav'
        data = testing.shaped_random(shape, numpy, dtype)
        scipy_io_wavfile.write(path, 8000, data)
        rate, out = scp.io.wavfile.read(path)
        assert rate == 8000
        return out

    def test_read_mmap(self, tmp_path, dtype, shape):
        path = tmp_path / 'test.wav'
        data = testing.shaped_random(shape, numpy, dtype)
        scipy_io_wavfile.write(path, 8000, data)
        _, out = cupyx.scipy.io.wavfile.read(path, mmap=True)
        assert isinstance(out, cupy.ndarray)
        testing.assert_array_equal(out, data)

    def test_write(self, tmp_path, dtype, shape):
        path = tmp_path / 'test.wav'
        data = testing.shaped_random(shape, cupy, dtype)
        cupyx.scipy.io.wavfile.write(path, 44100, data)
        rate, out = scipy_io_wavfile.read(path)
        assert rate == 44100
        testing.assert_array_equal(out, data)


def test_big_endian_to_device():
    data = numpy.arange(10, dtype='>i2')
    out = _to_device(data)
    assert isinstance(out, cupy.ndarray)
    assert out.dtype == numpy.int16
    testing.assert_array_equal(out, data)

