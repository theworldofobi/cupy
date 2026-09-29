from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing

scipy_io = pytest.importorskip('scipy.io')
import scipy.sparse  # NOQA: E402

import cupyx.scipy.io  # NOQA: E402
import cupyx.scipy.sparse  # NOQA: E402


def _write_sample(path):
    cell = numpy.empty((1, 2), dtype=object)
    cell[0, 0] = numpy.arange(3.0)
    cell[0, 1] = 'text'
    scipy_io.savemat(path, {
        'x': numpy.arange(6.0).reshape(2, 3),
        'n': numpy.arange(4, dtype=numpy.int32),
        's': 'hello',
        'c': cell,
        'st': {'a': numpy.ones(2), 'name': 'n'},
        'sp': scipy.sparse.csc_matrix(numpy.eye(3)),
    })


class TestLoadmat:

    @pytest.mark.parametrize('name', ['x', 'n'])
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_numeric(self, xp, scp, tmp_path, name):
        path = tmp_path / 'test.mat'
        _write_sample(path)
        return scp.io.loadmat(path)[name]

    def test_types(self, tmp_path):
        path = tmp_path / 'test.mat'
        _write_sample(path)
        out = cupyx.scipy.io.loadmat(path)

        assert isinstance(out['x'], cupy.ndarray)
        assert isinstance(out['__header__'], bytes)
        # Character arrays cannot be represented by CuPy.
        assert isinstance(out['s'], numpy.ndarray)
        assert out['s'][0] == 'hello'
        # Cell arrays stay on the host, their contents are transferred.
        assert out['c'].dtype == object
        assert isinstance(out['c'][0, 0], cupy.ndarray)
        testing.assert_array_equal(out['c'][0, 0], [[0.0, 1.0, 2.0]])
        # Structs (record arrays) likewise.
        st = out['st'][0, 0]
        assert isinstance(st['a'], cupy.ndarray)
        assert st['name'][0] == 'n'
        # Sparse matrices keep their format.
        assert isinstance(out['sp'], cupyx.scipy.sparse.csc_matrix)
        testing.assert_array_equal(out['sp'].toarray(), numpy.eye(3))

    def test_sparse_array(self, tmp_path):
        path = tmp_path / 'test.mat'
        _write_sample(path)
        out = cupyx.scipy.io.loadmat(path, spmatrix=False)
        assert isinstance(out['sp'], cupyx.scipy.sparse.csc_array)

    def test_mat_struct(self, tmp_path):
        path = tmp_path / 'test.mat'
        _write_sample(path)
        out = cupyx.scipy.io.loadmat(
            path, struct_as_record=False, squeeze_me=True)
        assert isinstance(out['st'], cupyx.scipy.io.matlab.mat_struct)
        assert isinstance(out['st'].a, cupy.ndarray)
        assert out['st'].name == 'n'

    def test_mdict(self, tmp_path):
        path = tmp_path / 'test.mat'
        _write_sample(path)
        mdict = {'existing': 1}
        out = cupyx.scipy.io.loadmat(path, mdict=mdict)
        assert out is mdict
        assert mdict['existing'] == 1
        assert isinstance(mdict['x'], cupy.ndarray)

    def test_whosmat(self, tmp_path):
        path = tmp_path / 'test.mat'
        _write_sample(path)
        assert cupyx.scipy.io.whosmat(path) == scipy_io.whosmat(path)


class TestSavemat:

    @pytest.mark.parametrize('format', ['4', '5'])
    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    def test_roundtrip(self, tmp_path, format, dtype):
        path = tmp_path / 'test.mat'
        a = testing.shaped_arange((2, 3), cupy, dtype)
        cupyx.scipy.io.savemat(path, {'a': a}, format=format)
        out = cupyx.scipy.io.loadmat(path)['a']
        assert isinstance(out, cupy.ndarray)
        testing.assert_array_equal(out, a)

    def test_nested_and_sparse(self, tmp_path):
        path = tmp_path / 'test.mat'
        a = cupy.arange(3.0)
        mdict = {
            'st': {'a': a, 'l': [a, 'text']},
            'sp': cupyx.scipy.sparse.csr_matrix(cupy.eye(3)),
        }
        cupyx.scipy.io.savemat(path, mdict)
        # The input is not modified.
        assert mdict['st']['a'] is a
        out = scipy_io.loadmat(path)
        testing.assert_array_equal(out['st'][0, 0]['a'], [[0.0, 1.0, 2.0]])
        testing.assert_array_equal(out['sp'].toarray(), numpy.eye(3))

    def test_mat_struct_not_modified(self, tmp_path):
        path = tmp_path / 'test.mat'
        st = cupyx.scipy.io.matlab.mat_struct()
        st._fieldnames = ['a']
        st.a = cupy.arange(3.0)
        cupyx.scipy.io.savemat(path, {'st': st})
        assert isinstance(st.a, cupy.ndarray)
        out = scipy_io.loadmat(path, struct_as_record=False,
                               squeeze_me=True)
        testing.assert_array_equal(out['st'].a, [0.0, 1.0, 2.0])