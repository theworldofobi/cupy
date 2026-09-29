.. module:: cupyx.scipy.io

Input and output (:mod:`cupyx.scipy.io`)
========================================

.. Hint:: `SciPy API Reference: Input and output (scipy.io) <https://docs.scipy.org/doc/scipy/reference/io.html>`_

.. note::

   Files are read and written on the host with SciPy, which must be
   installed. Arrays are transferred to or from the current device.


MATLAB files
-------------

.. autosummary::
   :toctree: generated/

   loadmat
   savemat
   whosmat


Wav sound files (:mod:`cupyx.scipy.io.wavfile`)
-----------------------------------------------

.. autosummary::
   :toctree: generated/

   wavfile.read
   wavfile.write

