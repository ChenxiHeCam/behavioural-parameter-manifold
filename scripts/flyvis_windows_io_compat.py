"""Close HDF5 handles before recreating a datamate array file on Windows.

datamate 1.0.0 _write_h5 unlinks a newly opened file in its exception path.
Windows refuses that unlink while the HDF5 handle is open. This replacement
uses a context manager, preserves array bytes and the SWMR storage convention,
and changes no network computation. It is unnecessary on Linux.
"""
import os


def install():
    if os.name != 'nt':
        return False
    import h5py
    import numpy as np
    import datamate.io as dmio
    import importlib
    directory = importlib.import_module('datamate.directory')

    def write_h5(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, libver='latest', mode='w') as handle:
            handle['data'] = np.asarray(value)
            handle.swmr_mode = True
            assert handle.swmr_mode

    dmio._write_h5 = write_h5
    directory._write_h5 = write_h5
    return True
