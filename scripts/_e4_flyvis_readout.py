"""Reproducible flyvis state, coordinate and readout comparison.

Historical initialized/raw-coordinate JSON remains unchanged. This producer
loads an explicit checkpoint or explicitly requests initialized state, retains
complete Jacobians and spectra, and compares finite-difference steps.
"""
from flyvis_sensitivity_revision import main

if __name__ == '__main__':
    from flyvis_windows_io_compat import install
    install()
    main()
