"""A wall-clock budget for a single simulation.

A divergent parameter set does not make an ODE solver fail; it makes it succeed
very slowly. RK45 shrinks the step without bound and the call simply never
returns, which on a worker pool looks exactly like work in progress: one core
pegged, the rest idle, no output. Every simulation in this project therefore runs
under a budget, and a call that exceeds it is treated the same way as one that
produced non-finite output -- as a parameter set the model cannot simulate.
"""
import time as _time

_T0 = [0.0]
_LIMIT = [20.0]


class Budget(Exception):
    """Raised inside the right-hand side once a simulation exceeds its budget."""


def start(limit=None):
    if limit is not None:
        _LIMIT[0] = float(limit)
    _T0[0] = _time.perf_counter()


def check():
    if _time.perf_counter() - _T0[0] > _LIMIT[0]:
        raise Budget()
