"""Full two-condition sensitivity union for 14 BAAIWorm cell-class gains.

Leak conductance is scaled within each named neuron class. Six normalized
trajectory summaries and log-gain steps delta=.25 define this finite-response
assay. Complete J, Gram matrices and spectra measure concentration and
numerical rank separately; conditions can add directions without increasing
d90 or d99.
"""
import os, json, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'fwd_world']
# neuron classes by name prefix (model skips any absent in its cell set)
CLASSES = {
  'AWA': ['AWA'], 'AWC': ['AWC'], 'ASE': ['ASE'], 'ASH': ['ASH'],     # chemosensory
  'AIY': ['AIY'], 'AIZ': ['AIZ'], 'RIA': ['RIA'], 'AIB': ['AIB'],     # interneurons
  'VA': ['VA'], 'VB': ['VB'], 'DA': ['DA'], 'DB': ['DB'],             # ventral/dorsal motor
  'VD': ['VD'], 'DD': ['DD'],                                          # GABA motor
}
CLABELS = list(CLASSES.keys())
DELTA = 0.25; NSTEP = 120
WORKER = '/root/_baai_perturb_worker_orig.py'
FOOD_CLOSE = [1.8275, -0.0276, -0.3082]; FOOD_FAR = [1.5, 0.2, -0.5]

def spec(cls, factor, food):
    s = {'syn':1.0,'gj':1.0,'wout':1.0,'ion':{},'passive':{},'n_steps':NSTEP,
         'food_xyz':food,'interaction_mode':'online'}
    if cls is not None:
        s['cells'] = CLASSES[cls]; s['passive'] = {'gpas': factor}   # perturb only this class's leak
    return s

def main():
    from _b6_common import measure_union
    result = measure_union(CLABELS, spec, DELTA, FOOD_CLOSE, FOOD_FAR,
                           "B6b_full_union_rerun")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
