"""Full two-condition sensitivity union for 20 BAAIWorm gains.

The six normalized trajectory summaries are finite responses to log-gain
perturbations at delta=.25. Complete J, Gram matrices and spectra distinguish
numerical rank at an explicit tolerance from cumulative sensitivity mass.
The union combines chemotaxis and locomotion conditions; its effective
spectral dimension need not exceed either separate value.
"""
import os, sys, json, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'fwd_world']
ION = ['gbshl1','gbshk1','gbkvs1','gbegl2','gbegl36','gbkqt3','gbegl19','gbunc2','gbcca1',
       'gbslo1_egl19','gbslo1_unc2','gbslo2_egl19','gbslo2_unc2','gbkcnl','gbnca','gbirk']
CHANNELS = ION + ['syn', 'gj', 'wout', 'gpas']
DELTA = 0.25; NSTEP = 120
WORKER = '/root/_baai_perturb_worker_orig.py'           # unified worker: food + interaction_mode + channels
FOOD_CLOSE = [1.8275, -0.0276, -0.3082]                 # chemotaxis (strong gradient at worm)
FOOD_FAR   = [1.5, 0.2, -0.5]                            # weak gradient = plain locomotion

def spec_for(channel, factor, food):
    s = {'syn': 1.0, 'gj': 1.0, 'wout': 1.0, 'ion': {}, 'passive': {}, 'n_steps': NSTEP,
         'food_xyz': food, 'interaction_mode': 'online'}
    if channel in ION:      s['ion'] = {channel: factor}
    elif channel == 'gpas': s['passive'] = {'gpas': factor}
    elif channel is not None: s[channel] = factor
    return s

def main():
    from _b6_common import measure_union
    result = measure_union(CHANNELS, spec_for, DELTA, FOOD_CLOSE, FOOD_FAR,
                           "B6_full_union_rerun")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
