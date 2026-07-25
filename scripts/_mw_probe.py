import sys, json, numpy as np
sys.path.insert(0, '/root')
import modworm_recovery_30d as m
print("dim", m.THETA_GT.size)
t = m.collect_targets(8)
print("loss_GT", float(m.loss(m.THETA_GT, t)))
print("targets_type", type(t).__name__)
if isinstance(t, dict):
    print("targets_keys", list(t.keys()))
print("module_attrs", [x for x in dir(m) if not x.startswith("_")][:50])
