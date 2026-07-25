import sys, numpy as np, inspect
sys.path.insert(0, '/root')
import modworm_recovery_30d as m
print("rollout sig:", inspect.signature(m.rollout))
print("collect_targets sig:", inspect.signature(m.collect_targets))
print("loss sig:", inspect.signature(m.loss))
out = m.rollout(m.THETA_GT, 0)
if isinstance(out, tuple):
    for i, x in enumerate(out):
        print(f"  out[{i}]", type(x).__name__, getattr(x,'shape',getattr(x,'__len__',lambda:None)()))
else:
    print("  out", type(out).__name__, getattr(out,'shape', None))
print("N_SEG=", m.N_SEG, "N_STEPS=", m.N_STEPS, "DT=", m.DT)
print("NAMES[:5]", m.NAMES[:5] if hasattr(m,'NAMES') else None)
print("rollout_src:")
print(inspect.getsource(m.rollout)[:2000])
