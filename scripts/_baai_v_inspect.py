import numpy as np, json, os, pickle, sys

base = "/root/BAAIWorm_clone_ready/BAAIWorm-main/eworm_learn/trial10"
out = {}

for fn in ["v_final_eworm_v4.npy", "v_initial_eworm_v4.npy",
           "weights_optimal_eworm_v4.npy", "x_optimal_eworm_v4.npy"]:
    p = os.path.join(base, fn)
    try:
        a = np.load(p, allow_pickle=True)
        out[fn] = {"shape": list(a.shape), "dtype": str(a.dtype),
                   "min": float(np.asarray(a).min()),
                   "max": float(np.asarray(a).max())}
    except Exception as e:
        out[fn] = {"err": repr(e)}
print(json.dumps(out, indent=2))
