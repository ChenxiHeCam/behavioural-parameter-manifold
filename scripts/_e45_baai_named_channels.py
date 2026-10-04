"""E-45: the named ion-channel and wiring mechanisms, re-measured without the
observable rank cap.

The deposited per-channel result on this model reads out six behavioural scalars,
so the Hessian it produces cannot have rank above six no matter what the circuit
does. Every biological statement the paper makes about which mechanisms behaviour
constrains -- wiring and passive properties stiff, fast calcium and
calcium-activated potassium sloppy -- rests on that measurement.

Here the same named mechanisms are scaled across all cells that express them and
the motor-neuron trajectory is used as the observable, giving thousands of
observables against twenty parameters. The channel list is read from the compiled
mechanism set, so it is the model's own, and each mechanism is checked for a
measurable effect before being counted.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import glob, json, re, subprocess, sys, time, hashlib
from pathlib import Path
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
PY = os.environ.get("BAAI_PYTHON", sys.executable)
WORKER = os.environ.get("BAAI_WORKER", str(Path(__file__).with_name("_baai_worker.py")))
MODDIR = os.path.join(os.environ.get("BAAI_ROOT", "/root/autodl-tmp/BAAIWorm_fresh"), "eworm/components/mechanism/modfile")
OUTDIR = Path(os.environ.get("PAPER2_OUTPUT_DIR", "results/revision_20261004/E45_named_channels"))
DELTA = 0.5
TSTOP = 1500
NWORK = int(os.environ.get("NWORK", "120"))

# named gene products, as the paper names them
GENE = {"shl1": "SHL-1 (Kv)", "shk1": "SHK-1 (Kv)", "kvs1": "KVS-1 (Kv)",
        "egl2": "EGL-2 (Kv-eag)", "egl36": "EGL-36 (Kv)", "kqt3": "KQT-3 (KCNQ)",
        "irk": "IRK (Kir)", "egl19": "EGL-19 (L-Ca)", "unc2": "UNC-2 (PQ-Ca)",
        "cca1": "CCA-1 (T-Ca)", "slo1": "SLO-1 (BK)", "slo2": "SLO-2 (SK/BK)",
        "kcnl": "KCNL (SK)", "nca": "NCA (Na-leak)", "cainternm": "Ca handling"}


def channels():
    out = []
    for f in sorted(glob.glob(os.path.join(MODDIR, "*.mod"))):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        m = re.search(r"SUFFIX\s+(\w+)", txt)
        if not m:
            continue
        suf = m.group(1)
        g = re.findall(r"RANGE[^\n]*?\b(gb\w+)", txt)
        if g:
            out.append((suf, g[0]))
    return out


def call(spec):
    try:
        p = subprocess.run([PY, WORKER, json.dumps(spec)],
                           capture_output=True, text=True, timeout=900)
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                return np.array(json.loads(ln[7:])["motor"], dtype=np.float32)
    except Exception:
        pass
    return None


def job(a):
    return a[:-1], call(a[-1])


def eff(H):
    ev = np.sort(np.linalg.eigvalsh(H))[::-1]
    ev = np.maximum(ev, 0)
    if ev.sum() <= 0:
        return 0, 0, 0.0
    cs = np.cumsum(ev) / ev.sum()
    return (int(np.searchsorted(cs, 0.90) + 1),
            int(np.searchsorted(cs, 0.99) + 1),
            float(ev.sum() ** 2 / (ev ** 2).sum()))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chans = channels()
    print(f"{len(chans)} conductances found in the compiled mechanism set", flush=True)

    specs = [("base", 0, {"conn": -1, "tstop": TSTOP})]
    for suf, g in chans:
        for sg in (+1, -1):
            specs.append((suf, sg, {"conn": -1, "tstop": TSTOP, "channel": g, "mech": suf,
                                  "sign": sg, "delta": DELTA}))
    print(f"{len(specs)} rollouts on {NWORK} workers", flush=True)

    store = {}
    with ProcessPoolExecutor(max_workers=NWORK, mp_context=CTX) as pool:
        for k, v in pool.map(job, specs, chunksize=1):
            store[k] = v

    base = store.get(("base", 0))
    if base is None:
        print("no baseline", flush=True)
        return
    sd = base.std(1).clip(1e-3)[:, None]
    cols, names, dead = [], [], []
    for suf, g in chans:
        p, q = store.get((suf, +1)), store.get((suf, -1))
        if p is None or q is None:
            dead.append((g, "simulation failed"))
            continue
        t = min(base.shape[1], p.shape[1], q.shape[1])
        col = (((p[:, :t] - q[:, :t]) / (2 * DELTA)) / sd).ravel()
        if np.abs(col).max() < 1e-9:
            dead.append((g, "no measurable effect"))
            continue
        cols.append(col)
        names.append(GENE.get(suf, suf))
    n = min(len(c) for c in cols)
    J = np.array([c[:n] for c in cols], dtype=np.float64).T
    H = J.T @ J
    e90, e99, pr = eff(H)
    elast = np.linalg.norm(J, axis=0)
    order = np.argsort(elast)[::-1]

    out = {"experiment": "E45_baaiworm_named_channels_rich",
           "model": "BAAIWorm point circuit",
           "observable": "motor-neuron membrane potentials",
           "n_observables": int(J.shape[0]), "n_mechanisms": len(names),
           "delta": DELTA,
           "eff_dim_90": e90, "eff_dim_99": e99, "participation_ratio": pr,
           "ranking_stiff_to_sloppy": [[names[i], float(elast[i])] for i in order],
           "mechanisms_without_measurable_effect": dead,
           "published_six_observable": {"eff_dim_90": 2, "eff_dim_99": 3,
                                        "n_mechanisms": 20},
           "eigenvalues": np.linalg.eigvalsh(H)[::-1].tolist(),
           "matrix_file": "full_matrices.npz",
           "worker_sha256": hashlib.sha256(Path(WORKER).read_bytes()).hexdigest(),
           "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "elapsed_sec": round(time.time() - t0, 1)}
    np.savez_compressed(OUTDIR / "full_matrices.npz", J=J, G=H, baseline=base,
                        output_scale=sd, eigenvalues=np.linalg.eigvalsh(H)[::-1],
                        **{f"response_{k[0]}_{k[1]:+d}": v for k,v in store.items() if v is not None})
    json.dump(out, open(OUTDIR / "result.json", "w"), indent=2)
    print("=== E-45 RESULT ===", flush=True)
    print(f"  {len(names)} live mechanisms, {J.shape[0]} observables", flush=True)
    print(f"  eff-dim {e90} (90%) / {e99} (99%)   PR {pr:.2f}", flush=True)
    print(f"  six-observable version gave 2/3 of 20", flush=True)
    print("  stiffest:", [names[i] for i in order[:5]], flush=True)
    print("  sloppiest:", [names[i] for i in order[-5:]], flush=True)
    if dead:
        print(f"  no measurable effect: {[d[0] for d in dead]}", flush=True)
    print("E45_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
