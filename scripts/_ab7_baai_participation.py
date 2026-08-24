"""AB7: recompute the stiff/sloppy eigenvector participation where it is defined.

The deposited participation figures, stiff 3.42 and sloppy 4.68, come from the
modWorm mechanism Hessian, whose two smallest eigenvalues are -6.5e-16 and
-1.5e-15. Those are numerically zero and of indeterminate sign, so the
corresponding eigenvector orientations, and any participation computed from them,
are not fixed by the data. The claim they support, that sloppy directions are
distributed combinations rather than unused single parameters, is worth keeping if
it can be measured somewhere it is well posed.

BAAIWorm is that place: the noise floor is exactly zero because repeat simulations
are bit-identical, and every one of its sixteen named conductances responds. This
rebuilds the same 16-mechanism Gauss-Newton curvature as E45b and reports, for each
eigenvector, its eigenvalue and its participation ratio over the mechanism
coordinates, so that a reader can see which eigenvectors are resolved above the
floor before reading any participation from them.
"""
import json, os, re, glob, subprocess, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
PY = "/root/miniconda3/bin/python"
WORKER = "/root/autodl-tmp/cell/baai_worker.py"
MODDIR = "/root/autodl-tmp/BAAIWorm_fresh/eworm/components/mechanism/modfile"
OUT = "/root/autodl-tmp/cell/AB7_baai_participation.json"
DELTA, TSTOP, NWORK = 0.5, 1500, 26

GENE = {"shl1": "SHL-1", "shk1": "SHK-1", "kvs1": "KVS-1", "egl2": "EGL-2",
        "egl36": "EGL-36", "kqt3": "KQT-3", "egl19": "EGL-19", "unc2": "UNC-2",
        "cca1": "CCA-1", "slo1_egl19": "SLO-1/EGL-19", "slo1_unc2": "SLO-1/UNC-2",
        "slo2_egl19": "SLO-2/EGL-19", "slo2_unc2": "SLO-2/UNC-2", "kcnl": "KCNL",
        "nca": "NCA", "irk": "IRK"}


def channels():
    out = []
    for f in sorted(glob.glob(os.path.join(MODDIR, "*.mod"))):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        m = re.search(r"SUFFIX\s+(\w+)", txt)
        if not m:
            continue
        g = re.findall(r"RANGE[^\n]*?\b(gb\w+)", txt)
        if g:
            out.append((m.group(1), g[0]))
    return out


def call(spec):
    p = subprocess.run([PY, WORKER, json.dumps(spec)], capture_output=True,
                       text=True, timeout=1800)
    if "not found" in p.stderr:
        return None, "mechanism not found"
    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT "):
            return np.array(json.loads(ln[7:])["motor"], dtype=np.float32), None
    return None, "no result"


def job(a):
    key, spec = a
    v, err = call(spec)
    return key, v, err


def participation(v):
    """Participation ratio of one unit eigenvector over the mechanism coordinates."""
    w = np.abs(np.asarray(v, float)) ** 2
    return float(w.sum() ** 2 / (w ** 2).sum())


def main():
    t0 = time.time()
    chans = channels()
    specs = [(("base", 0), {"conn": -1, "tstop": TSTOP})]
    for suf, g in chans:
        for sg in (+1, -1):
            specs.append(((suf, sg), {"conn": -1, "tstop": TSTOP, "channel": g,
                                      "mech": suf, "sign": sg, "delta": DELTA}))
    print(f"{len(specs)} rollouts over {len(chans)} conductances", flush=True)

    store, errs = {}, {}
    with ProcessPoolExecutor(max_workers=NWORK, mp_context=CTX) as pool:
        for k, v, e in pool.map(job, specs, chunksize=1):
            store[k] = v
            if e:
                errs[k] = e
    if errs:
        raise SystemExit(f"perturbations failed, refusing to report: {errs}")

    base = store[("base", 0)]
    sd = base.std(1).clip(1e-3)[:, None]
    cols, names = [], []
    for suf, g in chans:
        p, q = store[(suf, +1)], store[(suf, -1)]
        t = min(base.shape[1], p.shape[1], q.shape[1])
        cols.append((((p[:, :t] - q[:, :t]) / (2 * DELTA)) / sd).ravel())
        names.append(GENE.get(suf, suf))

    n = min(len(c) for c in cols)
    J = np.array([c[:n] for c in cols], dtype=np.float64).T
    H = J.T @ J
    ev, V = np.linalg.eigh(H)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]

    # the resolution floor: the largest eigenvalue times double precision, below
    # which an eigenvector's orientation is not determined by the data
    floor = float(ev.max()) * 1e-12
    rows = []
    for i in range(len(ev)):
        vec = V[:, i]
        top = int(np.argmax(np.abs(vec)))
        rows.append({"rank": i + 1, "eigenvalue": float(ev[i]),
                     "resolved": bool(ev[i] > floor),
                     "participation": participation(vec),
                     "largest_loading": names[top],
                     "loadings": {names[j]: round(float(vec[j]), 4)
                                  for j in range(len(names))}})

    resolved = [r for r in rows if r["resolved"]]
    stiff, sloppy = resolved[0], resolved[-1]
    cs = np.cumsum(ev) / ev.sum()
    out = {
        "experiment": "AB7_baai_stiff_sloppy_participation",
        "question": "are stiff and sloppy directions distributed combinations, "
                    "measured where the eigenvectors are resolved?",
        "supersedes_for_this_claim": "CP_coupling.json (modWorm; two eigenvalues "
                                     "at 1e-15, orientation not determined)",
        "n_mechanisms": len(names), "mechanisms": names,
        "delta": DELTA, "tstop_ms": TSTOP, "n_observables": int(n),
        "eff_dim_90": int(np.searchsorted(cs, 0.90) + 1),
        "eff_dim_99": int(np.searchsorted(cs, 0.99) + 1),
        "resolution_floor": floor,
        "n_resolved_eigenvectors": len(resolved),
        "stiffest": {"eigenvalue": stiff["eigenvalue"],
                     "participation": stiff["participation"],
                     "largest_loading": stiff["largest_loading"]},
        "sloppiest_resolved": {"eigenvalue": sloppy["eigenvalue"],
                               "participation": sloppy["participation"],
                               "largest_loading": sloppy["largest_loading"]},
        "participation_all_resolved": [r["participation"] for r in resolved],
        "per_eigenvector": rows,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT, flush=True)
    print(f"{len(resolved)} of {len(ev)} eigenvectors resolved above {floor:.3g}")
    print(f"stiffest   lambda {stiff['eigenvalue']:.4g}  participation "
          f"{stiff['participation']:.2f} of {len(names)}  ({stiff['largest_loading']})")
    print(f"sloppiest  lambda {sloppy['eigenvalue']:.4g}  participation "
          f"{sloppy['participation']:.2f} of {len(names)}  ({sloppy['largest_loading']})")


if __name__ == "__main__":
    main()
