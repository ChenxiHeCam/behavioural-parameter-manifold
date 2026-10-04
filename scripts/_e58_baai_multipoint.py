"""E-58: is the connectome-scale low dimension a property of one parameter point?

Every reviewer of the current draft raised the same objection: the flagship
BAAIWorm measurement is a local Gauss-Newton curvature evaluated at the published
parameter set, and the paper speaks of a manifold. The five-point and
twenty-four-point checks that exist cover only the seven-mechanism modWorm space;
the 3076-dimensional measurement itself has never been repeated anywhere else in
parameter space.

This repeats it. The whole weight vector is displaced to new operating points,
theta_i = theta_i * exp(U(-ln f, ln f)) drawn independently per connection at
f = 1.25 and f = 1.5, two points each. At every point, including the published
one, the same randomly chosen subsample of 300 connections is perturbed both
ways and the Gauss-Newton spectrum assembled over the motor output, so the five
measurements are like for like. If the effective dimension stays low at every
point, the low dimension is a property of the region of parameter space the
model occupies; if it does not, the published point is special and the paper's
language must change.

A point only counts if its baseline still produces structured motor output
(finite, non-constant); a displacement that silences the model is reported as
such, not folded into the spectrum.
"""
import json, os, subprocess, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
PY = "/root/miniconda3/bin/python"
WORKER = "/root/autodl-tmp/cell/baai_worker.py"
OUT = "/root/autodl-tmp/cell/E58_baai_multipoint.json"
NW = int(os.environ.get("NW", "28"))
N_SAMPLE = 300
DELTA = 0.5
TSTOP = 1500
POINTS = [{"tag": "published", "fold": 0, "seed": 0},
          {"tag": "f1.25_a", "fold": 1.25, "seed": 11},
          {"tag": "f1.25_b", "fold": 1.25, "seed": 12},
          {"tag": "f1.5_a", "fold": 1.5, "seed": 21},
          {"tag": "f1.5_b", "fold": 1.5, "seed": 22}]


def call(spec):
    try:
        p = subprocess.run([PY, WORKER, json.dumps(spec)],
                           capture_output=True, text=True, timeout=1800)
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                return np.array(json.loads(ln[7:])["motor"], dtype=np.float32)
    except Exception:
        pass
    return None


def job(a):
    tag, i, sg, spec = a
    return (tag, i, sg), call(spec)


def eff(H):
    ev = np.sort(np.maximum(np.linalg.eigvalsh(H), 0))[::-1]
    if ev.sum() <= 0:
        return None
    cs = np.cumsum(ev) / ev.sum()
    pos = ev[ev > ev.max() * 1e-14]
    return {"eff_dim_90": int(np.searchsorted(cs, 0.90) + 1),
            "eff_dim_99": int(np.searchsorted(cs, 0.99) + 1),
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "spectrum_decades": float(np.log10(pos.max() / pos.min()))
            if len(pos) > 1 else 0.0}


def leading(H, k):
    ev, V = np.linalg.eigh(H)
    return V[:, np.argsort(ev)[::-1][:k]]


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    conns = sorted(rng.choice(3076, N_SAMPLE, replace=False).tolist())

    specs = []
    for p in POINTS:
        base = {"point_fold": p["fold"], "point_seed": p["seed"], "tstop": TSTOP}
        specs.append((p["tag"], -1, 0, dict(base, conn=-1)))
        for i in conns:
            for sg in (+1, -1):
                specs.append((p["tag"], i, sg,
                              dict(base, conn=i, sign=sg, delta=DELTA)))
    print(f"{len(specs)} rollouts: {N_SAMPLE} connections x {len(POINTS)} "
          f"operating points", flush=True)

    store = {}
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX) as pool:
        for n, (k, v) in enumerate(pool.map(job, specs, chunksize=1), 1):
            store[k] = v
            if n % 200 == 0:
                print(f"  {n}/{len(specs)}  {time.time()-t0:.0f}s", flush=True)

    out = {"experiment": "E58_baai_multipoint",
           "question": "does the low effective dimension hold away from the "
                       "published parameter point?",
           "n_connections_sampled": N_SAMPLE, "delta": DELTA, "tstop_ms": TSTOP,
           "points": POINTS, "per_point": {}, "pairwise_stiff_overlap": []}

    H = {}
    for p in POINTS:
        tag = p["tag"]
        base = store.get((tag, -1, 0))
        if base is None:
            out["per_point"][tag] = {"status": "baseline failed"}
            print(f"  {tag}: baseline failed", flush=True)
            continue
        if not np.all(np.isfinite(base)) or float(base.std()) < 1e-9:
            out["per_point"][tag] = {"status": "baseline silent or non-finite",
                                     "baseline_std": float(base.std())}
            print(f"  {tag}: baseline silent (std {base.std():.2e})", flush=True)
            continue
        sd = base.std(0).clip(1e-6)
        cols, used = [], []
        for i in conns:
            a, b = store.get((tag, i, +1)), store.get((tag, i, -1))
            if a is None or b is None:
                continue
            t = min(base.shape[0], a.shape[0], b.shape[0])
            cols.append((((a[:t] - b[:t]) / (2 * DELTA)) / sd).ravel())
            used.append(i)
        if len(used) < 100:
            out["per_point"][tag] = {"status": f"only {len(used)} usable columns"}
            continue
        L = min(len(c) for c in cols)
        J = np.array([c[:L] for c in cols], dtype=np.float64).T
        M = J.T @ J
        Hm = M / max(np.trace(M), 1e-30)
        H[tag] = Hm
        e = eff(Hm)
        e.update({"status": "ok", "n_usable": len(used),
                  "baseline_std": float(base.std())})
        out["per_point"][tag] = e
        print(f"  {tag:10s} eff {e['eff_dim_90']:3d}/{e['eff_dim_99']:3d} of "
              f"{len(used)}  PR {e['participation_ratio']:.2f}  "
              f"{e['spectrum_decades']:.1f}dec", flush=True)

    tags = list(H)
    for a in range(len(tags)):
        for b in range(a + 1, len(tags)):
            k = max(2, min(out["per_point"][tags[a]]["eff_dim_90"],
                           out["per_point"][tags[b]]["eff_dim_90"]))
            A, B = leading(H[tags[a]], k), leading(H[tags[b]], k)
            sv = np.clip(np.linalg.svd(A.T @ B, compute_uv=False), 0, 1)
            out["pairwise_stiff_overlap"].append(
                {"a": tags[a], "b": tags[b], "k": int(k),
                 "mean_principal_cos": float(np.mean(sv))})
            print(f"  overlap {tags[a]} vs {tags[b]} (k={k}): "
                  f"{np.mean(sv):.3f}", flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"saved -> {OUT}  ({out['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
