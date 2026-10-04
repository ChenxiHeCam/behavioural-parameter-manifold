"""Legacy full-connectome readout producer with an explicit observable choice.

Native muscle_activation is clip((V @ W + 80)/100, 0, .8).
The unbounded muscle_preactivation is V @ W and has different units.
Use _e42_recover_full_spectrum.py for paired readouts from the same saved
rollouts and complete J/G/eigenvalue retention. Historical E60 scalar values
are retained separately and are not labelled as native clipped activation.
"""
import json, os, subprocess, time, sys
from pathlib import Path
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
PY = os.environ.get("BAAI_PYTHON", sys.executable)
WORKER = os.environ.get("BAAI_WORKER", str(Path(__file__).with_name("_baai_worker.py")))
OUTDIR = Path(os.environ.get("PAPER2_OUTPUT_DIR", "results/revision_20261004/E60_explicit_readout"))
OUT = str(OUTDIR / "result.json")
NW = int(os.environ.get("NW", "28"))
N_SAMPLE = 3076
DELTA = 0.5
TSTOP = 1500
POINTS = [{"tag": "muscle_full", "fold": 0, "seed": 0}]
OBSERVABLE = os.environ.get("OBSERVABLE", "muscle_activation")


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
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    conns = list(range(3076))     # every connection, as in E42

    specs = []
    for p in POINTS:
        base = {"point_fold": p["fold"], "point_seed": p["seed"], "tstop": TSTOP,
                "observable": OBSERVABLE}
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

    out = {"experiment": "E60_explicit_readout", "observable": OBSERVABLE,
           "question": "does the low effective dimension hold away from the "
                       "published parameter point?",
           "n_connections_sampled": N_SAMPLE, "delta": DELTA, "tstop_ms": TSTOP,
           "normalization": "baseline temporal SD per output, floor1e-3",
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
        sd = base.std(1).clip(1e-3)[:, None]
        cols, used = [], []
        for i in conns:
            a, b = store.get((tag, i, +1)), store.get((tag, i, -1))
            if a is None or b is None:
                continue
            t = min(base.shape[1], a.shape[1], b.shape[1])
            cols.append((((a[:, :t] - b[:, :t]) / (2 * DELTA)) / sd).ravel())
            used.append(i)
        if len(used) < 100:
            out["per_point"][tag] = {"status": f"only {len(used)} usable columns"}
            continue
        L = min(len(c) for c in cols)
        J = np.array([c[:L] for c in cols], dtype=np.float64).T
        M = J.T @ J
        Hm = M / max(np.trace(M), 1e-30)
        H[tag] = Hm
        np.savez_compressed(OUTDIR / f"{tag}_matrices.npz", J=J, G=M,
                            baseline=base, output_scale=sd, connection_indices=np.asarray(used),
                            eigenvalues=np.linalg.eigvalsh(M)[::-1])
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
