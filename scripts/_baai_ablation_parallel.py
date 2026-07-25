"""Parallel driver for Claim-4 BAAIWorm modeling-ablation.

Reuses ALL logic from /root/_baai_ablation_n10.py (run_one, build_cond_cache,
SOTA weights). The only change: the nested (level, seed) loop is dispatched
across a multiprocessing.Pool so the ~30 independent NEURON cells run
concurrently instead of serially.

Each worker process:
  - imports the BAAIWorm sim stack fresh (its own NEURON instance),
  - loads SOTA base weights + cond_cache once (per-worker, via initializer),
  - executes run_one(level, seed) for tasks handed to it by the pool.

Results are streamed to --out as each cell finishes (crash-safe).
"""
from __future__ import annotations
import os, sys, json, time, argparse, logging
import multiprocessing as mp

# Limit per-worker BLAS/NEURON thread fan-out so workers don't oversubscribe.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, "/root")
import _baai_ablation_n10 as abl  # reuse run_one, build_cond_cache, scale/corrupt, SOTA

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("BAAI_ABL_PAR")

# Per-worker globals (populated by _init).
_BASE_W = None
_COND_CACHE = None


def _init():
    """Worker initializer: load SOTA base weights + cond cache once per process."""
    global _BASE_W, _COND_CACHE
    # Smoke override: shrink SPSA iterations only (keep N_STEPS=100 so the
    # baseline trajectory pkl cache key still resolves).
    _sm = os.environ.get("ABL_SMOKE_NITER")
    if _sm:
        abl.N_ITER = int(_sm)
    from recovery.utils.io_utils import load_pickle
    res = load_pickle(abl.SOTA)
    _BASE_W = res["recovered_weights"]
    _COND_CACHE = abl.build_cond_cache()
    if not _COND_CACHE:
        raise RuntimeError("worker: empty cond_cache")
    log.info("[worker %d] init done cond_cache n=%d", os.getpid(), len(_COND_CACHE))


def _task(args):
    level, seed = args
    t0 = time.time()
    try:
        r = abl.run_one(level, seed, _BASE_W, _COND_CACHE)
    except Exception as ex:
        log.exception("cell fail L%d s%d", level, seed)
        r = {"ok": False, "level": level, "seed": seed, "err": str(ex)}
    r["_wall"] = time.time() - t0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1,2,3,4,5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny N_ITER/N_STEPS to verify parallelism+memory fast")
    ap.add_argument("--out", default="/root/baaiworm_ablation_claim4_n5.json")
    args = ap.parse_args()
    if args.smoke:
        os.environ["ABL_SMOKE_NITER"] = "1"
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    tasks = [(L, s) for L in levels for s in seeds]
    nw = min(args.workers, len(tasks))
    log.info("PARALLEL ablation: %d cells, %d workers -> %s",
             len(tasks), nw, args.out)

    results = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nw, initializer=_init, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_task, tasks):
            results.append(r)
            ok = r.get("ok")
            log.info("DONE L%s s%s ok=%s (%d/%d) wall=%.0fs",
                     r.get("level"), r.get("seed"), ok, len(results), len(tasks),
                     r.get("_wall", 0.0))
            with open(args.out, "w") as fh:
                json.dump({"sim": "BAAIWorm", "results": results,
                           "config": {"levels": levels, "seeds": seeds,
                                      "workers": nw, "parallel": True}}, fh, indent=2)
    log.info("ALL DONE %d cells in %.0fs -> %s",
             len(results), time.time() - t0, args.out)


if __name__ == "__main__":
    main()
