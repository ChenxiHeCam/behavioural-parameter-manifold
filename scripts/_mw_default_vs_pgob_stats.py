"""Default-arm vs PGOB-arm statistical contrast for modWorm N=50 multistart.

Honest, self-contained framing (no cross-encoding to OWMD state-proportion
features, which would be a unit mismatch -- see project memory surrogate
caveat). Within the modWorm 30-d forward model we contrast:

  DEFAULT arm : behaviour distance of the random default-init theta to the
                theta*-target behaviour distribution  (= d_init, N=50)
  PGOB    arm : behaviour distance of the PGOB-recovered theta to the same
                theta*-target                          (= d_rec, N=50)

We report, per method (CMA / SPSA):
  - default mean W1 + bootstrap 95% CI
  - PGOB    mean W1 + bootstrap 95% CI
  - paired improvement (default - PGOB) mean + bootstrap CI
  - Wilcoxon signed-rank one-sided p (PGOB < default)
  - permutation p (label-shuffle of default/PGOB, 20000 perms)
  - effect size (median ratio default/PGOB)
This completes the 'default-arm' statistics that were previously partial.
"""
import json, sys
import numpy as np
from scipy.stats import wilcoxon

def boot_ci(v, nb=20000, seed=7, alpha=0.05):
    v=np.asarray([x for x in v if np.isfinite(x)],float)
    r=np.random.default_rng(seed)
    m=[r.choice(v,size=len(v),replace=True).mean() for _ in range(nb)]
    return [float(np.percentile(m,100*alpha/2)),float(np.percentile(m,100*(1-alpha/2)))]

def perm_p(default, pgob, nb=20000, seed=11):
    # paired permutation: randomly swap default<->pgob within each pair,
    # test statistic = mean(default - pgob); one-sided (default > pgob)
    d=np.asarray(default,float); p=np.asarray(pgob,float)
    diff=d-p; obs=diff.mean(); r=np.random.default_rng(seed); n=len(diff)
    cnt=0
    for _ in range(nb):
        s=r.choice([-1.0,1.0],size=n)
        if (diff*s).mean()>=obs: cnt+=1
    return (cnt+1)/(nb+1)

def analyze(path):
    j=json.load(open(path))
    trials=j["trials"]
    d_default=np.array([t["d_init"] for t in trials],float)  # default-init arm
    d_pgob   =np.array([t["d_rec"]  for t in trials],float)   # PGOB-recovered arm
    re_default=np.array([t["relerr_init"] for t in trials],float)
    re_pgob   =np.array([t["relerr_rec"]  for t in trials],float)
    imp=d_default-d_pgob
    try:
        w_stat,w_p=wilcoxon(d_default,d_pgob,alternative="greater")
    except Exception:
        w_stat,w_p=None,None
    return {
        "method": j["method"], "N": j["N"], "n_iter": j["n_iter"],
        "default_W1_mean": float(d_default.mean()),
        "default_W1_median": float(np.median(d_default)),
        "default_W1_ci95": boot_ci(d_default),
        "pgob_W1_mean": float(d_pgob.mean()),
        "pgob_W1_median": float(np.median(d_pgob)),
        "pgob_W1_ci95": boot_ci(d_pgob),
        "improvement_default_minus_pgob_mean": float(imp.mean()),
        "improvement_ci95": boot_ci(imp),
        "median_ratio_default_over_pgob": float(np.median(d_default)/max(np.median(d_pgob),1e-9)),
        "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p_onesided_pgob_lt_default": float(w_p) if w_p is not None else None,
        "permutation_p_onesided": float(perm_p(d_default,d_pgob)),
        "relerr_default_mean": float(re_default.mean()),
        "relerr_pgob_mean": float(re_pgob.mean()),
        "n_pgob_better": int((d_pgob<d_default).sum()),
        # manifold metrics carried through
        "manifold_ratio_param_over_behav": j.get("manifold_ratio_param_over_behav"),
        "theta_dispersion_normstd": j.get("theta_dispersion_normstd"),
        "manifold_pca_eff_dim_participation": j.get("manifold_pca_eff_dim_participation"),
    }

if __name__=="__main__":
    out={}
    for p in sys.argv[1:]:
        try:
            r=analyze(p); out[r["method"]]=r
            print(f"[{r['method']}] default_W1={r['default_W1_mean']:.3f} "
                  f"CI{r['default_W1_ci95']}  pgob_W1={r['pgob_W1_mean']:.3f} "
                  f"CI{r['pgob_W1_ci95']}  ratio={r['median_ratio_default_over_pgob']:.2f} "
                  f"wilcoxon_p={r['wilcoxon_p_onesided_pgob_lt_default']} "
                  f"perm_p={r['permutation_p_onesided']:.5f} "
                  f"n_better={r['n_pgob_better']}/{r['N']} "
                  f"manifold_ratio={r['manifold_ratio_param_over_behav']}")
        except FileNotFoundError:
            print(f"[skip] {p} not found yet")
    json.dump(out, open("/root/_mw_default_vs_pgob_N50_stats.json","w"), indent=2)
    print("WROTE /root/_mw_default_vs_pgob_N50_stats.json")
