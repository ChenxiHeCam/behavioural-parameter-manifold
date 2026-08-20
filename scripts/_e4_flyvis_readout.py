"""E-4 and E-22: is the flyvis high-dimensionality real, or a readout-count effect?

The Discussion contrasts flyvis (effective dimension 43 of 65 cell types) with the
worm models (2-3 of 7) to bound the scope of the low-dimensionality claim. But the
flyvis probe reads out 64 channels and the worm probe six, and rank(J^T W J) is
bounded by the readout count, so the two numbers are not comparable as they stand.

Two controls:
  E-4  the same flyvis Jacobian read out through 64, 32, 16, 8 and 6 channels.
       If the effective dimension tracks the readout count, the published contrast
       is a measurement artefact rather than a difference between the systems.
  E-22 the probe currently covers only the per-cell-type biases, 65 of the model's
       734 free parameters. The other two named groups are added here so the
       reported dimension refers to the model rather than to one parameter group.
"""
import json, time
import numpy as np
import torch

torch.set_num_threads(16)
import flyvis
from flyvis import Network

DELTA = 0.2
T = 20
DT = 1 / 50
N_IN = 721


def eff_dim(H):
    ev = np.sort(np.linalg.eigvalsh(H))[::-1]
    ev = np.maximum(ev, 0)
    tot = max(ev.sum(), 1e-30)
    cs = np.cumsum(ev) / tot
    return (int(np.searchsorted(cs, 0.90) + 1), int(np.searchsorted(cs, 0.99) + 1),
            float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30)),
            [float(x) for x in ev[:12]])


def main():
    t0 = time.time()
    net = Network()
    free = {n: p for n, p in net.named_parameters() if p.requires_grad}
    print("free parameter groups:", {k: int(v.numel()) for k, v in free.items()},
          flush=True)

    xx = np.arange(N_IN)
    mov = np.stack([0.5 + 0.5 * np.sin(2 * np.pi * (xx / 40.0 - f * 0.1))
                    for f in range(T)])[None]
    movie = torch.tensor(mov, dtype=torch.float32).unsqueeze(2)

    def sim():
        with torch.no_grad():
            act = net.simulate(movie, DT)
            return act[0, -5:].mean(0).detach().numpy()

    a0 = sim()
    print(f"one simulate {time.time()-t0:.1f}s, raw response dim {a0.size}", flush=True)

    def binner(g):
        idx = np.linspace(0, a0.size, g + 1).astype(int)
        return lambda a: np.array([a[idx[k]:idx[k + 1]].mean() for k in range(g)])

    GRIDS = [64, 32, 16, 8, 6]
    readers = {g: binner(g) for g in GRIDS}
    base = {g: readers[g](a0) for g in GRIDS}

    # ---- one sweep over every free parameter, all readouts recorded at once ----
    groups = {}
    for gname, p in free.items():
        n = int(p.numel())
        if n > 200:                      # keep the sweep affordable
            sel = np.linspace(0, n - 1, 200).astype(int)
        else:
            sel = np.arange(n)
        groups[gname] = sel

    resp = {g: {} for g in GRIDS}
    flat = {gname: p.data.view(-1) for gname, p in free.items()}
    for gname, sel in groups.items():
        v = flat[gname]
        orig = v.detach().clone()
        cols = {g: [] for g in GRIDS}
        for j, i in enumerate(sel):
            d = DELTA * max(abs(float(orig[i])), 0.05)
            v[i] = orig[i] + d
            ap = sim()
            v[i] = orig[i] - d
            am = sim()
            v[i] = orig[i]
            for g in GRIDS:
                cols[g].append((readers[g](ap) - readers[g](am)) / (2 * d))
            if j % 40 == 0:
                print(f"  {gname} {j}/{len(sel)}  {time.time()-t0:.0f}s", flush=True)
        for g in GRIDS:
            resp[g][gname] = np.array(cols[g]).T      # (g, n_sel)

    out = {"experiment": "E4_E22_flyvis_readout_and_groups",
           "delta": DELTA,
           "free_parameter_groups": {k: int(v.numel()) for k, v in free.items()},
           "params_probed_per_group": {k: int(len(v)) for k, v in groups.items()},
           "readout_sweep": {}, "group_sweep": {}}

    # E-4: bias group only, as published, at shrinking readout dimension
    bias_key = "nodes_bias" if "nodes_bias" in resp[64] else list(groups)[0]
    for g in GRIDS:
        J = resp[g][bias_key] / (np.abs(base[g])[:, None] + 1e-6)
        e90, e99, pr, top = eff_dim(J.T @ J)
        out["readout_sweep"][str(g)] = {"n_readouts": g,
                                        "n_params": int(J.shape[1]),
                                        "eff_dim_90": e90, "eff_dim_99": e99,
                                        "participation_ratio": pr}
        print(f"readout {g:3d} -> eff-dim {e90}/{e99} of {J.shape[1]}  PR {pr:.2f}",
              flush=True)

    # E-22: every group, at the published readout dimension
    for gname in groups:
        J = resp[64][gname] / (np.abs(base[64])[:, None] + 1e-6)
        e90, e99, pr, top = eff_dim(J.T @ J)
        out["group_sweep"][gname] = {"n_params_probed": int(J.shape[1]),
                                     "eff_dim_90": e90, "eff_dim_99": e99,
                                     "participation_ratio": pr}
        print(f"group {gname:22s} -> eff-dim {e90}/{e99} of {J.shape[1]}  PR {pr:.2f}",
              flush=True)

    # all groups jointly
    Jall = np.concatenate([resp[64][g] for g in groups], axis=1)
    Jall = Jall / (np.abs(base[64])[:, None] + 1e-6)
    e90, e99, pr, top = eff_dim(Jall.T @ Jall)
    out["all_groups_joint"] = {"n_params_probed": int(Jall.shape[1]),
                               "eff_dim_90": e90, "eff_dim_99": e99,
                               "participation_ratio": pr}
    print(f"all groups joint -> eff-dim {e90}/{e99} of {Jall.shape[1]}  PR {pr:.2f}",
          flush=True)

    out["published"] = {"eff_dim_90": 43, "n_cell_types": 65, "readout_dim": 64,
                        "note": "bias group only"}
    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open("/root/autodl-tmp/E4_flyvis_readout.json", "w"), indent=2)
    print("E4_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
