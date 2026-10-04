"""Rerun the point-circuit probe, retaining every response and full spectrum.

Run on Linux with BAAI_ROOT, BAAI_MECHANISM_DIR, BAAI_PYTHON and
PAPER2_OUTPUT_DIR set. The rerun is separate from all historical result files.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from _baai_worker_support import select_readout

ROOT = Path(os.environ.get("BAAI_ROOT", "/root/autodl-tmp/BAAIWorm_fresh"))
OUT = Path(os.environ.get("PAPER2_OUTPUT_DIR", "results/revision_20261004"))
WORKER = Path(__file__).with_name("_baai_worker.py")
PY = os.environ.get("BAAI_PYTHON", sys.executable)
NWORK = int(os.environ.get("NWORK", "24"))
DELTA = float(os.environ.get("DELTA", "0.5"))
TSTOP = float(os.environ.get("TSTOP", "2000"))
N_CONN = int(os.environ.get("N_CONN", "3076"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(job):
    idx, sign = job
    spec = {"conn": idx, "sign": sign, "delta": DELTA,
            "tstop": TSTOP, "observable": "motor_voltage"}
    stem = f"connection_{idx:04d}_{sign:+d}"
    target = OUT / "responses" / f"{stem}.npz"
    if target.exists():
        cached = json.loads(target.with_suffix(".json").read_text())
        if cached["spec"] != spec:
            raise ValueError(f"Cached rollout differs from requested configuration: {stem}")
        with np.load(target) as z:
            return job, z["voltage"]
    p = subprocess.run([PY, str(WORKER), json.dumps(spec)],
                       capture_output=True, text=True, timeout=900)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
            voltage = np.asarray(result.pop("motor"), dtype=np.float64)
            if not np.isfinite(voltage).all():
                raise ValueError(f"Non-finite response in {stem}")
            np.savez_compressed(target, voltage=voltage)
            (target.with_suffix(".json")).write_text(json.dumps({"spec": spec, "result": result}))
            return job, voltage
    (OUT / f"failure_{stem}.log").write_text(p.stdout + p.stderr)
    raise RuntimeError(f"Rollout failed: {stem}, exit {p.returncode}")


def spectrum(J):
    G = J.T @ J
    eigenvalues = np.linalg.eigvalsh(G)[::-1]
    nonnegative = np.maximum(eigenvalues, 0)
    total = nonnegative.sum()
    cumulative = np.cumsum(nonnegative) / total if total else nonnegative
    # Report a numerical algebraic tolerance separately from a physical noise floor.
    tolerance = float(max(J.shape) * np.finfo(np.float64).eps * nonnegative[0])
    return G, {"eigenvalues": eigenvalues.tolist(),
               "eff_dim_90": int(np.searchsorted(cumulative, .90) + 1) if total else 0,
               "eff_dim_99": int(np.searchsorted(cumulative, .99) + 1) if total else 0,
               "participation_ratio": float(total ** 2 / (nonnegative ** 2).sum()) if total else 0,
               "numerical_rank": int((nonnegative > tolerance).sum()),
               "eigenvalue_rank_tolerance": tolerance,
               "rank_tolerance_interpretation": "floating-point Gram-matrix tolerance; not a measured simulation noise floor"}


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "responses").mkdir(exist_ok=True)
    sys.path[:0] = [str(ROOT), str(ROOT / "eworm/ghost_in_mesh_sim")]
    import pickle
    tuned = ROOT / "eworm/ghost_in_mesh_sim/data/tuned/video_offline"
    with open(tuned / "video_offline_abscircuit.pkl", "rb") as f:
        circuit = pickle.load(f)
    if len(circuit.connections) != 3076:
        raise ValueError("Unexpected model connection count")
    with open(tuned / "video_offline_wout.pkl", "rb") as f:
        wout = pickle.load(f)
    config = {"delta": DELTA, "tstop_ms": TSTOP, "n_connections": N_CONN,
              "n_workers": NWORK, "observable": "motor_voltage", "simulation_dt_ms": 5/3,
              "parameter_coordinate": "relative weight increment (theta=theta0*(1+u)); central secant at u=+-delta",
              "recorded_input_present": (tuned / "video_offline_loc.npy").exists(),
              "normalization_motor_voltage": "baseline standard deviation over time per neuron; floor 1e-3 mV",
              "normalization_muscle": "baseline standard deviation over time per muscle; floor 1e-3 in each observable's units"}
    sources = [Path(__file__), WORKER, WORKER.with_name("_baai_worker_support.py"),
               ROOT / "eworm/network/config.json", tuned / "video_offline_abscircuit.pkl",
               tuned / "video_offline_wout.pkl", ROOT / "eworm/network/point_circuit.py",
               ROOT / "eworm/network/transform.py", ROOT / "eworm/ghost_in_mesh_sim/worm_neural_network.py"]
    sources += sorted((ROOT / "eworm/components/param").rglob("*.json"))
    sources += sorted((ROOT / "eworm/components/model").rglob("*.hoc"))
    sources += sorted((ROOT / "eworm/components/mechanism/modfile").glob("*.mod"))
    if (tuned / "video_offline_loc.npy").exists():
        sources.append(tuned / "video_offline_loc.npy")
    provenance = {"config": config, "source_sha256": {str(p): sha256(p) for p in sources if p.is_file()},
                  "python": sys.version, "numpy": np.__version__, "historical_result_overwritten": False}
    if (OUT / "E42_provenance.json").exists():
        previous = json.loads((OUT / "E42_provenance.json").read_text())
        comparable = {k: v for k, v in config.items() if k != "n_workers"}
        previous_config = {k: v for k, v in previous["config"].items() if k != "n_workers"}
        if comparable != previous_config or previous["source_sha256"] != provenance["source_sha256"]:
            raise ValueError("Cached model, sources or assay changed; use a new output directory")
    (OUT / "E42_provenance.json").write_text(json.dumps(provenance, indent=2))
    jobs = [(-1, 0)] + [(i, sign) for i in range(N_CONN) for sign in (1, -1)]
    store = {}
    with ThreadPoolExecutor(max_workers=NWORK) as pool:
        for n, (job, voltage) in enumerate(pool.map(run, jobs), 1):
            store[job] = voltage
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} rollouts, elapsed {time.time()-start:.1f}s", flush=True)
    output = {"experiment": "E42_full_spectrum_rerun_20261004", "config": config,
              "readouts": {}, "provenance_file": "E42_provenance.json"}
    for observable in ("motor_voltage", "muscle_preactivation", "muscle_activation"):
        base = select_readout(store[(-1, 0)], observable, wout)
        scale = base.std(axis=1).clip(1e-3)[:, None]
        cols = []
        for i in range(N_CONN):
            p = select_readout(store[(i, 1)], observable, wout)
            q = select_readout(store[(i, -1)], observable, wout)
            cols.append(((p-q)/(2*DELTA)/scale).ravel())
        J = np.asarray(cols, dtype=np.float64).T
        G, summary = spectrum(J)
        np.savez_compressed(OUT / f"E42_{observable}_matrices.npz", J=J, G=G, baseline=base,
                            output_scale=scale, connection_indices=np.arange(N_CONN),
                            eigenvalues=np.asarray(summary["eigenvalues"]))
        summary.update({"observable": observable, "n_parameters": N_CONN,
                        "n_observables": J.shape[0], "baseline_std": float(base.std()),
                        "matrix_file": f"E42_{observable}_matrices.npz"})
        output["readouts"][observable] = summary
        print(observable, summary["eff_dim_90"], summary["eff_dim_99"], flush=True)
    output["elapsed_seconds"] = time.time() - start
    (OUT / "E42_full_spectrum_rerun.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
