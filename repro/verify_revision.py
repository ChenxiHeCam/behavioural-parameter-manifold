"""Verify deposited revision results without installing or running simulators.

Usage: python repro/verify_revision.py [--json-out report.json]
Run --seal-artifacts once AFTER the final figure build to record byte identities.
Sealing is separate from verification: it does not establish numerical correctness.

Default checks recompute the complete E42 Gram matrices and spectra, and all
flyvis spectra from retained Jacobians. --fast uses deterministic Gram probes
instead of the complete E42 multiplication/eigendecomposition; the report states
this reduced scope. Neither mode certifies forward-model reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

# Avoid excessive BLAS thread creation on machines with many logical processors.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

FIGURES = ("output_concentration", "measurement_dependence", "context_dependence",
           "sensitivity_and_recovery", "cell_network_scaling", "Fig_visual_sensitivity")
MANIFEST = "repro/revision_artifacts.sha256.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, rtol=2e-8, atol=1e-11):
    require(np.allclose(actual, expected, rtol=rtol, atol=atol),
            f"{label}: numerical mismatch")


def mass_summary(eigenvalues):
    ev = np.maximum(np.asarray(eigenvalues, dtype=float), 0)
    total = float(ev.sum())
    require(total > 0, "Spectrum has no positive sensitivity")
    return {"eff_dim_90": int(np.searchsorted(np.cumsum(ev), .9 * total) + 1),
            "eff_dim_99": int(np.searchsorted(np.cumsum(ev), .99 * total) + 1),
            "participation_ratio": float(total ** 2 / (ev @ ev))}


def check_summary(ev, stored, label):
    summary = mass_summary(ev)
    for key in ("eff_dim_90", "eff_dim_99"):
        require(summary[key] == stored[key], f"{label}: {key} mismatch")
    close(summary["participation_ratio"], stored["participation_ratio"], label + " PR")
    close(ev, stored["eigenvalues"], label + " complete eigenvalues",
          rtol=2e-7, atol=max(float(ev[0]) * 2e-12, 1e-12))
    return summary


def svd_check(J, stored, label):
    singular = np.linalg.svd(J, compute_uv=False)
    ev = np.pad(singular ** 2, (0, J.shape[1] - len(singular)))
    summary = check_summary(ev, stored, label)
    rank = int(np.count_nonzero(singular > np.finfo(float).eps * max(J.shape) * singular[0]))
    require(rank == stored["numeric_rank"], label + ": numeric rank mismatch")
    require(stored["n_readouts"] == J.shape[0] and stored["n_params"] == J.shape[1],
            label + ": declared shape mismatch")
    close(ev.sum(), stored["total_sensitivity"], label + " trace")
    return dict(summary, numeric_rank=rank)


def verify_e42(root, fast):
    result = root / "results/revision_20261004"
    d = read_json(result / "E42_full_spectrum_rerun.json")
    p = d["config"]["n_connections"]
    require(p == 3076 and d["config"]["delta"] == .5, "E42 assay definition changed")
    require(d["config"]["recorded_input_present"] is False, "E42 input reconstruction changed")
    summaries = {}
    for name, stored in d["readouts"].items():
        with np.load(result / stored["matrix_file"], allow_pickle=False) as arrays:
            J, G, ev = arrays["J"], arrays["G"], arrays["eigenvalues"]
            require(J.shape == (stored["n_observables"], p) and G.shape == (p, p),
                    name + ": inconsistent matrix shapes")
            require(ev.shape == (p,), name + ": truncated spectrum")
            require(np.all(np.isfinite(J)) and np.all(np.isfinite(G)), name + ": nonfinite matrix")
            close(G, G.T, name + " symmetry", atol=max(np.max(np.abs(G)) * 1e-12, 1e-12))
            close(np.diag(G), np.einsum("ij,ij->j", J, J), name + " Gram diagonal")
            close(np.trace(G), ev.sum(), name + " spectral trace")
            require(np.all(np.diff(ev) <= max(ev[0] * 1e-13, 1e-12)), name + ": spectrum not ordered")
            require(ev[-1] >= -ev[0] * 2e-12, name + ": materially negative eigenvalue")
            if fast:
                probes = np.random.default_rng(20261004).normal(size=(p, 12))
                close(G @ probes, J.T @ (J @ probes), name + " 12 Gram probes",
                      atol=max(np.max(np.abs(G)) * 1e-10, 1e-10))
                recomputed = ev
            else:
                close(G, J.T @ J, name + " complete Gram", atol=max(np.max(np.abs(G)) * 2e-12, 1e-12))
                recomputed = np.maximum(np.linalg.eigvalsh(G)[::-1], 0)
                close(recomputed, ev, name + " eigendecomposition", rtol=2e-7,
                      atol=max(ev[0] * 2e-12, 1e-12))
            summary = check_summary(recomputed, stored, name)
            tol = stored["eigenvalue_rank_tolerance"]
            require(np.count_nonzero(recomputed > tol) == stored["numerical_rank"], name + ": Gram rank mismatch")
            np.testing.assert_array_equal(arrays["connection_indices"], np.arange(p))
            baseline, scales = arrays["baseline"], arrays["output_scale"]
            require(baseline.size == J.shape[0], name + ": baseline/readout mismatch")
            close(scales, np.maximum(baseline.std(axis=1, keepdims=True), 1e-3), name + " baseline weights")
            norms = np.linalg.norm(J, axis=0)
            column_cutoff = 1e-12 * max(float(norms.max()), 1e-30)
            require(int(np.count_nonzero(norms < column_cutoff)) == stored["n_parameters_with_no_measurable_effect"],
                    name + ": numerical-zero column count mismatch")
            close(np.percentile(norms, [1, 25, 50, 75, 99]),
                  [stored["elasticity_percentiles"][str(q)] for q in [1, 25, 50, 75, 99]], name + " column percentiles")
            summaries[name] = dict(summary, numerical_rank=stored["numerical_rank"],
                                   shape=list(J.shape), gram_check="12 probes" if fast else "complete")
    return summaries


def verify_e42_sources(root):
    directory = root / "results/revision_20261004"
    manifest = read_json(directory / "E42_source_manifest.json")
    provenance = read_json(directory / "E42_provenance.json")
    with zipfile.ZipFile(directory / "E42_model_and_runtime_sources.zip") as archive:
        for name, entry in manifest.items():
            value = archive.read(name)
            require(len(value) == entry["bytes"], "E42 retained source size mismatch: " + name)
            require(hashlib.sha256(value).hexdigest() == entry["sha256"], "E42 retained source digest mismatch: " + name)
        digests = {entry["sha256"] for entry in manifest.values()}
        missing = [name for name, digest in provenance["source_sha256"].items() if digest not in digests]
        require(not missing, "E42 provenance source digests absent from retained archive: " + "; ".join(missing))
    require(provenance["historical_result_overwritten"] is False, "Historical E42 overwrite flag changed")
    metadata = read_json(directory / "E42_runtime_metadata.json")
    require(metadata["n_connections"] == 3076 and metadata["negative_weights"] == 1341
            and metadata["positive_weights"] == 1735 and metadata["zero_weights"] == 0,
            "E42 signed-coordinate count metadata mismatch")
    require(metadata["negative_weights"] + metadata["positive_weights"] == metadata["n_connections"],
            "E42 signed-coordinate count sum mismatch")
    return {"retained_source_members": len(manifest), "provenance_digests": len(provenance["source_sha256"]),
            "signed_weight_metadata": {key: metadata[key] for key in ("negative_weights", "positive_weights", "zero_weights")}}


def verify_e42_fd(root):
    directory = root / "results/revision_20261004"
    d = read_json(directory / "E42_fd_validation.json")
    require(d["n_rollouts"] == 2 + 2 * len(d["samples"]) * len(d["steps"]), "E42 FD rollout count mismatch")
    report = {}
    with np.load(directory / "E42_motor_voltage_matrices.npz", allow_pickle=False) as main:
        with np.load(directory / "E42_fd_validation_arrays.npz", allow_pickle=False) as a:
            np.testing.assert_array_equal(a["repeat0"], main["baseline"])
            np.testing.assert_array_equal(a["repeat1"], main["baseline"])
            np.testing.assert_array_equal(a["output_scale"], main["output_scale"])
            for name, index in d["samples"].items():
                reference = a["derivative_" + name + "_0.05"]
                ref_norm = np.linalg.norm(reference)
                for entry in d["columns"][name]["steps"]:
                    step = f'{entry["delta"]:g}'
                    response = a[f"{name}_{step}_+1"] - a[f"{name}_{step}_-1"]
                    derivative = (response / (2 * entry["delta"]) / a["output_scale"]).ravel()
                    close(derivative, a[f"derivative_{name}_{step}"], name + " FD response reconstruction", rtol=2e-8)
                    close(np.linalg.norm(derivative), entry["normalized_derivative_norm"], name + " FD norm")
                    close(np.linalg.norm(response / a["output_scale"]), entry["normalized_symmetric_response_norm"], name + " symmetric response")
                    if ref_norm:
                        close(np.linalg.norm(derivative-reference) / ref_norm,
                              entry["relative_change_from_delta0_05"], name + " sampled convergence")
                    else:
                        require(entry["relative_change_from_delta0_05"] is None, name + ": unresolved reference derivative was assigned a convergence ratio")
                close(a[f"derivative_{name}_0.5"], main["J"][:, index], name + " main column match")
                report[name] = {"connection_index": index,
                                "relative_change_0.5_vs_0.05": d["columns"][name]["steps"][-1]["relative_change_from_delta0_05"]}
    return dict(n_rollouts=d["n_rollouts"], sampled_columns=report)


def verify_flyvis(root):
    directory = root / "results/revision_20261004"
    comparison = read_json(directory / "flyvis_state_coordinate_comparison.json")
    timestep = read_json(directory / "flyvis_timestep_control.json")
    stems = ["flyvis_trained_dimensionless", "flyvis_initialized_dimensionless",
             "flyvis_trained_dt001", "flyvis_trained_dt0005"]
    d = {stem: read_json(directory / (stem + ".json")) for stem in stems}
    arrays = {}
    report = {}
    try:
        for stem in stems:
            require(sha256(directory / d[stem]["arrays_file"]) == d[stem]["arrays_sha256"], stem + ": NPZ digest mismatch")
            arrays[stem] = np.load(directory / d[stem]["arrays_file"], allow_pickle=False)
        native = arrays[stems[0]]
        mask = native["interior_mask"]
        require(mask.shape == (734,) and mask.sum() == 703, "Flyvis shared interior subset mismatch")
        identities = [(entry["group"], entry["index"]) for entry in d[stems[0]]["coordinate_manifest"]]
        scales = native["output_scales"]
        for stem in stems:
            data, a = d[stem], arrays[stem]
            require([(e["group"], e["index"]) for e in data["coordinate_manifest"]] == identities, stem + ": coordinate ordering changed")
            require(data["parameters_restored_exactly"], stem + ": parameters not restored")
            own_mask = a["interior_mask"]
            expected_mask = np.array([not e["boundary"] for e in data["coordinate_manifest"]])
            np.testing.assert_array_equal(own_mask, expected_mask)
            close(a["output_scales"], np.abs(a["baseline"]) + 1e-6, stem + " own reference weights")
            repeats = a["repeated_baseline"] - a["baseline"][None, :]
            close(np.max(np.abs(repeats)), data["null_max_absolute"], stem + " repeat null", atol=1e-15)
            for step, grids in data["spectra_by_step"].items():
                raw = a["J_raw_step_" + step]
                standardized = a["J_scaled_step_" + step]
                close(standardized, raw / a["output_scales"][:, None], stem + " standardization")
                offset = 0
                for bins in (64, 32, 16, 8, 6):
                    block = standardized[offset:offset + bins]
                    for subset, selection in (("interior", own_mask), ("including_boundary", np.ones(734, dtype=bool)),
                                               ("bias_only", np.array([e["group"] == "nodes_bias" for e in data["coordinate_manifest"]]))):
                        svd_check(block[:, selection], grids[str(bins)][subset], stem + f" step={step} bins={bins} {subset}")
                    offset += bins
            common = a["J_raw_step_0.01"] / scales[:, None]
            if "dt" not in stem:
                state = data["model_state"]
                saved = comparison["states"][state]["common_weight_matched_703_readout_sweep"]
                for step, validation in comparison["states"][state]["common_weight_step_validation"].items():
                    candidate = (a["J_raw_step_" + step] / scales[:, None])[:64, mask]
                    reference = common[:64, mask]
                    difference = np.linalg.norm(candidate-reference) / np.linalg.norm(reference)
                    close(difference, validation["relative_frobenius"], stem + " parameter step agreement")
            else:
                dt = f'{data["measurement"]["dt"]:g}'
                saved = timestep["conditions"][dt]["common_weight_readout_sweep"]
                require(data["coordinate_manifest"] == d[stems[0]]["coordinate_manifest"], stem + ": integration control parameters differ")
                np.testing.assert_array_equal(own_mask, mask)
                repeat = int(round(.02 / data["measurement"]["dt"]))
                np.testing.assert_array_equal(a["movie"], np.repeat(native["movie"], repeat, axis=1))
            dt = f'{data["measurement"]["dt"]:g}'
            if data["model_state"] == "trained":
                entry = timestep["conditions"][dt]
                reference = (native["J_raw_step_0.01"] / scales[:, None])[:64, mask]
                difference = np.linalg.norm(common[:64, mask]-reference) / np.linalg.norm(reference)
                close(difference, entry["jacobian_difference_from_native"]["relative_frobenius"], stem + " integration step agreement")
                clipped = sum(e["group"] == "nodes_time_const" and e["baseline"] < data["measurement"]["dt"]
                              for e in data["coordinate_manifest"])
                require(clipped == entry["native_clipped_time_constant_count"], stem + ": time constant floor count mismatch")
            offset = 0
            for bins in (64, 32, 16, 8, 6):
                summary = svd_check(common[offset:offset + bins, mask], saved[str(bins)], stem + f" common W bins={bins}")
                if bins == 64:
                    report[stem] = summary
                offset += bins
        coarse = (arrays[stems[2]]["J_raw_step_0.01"] / scales[:, None])[:64, mask]
        fine = (arrays[stems[3]]["J_raw_step_0.01"] / scales[:, None])[:64, mask]
        close(np.linalg.norm(fine-coarse) / np.linalg.norm(coarse),
              timestep["comparison_005_vs_01"]["relative_frobenius"], "flyvis two fine integration steps")
        return report
    finally:
        for a in arrays.values():
            a.close()


def numeric_leaves(value, prefix=""):
    leaves = {}
    for name, child in value.items():
        path = prefix + name
        if isinstance(child, dict):
            leaves.update(numeric_leaves(child, path + "."))
        elif isinstance(child, (int, float)) and not isinstance(child, bool):
            leaves[path] = child
    return leaves


def verify_context_unions(root):
    directory = root / "results/revision_20261004"
    report = {}
    for folder in ("B6_full_union_rerun", "B6b_full_union_rerun"):
        location = directory / folder
        require(location.is_dir(), folder + ": required full union evidence is missing")
        require((location / "result.json").is_file(), folder + ": incomplete deposit has no result.json")
        d = read_json(location / "result.json")
        labels, names, delta = d["labels"], d["observable_names"], d["delta"]
        require(len(set(labels)) == len(labels), folder + ": duplicate parameter labels")
        require(d["parameter_coordinate"] == "log multiplicative gain", folder + ": unspecified coordinate")
        summaries = {}
        with np.load(location / d["matrix_file"], allow_pickle=False) as a:
            matrices = {}
            for condition in ("chemo", "loco"):
                baseline = read_json(location / (condition + "_base.json"))
                require(baseline["result"]["obs"] == d["baseline_observables"][condition], condition + ": baseline metadata mismatch")
                scales = np.abs([baseline["result"]["obs"][name] for name in names]) + 1e-9
                close(a["output_scale_" + condition], scales, folder + " baseline scales")
                columns = []
                for label in labels:
                    plus = read_json(location / f"{condition}_{label}_+1.json")
                    minus = read_json(location / f"{condition}_{label}_-1.json")
                    plus_leaves, minus_leaves = numeric_leaves(plus["spec"]), numeric_leaves(minus["spec"])
                    require(plus_leaves.keys() == minus_leaves.keys(), folder + ": positive/negative parameter specs differ")
                    changes = [key for key in plus_leaves if plus_leaves[key] != minus_leaves[key]]
                    require(len(changes) == 1, folder + ": perturbation changes more than one numeric coordinate")
                    key = changes[0]
                    close(plus_leaves[key], np.exp(delta), folder + " positive log step")
                    close(minus_leaves[key], np.exp(-delta), folder + " negative log step")
                    require(plus["spec"]["food_xyz"] == minus["spec"]["food_xyz"] == baseline["spec"]["food_xyz"], folder + ": stimulus differs within perturbation pair")
                    require(plus["spec"].get("cells") == minus["spec"].get("cells"), folder + ": class targeting differs within pair")
                    if folder.startswith("B6b"):
                        require(plus["spec"].get("cells") == [label] and key == "passive.gpas", folder + ": neuron-class label/target mismatch")
                    else:
                        expected = label if label in ("syn", "gj", "wout") else "passive.gpas" if label == "gpas" else "ion." + label
                        require(key == expected, folder + ": global-gain label/target mismatch")
                    yp = np.array([plus["result"]["obs"][name] for name in names])
                    ym = np.array([minus["result"]["obs"][name] for name in names])
                    columns.append((yp - ym) / (2 * delta) / scales)
                J = np.stack(columns, axis=1)
                close(J, a["J_" + condition], folder + " response-derived " + condition + " Jacobian")
                close(J.T @ J, a["G_" + condition], folder + " complete " + condition + " Gram")
                matrices[condition] = J
            combined = np.vstack((matrices["chemo"], matrices["loco"]))
            close(combined, a["J_union"], folder + " stacked union Jacobian")
            close(a["G_chemo"] + a["G_loco"], a["G_union"], folder + " sum of condition Grams")
            for condition, J in {**matrices, "union": combined}.items():
                G = a["G_" + condition]
                close(G, J.T @ J, folder + " complete " + condition + " Gram")
                saved = d["summary"][condition]
                ev = np.linalg.eigvalsh(G)[::-1]
                require(len(saved["eigenvalues"]) == len(labels), folder + ": truncated context spectrum")
                close(ev, saved["eigenvalues"], folder + " complete context spectrum", atol=max(ev[0] * 1e-12, 1e-12))
                singular = np.linalg.svd(J, compute_uv=False)
                close(singular, saved["singular_values"], folder + " context SVD", atol=max(singular[0] * 1e-13, 1e-13))
                tolerance = np.finfo(float).eps * max(J.shape) * singular[0]
                close(tolerance, saved["singular_value_rank_tolerance"], folder + " algebraic rank tolerance", atol=0)
                require(int(np.count_nonzero(singular > tolerance)) == saved["numerical_rank"], folder + ": floating-point SVD rank mismatch")
                relative_ranks = {}
                for cutoff, expected in saved["rank_at_relative_singular_tolerance"].items():
                    rank = int(np.count_nonzero(singular > float(cutoff) * singular[0]))
                    require(rank == expected, folder + ": relative-tolerance rank mismatch")
                    relative_ranks[cutoff] = rank
                mass = mass_summary(ev)
                for key in ("eff_dim_90", "eff_dim_99"):
                    require(mass[key] == saved[key], folder + ": context concentration count mismatch")
                summaries[condition] = dict(mass, floating_point_rank=saved["numerical_rank"], relative_tolerance_ranks=relative_ranks)
                if condition == "union":
                    summaries[condition]["mass_beyond_rank_four"] = float(np.maximum(ev[4:], 0).sum() / np.maximum(ev, 0).sum())
            bases = [np.linalg.eigh(a["G_" + condition])[1][:, -3:] for condition in ("chemo", "loco")]
            cosines = np.linalg.svd(bases[0].T @ bases[1], compute_uv=False)
            overlap = float(np.sqrt(np.mean(cosines ** 2)))
            close(overlap, d["stiff_subspace_overlap_top3_cos"], folder + " leading-three RMS principal-angle cosine")
            report[folder] = {"status": "PASS", "parameter_labels": labels,
                              "raw_rollouts": 2 + 4 * len(labels), "summary": summaries,
                              "leading_three_rms_cosine": overlap,
                              "leading_three_principal_angle_cosines": cosines.tolist()}
    return report


def verify_context_sources(root):
    directory = root / "results/revision_20261004"
    manifest = read_json(directory / "B6_source_manifest.json")
    provenance = read_json(directory / "B6_model_runtime_provenance.json")
    digests = set()
    with zipfile.ZipFile(directory / "B6_model_and_runtime_sources.zip") as archive:
        for name, entry in manifest.items():
            value = archive.read(name)
            digest = hashlib.sha256(value).hexdigest()
            require(len(value) == entry["bytes"] and digest == entry["sha256"],
                    "B6 retained source mismatch: " + name)
            digests.add(digest)
    for name, entry in provenance["model_files"].items():
        require(entry["sha256"] in digests, "B6 runtime provenance source not retained: " + name)
    for folder in ("B6_full_union_rerun", "B6b_full_union_rerun"):
        location = directory / folder
        if location.exists():
            result = read_json(location / "result.json")
            for key in ("producer_sha256", "worker_sha256"):
                require(result[key] in digests, folder + ": executed code digest not retained: " + key)
    return {"retained_source_members": len(manifest), "model_provenance_files": len(provenance["model_files"])}


def verify_e45(root):
    directory = root / "results/revision_20261004"
    d = read_json(directory / "E45_named_channels_rerun.json")
    historical = read_json(root / "results/E45b_baai_named_channels.json")
    participation = read_json(root / "results/AB7_baai_participation.json")
    with np.load(directory / "E45_named_channels_matrices.npz", allow_pickle=False) as a:
        J, G = a["J"], a["G"]
        close(J.T @ J, G, "E45 complete Gram")
        ev = np.maximum(np.linalg.eigvalsh(G)[::-1], 0)
        summary = check_summary(ev, d, "E45 rerun")
        for key in ("eff_dim_90", "eff_dim_99", "n_observables"):
            require(d[key] == historical[key], "E45 historical scalar mismatch: " + key)
        close(d["participation_ratio"], historical["participation_ratio"], "E45 historical PR")
        close(ev, [entry["eigenvalue"] for entry in participation["per_eigenvector"]], "E45 AB7 full eigenvalue parity", atol=1e-8)
        close(sorted(np.linalg.norm(J, axis=0), reverse=True),
              [entry[1] for entry in d["ranking_stiff_to_sloppy"]], "E45 ranked column norms")
        return dict(summary, shape=list(J.shape), historical_scalar_and_spectrum_parity="PASS")


def artifact_paths(root):
    files = [f"paper/figures/{name}.{extension}" for name in FIGURES for extension in ("pdf", "png")]
    files += ["scripts/redraw_output_sensitivity.py", "scripts/redraw_visual_sensitivity.py",
              "results/revision_20261004/figure_sources.json"]
    # Include all plotted sources and the separately produced visual-network source.
    sources = read_json(root / "results/revision_20261004/figure_sources.json")
    files += ["results/" + source for source in sources]
    files += ["results/revision_20261004/flyvis_state_coordinate_comparison.json"]
    files += ["results/revision_20261004/E42_source_manifest.json",
              "results/revision_20261004/E42_model_and_runtime_sources.zip",
              "results/revision_20261004/E42_provenance.json"]
    files += ["results/revision_20261004/E42_runtime_metadata.json",
              "results/revision_20261004/B6_source_manifest.json",
              "results/revision_20261004/B6_model_and_runtime_sources.zip",
              "results/revision_20261004/B6_model_runtime_provenance.json"]
    for folder in ("B6_full_union_rerun", "B6b_full_union_rerun"):
        location = root / "results/revision_20261004" / folder
        if location.exists():
            files += [path.relative_to(root).as_posix() for path in location.glob("*.json")]
            files += [path.relative_to(root).as_posix() for path in location.glob("*.npz")]
    return sorted(set(files))


def verify_hashes(root):
    sources = read_json(root / "results/revision_20261004/figure_sources.json")
    for source, expected in sources.items():
        require(sha256(root / "results" / source) == expected, "Figure input digest mismatch: " + source)
    manifest = read_json(root / MANIFEST)
    require(set(manifest["sha256"]) == set(artifact_paths(root)), "Artifact manifest scope changed; rebuild then reseal explicitly")
    for path, expected in manifest["sha256"].items():
        require(sha256(root / path) == expected, "Artifact digest mismatch: " + path)
    return {"figure_input_digests": len(sources), "sealed_artifact_digests": len(manifest["sha256"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fast", action="store_true", help="Use 12 deterministic E42 Gram probes; omit large eigendecomposition")
    parser.add_argument("--seal-artifacts", action="store_true", help="Record current figures/scripts/sources byte identities, separately from verification")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.seal_artifacts:
        path = root / MANIFEST
        manifest = {"description": "Byte identity after reviewed figure build; numerical checks are independent",
                    "sha256": {name: sha256(root / name) for name in artifact_paths(root)}}
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("Recorded artifact identities: " + str(path))
        return 0
    report = {"scope": "deposited matrix algebra and figure/source byte identities; no forward simulation",
              "e42_mode": "12 deterministic probes" if args.fast else "complete Gram and eigendecomposition",
              "checks": {}, "failures": []}
    started = time.monotonic()
    for label, action in (("E42", lambda: verify_e42(root, args.fast)),
                          ("E42_sources", lambda: verify_e42_sources(root)),
                          ("E42_sampled_FD", lambda: verify_e42_fd(root)),
                          ("E45", lambda: verify_e45(root)),
                          ("context_unions", lambda: verify_context_unions(root)),
                          ("context_sources", lambda: verify_context_sources(root)),
                          ("flyvis", lambda: verify_flyvis(root)),
                          ("hashes", lambda: verify_hashes(root))):
        try:
            report["checks"][label] = action()
            print(label + ": PASS", flush=True)
        except Exception as exc:
            report["failures"].append({"check": label, "error": str(exc)})
            print(label + ": FAIL: " + str(exc), flush=True)
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    report["elapsed_seconds"] = time.monotonic() - started
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    sys.exit(main())
