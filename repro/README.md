# Reproduction instructions

Run from the repository root. Stored-result analysis uses NumPy, SciPy and Matplotlib; forward simulation has separate model dependencies.

```bash
python -m pip install -r repro/requirements.txt
python repro/download_release_assets.py
python repro/verify_revision.py --fast --json-out repro/public_verification.json
```

The report distinguishes deterministic probes of the large connection Gram matrices from full calculations. It also checks stored spectra, matched visual-network measurements, complete combined-condition matrices, model/source archives and artifact hashes. Omit `--fast` for full connection Gram matrices and eigenspectra. Neither mode reruns simulators.

## Figures

```bash
python scripts/redraw_output_sensitivity.py
python scripts/redraw_visual_sensitivity.py
python scripts/prospective/plot_formal.py
```

The prospective plot is written under `results/prospective_20261004/figures/`; manuscript figures are provided under `paper/figures/`. Plots read retained data directly.

## Raw-response and focused code checks

```bash
python repro/download_release_assets.py --raw --extract-prospective
python scripts/prospective/verify_formal.py
python repro/test_baai_worker_dispatch.py
python scripts/test_flyvis_sensitivity_revision.py
```

The prospective verifier reconstructs observations, fit objectives, compatible ensembles, selection scores and endpoints. The statistical replicates are the 12 generating parameter cases. Focused tests check conductance addressing, readout definitions and coordinate/derivative handling.

## Docker

Download the result matrices first, then:

```bash
docker build -f repro/Dockerfile -t paper2-reproduction .
docker run --rm paper2-reproduction
```

The image performs the stored-result checks, including the full connection calculation. Raw archives are unnecessary for this check.

## Historical entry points

`regenerate_all.py` checks 72 historical fixtures. `check_manuscript.py` uses earlier Nature Communications limits, which are not current journal requirements. `check_style.py` is an optional contextual prose checker. Use `verify_revision.py` for the current result-level checks.

Sealing artifacts records byte identities; it does not establish numerical correctness. Regenerating PDFs can change byte hashes while preserving plotted arrays. Model environments and new output directories are described in [MODEL_SETUP.md](MODEL_SETUP.md).
