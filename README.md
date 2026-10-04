# Concentrated output sensitivity in connectome-scale biological models

Analysis and simulation code, model inputs and results for Chenxi He's study of output sensitivity, measurement choice and parameter recovery. The current manuscript and supplementary information are in [paper/](paper/).

## Free, anonymous access

All repository files and [release assets](https://github.com/ChenxiHeCam/behavioural-parameter-manifold/releases/tag/paper2-code-20261004) are public. Browse directly, select **Code → Download ZIP**, or clone:

```bash
git clone https://github.com/ChenxiHeCam/behavioural-parameter-manifold.git
cd behavioural-parameter-manifold
```

No account, login, password or personal information is required. Tag `paper2-code-20261004` identifies this version. Large matrices and raw response archives are release downloads; [release_assets.json](repro/release_assets.json) specifies their URLs, paths, sizes and SHA256 values.

## Recalculate stored results and figures

Use Python 3.11 or a compatible environment and run from the repository root:

```bash
python -m pip install -r repro/requirements.txt
python repro/download_release_assets.py
python repro/verify_revision.py --fast
python scripts/redraw_output_sensitivity.py
python scripts/redraw_visual_sensitivity.py
python scripts/prospective/plot_formal.py
```

The fast verifier uses deterministic Gram-matrix probes for the large connection assay; omit `--fast` to recompute its complete Gram matrices and eigenspectra. These checks operate on retained outputs and require no simulator or GPU. See [repro/README.md](repro/README.md).

## Code and computational models

| Analysis | Code under scripts/ | Model inputs |
|---|---|---|
| Nematode connection perturbations | `_e42_recover_full_spectrum.py`, `_baai_worker.py`, `_baai_worker_support.py` | E42 model/runtime archive and NEURON mechanism sources |
| Named conductance gains | `_e45_baai_named_channels.py` | Same point-neuron model, with explicit mechanism addressing |
| Combined stimulation conditions | `_b6_union_crossbehaviour.py`, `_b6b_cellclass_union.py`, `_b6_common.py`, `_b6_physics_worker.py` | B6 model/runtime archive including headless physics sources |
| Fly visual model states | `flyvis_sensitivity_revision.py` | flyvis 1.1.2, retained checkpoint and network metadata |
| Cell-model recovery | `cell_panel.py`, `cell_panel3.py`, `cell_ladder.py`, `cell_scale.py` | Self-contained ODEs and generating parameters |
| Prospective stimulus comparison | `prospective/run_formal.py`, `nonlinear_engine.py`, `persistent_backend.py`, `run_pilot.py` | Point-neuron bundle and locked experimental specification |

Results are in `results/revision_20261004/` and `results/prospective_20261004/`. [Model setup](repro/MODEL_SETUP.md), [result mapping](scripts/RESULTS_MANIFEST.md) and [figure mapping](scripts/FIGURE_MANIFEST.md) describe the dependencies and input files.

For original raw responses and the full prospective response check:

```bash
python repro/download_release_assets.py --raw --extract-prospective
python scripts/prospective/verify_formal.py
```

Forward simulations require the corresponding model environments. Stored-result verification and figure generation use the supplied arrays and summaries.

## Previous versions and licenses

Earlier commits and the [previous Zenodo deposit](https://doi.org/10.5281/zenodo.22120856) retain the earlier study version. `repro/regenerate_all.py` checks 72 historical fixtures; it does not verify every current manuscript claim. The current comparisons include a prospective negative result: ensemble selection did not demonstrate an average prediction benefit.

Study code is covered by [LICENSE](LICENSE). Third-party models retain their upstream licenses; see [third_party_licenses/README.md](third_party_licenses/README.md). Cover letters, submission forms and private editorial records are excluded.
