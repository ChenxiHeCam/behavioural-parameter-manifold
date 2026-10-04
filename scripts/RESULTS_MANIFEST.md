# Which script produced which result file

## Current reconstruction, 2026-10-04

Files below are under `results/revision_20261004/` unless specified. Historical results remain unchanged.

| Evidence | Files | Producer / validation |
|---|---|---|
| Full connectome, paired outputs | E42_full_spectrum_rerun.json; three E42_*_matrices.npz | _e42_recover_full_spectrum.py; _baai_worker.py; source/runtime archive and E42_provenance.json |
| Historical precision discrepancy | E42_precision_diagnostic.json | _e42_precision_diagnostic.py; 8/47 is not reproduced by new responses |
| Correct named mechanisms | E45_named_channels_rerun.json; E45_named_channels_matrices.npz; BAAI_worker_smoke.json | _e45_baai_named_channels.py; _e45_worker_smoke.py; explicit dispatch tests |
| Full context union | B6/B6b reconstruction directories and matrices | _b6_union_crossbehaviour.py / _b6b_cellclass_union.py; _b6_common.py; headless physics adapter |
| Explicit fly visual states | flyvis_trained_dimensionless.json/.npz; flyvis_initialized_dimensionless.json/.npz | flyvis_sensitivity_revision.py; retained checkpoint and coordinate/readout manifests |
| Matched fly visual comparisons | flyvis_state_coordinate_comparison.json | summarize_flyvis_revision.py; common trained-reference output scales and shared 703 indices |
| Integration controls | flyvis_trained_dt001 and dt0005 JSON/NPZ/producers; flyvis_timestep_control.json | Executed retained producers; summarize_flyvis_timestep_control.py; all-condition verifier |
| MAPK time axis | mapk_trace_metadata.json | prepare_revision_context.py; generating trace reproduced exactly before plotting |
| Historical animal hypotheses | two_connectome_consistency.json; leifer_class_test.json | Exact copied external results with source hashes; not tests of revised model ranking |

Current result-level verification: `repro/verify_revision.py`. Historical 72-fixture regression: `repro/regenerate_all.py`. These do not replace forward-simulation controls.

## Prospective comparison, completed 2026-10-04

Files are under `results/prospective_20261004/`; see its README for complete archive and verification pointers. `run_formal.py`, `nonlinear_engine.py` and `persistent_backend.py` produced 12 independent cases and 48 outcomes. `formal_results_20261004/raw/` retains 24,236 response/request pairs. `verify_formal.py` recomputes observations, fit objectives, compatible ensembles, scores and endpoints; `check_formal_fresh_parity.py` adds six independent-process controls. `summarize_formal.py` uses paired independent-case bootstrap intervals. All training-quality failures remain in summaries. The prior exploratory pilot and development diagnostics are separate. Runtime identity follows the locked snapshot plus the preserved `run_pilot.py` dependency supplement.

## Archived result mapping

The entries below describe historical results. A current producer may contain repairs absent from its historical execution; use the retained current source/runtime manifests for reconstructed runs.

Analysis scripts here fall into two classes. Scripts that read only the deposited
JSONs rerun anywhere (the audit and figure scripts). Scripts that produced the
result files themselves ran against a simulator installation on compute nodes;
they are deposited for inspection and record the exact settings, but rerunning
them requires the named simulator (see the README's simulator list). Paths of the
form `/root/...` in those scripts are the compute-node layout, kept as run.

Result files carrying main-text claims:

| result file | producer script | needs |
|---|---|---|
| E42_baaiworm_connectome.json | _e42_baai_connectome.py + _baai_worker.py | BAAIWorm |
| E58_baai_multipoint.json | _e58_baai_multipoint.py | BAAIWorm |
| E60_baai_muscle_full.json | _e60_muscle_full.py | BAAIWorm |
| E59_baai_muscle_observable.json | _e59_muscle_observable.py | BAAIWorm |
| E45b_baai_named_channels.json | _e45b_named_channels fix in _e45_baai_named_channels.py | BAAIWorm |
| AB7_baai_participation.json | _ab7_baai_participation.py | BAAIWorm |
| B6_union_crossbehaviour.json | _b6_union_crossbehaviour.py + _baai_perturb_worker_orig.py | BAAIWorm |
| B6b_cellclass_union.json | _b6b_cellclass_union.py (its per-cell worker was not retained) | BAAIWorm |
| BAAI_chemo.json | _baai_chemo.py (its per-cell worker was not retained) | BAAIWorm |
| E4_flyvis_readout.json | _e4_flyvis_readout.py | flyvis |
| AB4_flygym_delta.json | _ab4_flygym_delta.py | FlyGym |
| E36b_flygym_validstep.json | _e36b_flygym_validstep.py | FlyGym |
| SAT_saturation.json / MB_behaviour_specificity.json | _sat_saturation.py / _mb_behaviour_specificity.py | modWorm |
| D3_modworm_stats.json | statistics assembled from the modWorm probe runs; assembly script not retained | — |
| LARVA_noise_floor.json | _larva_noise_floor.py | larvaworld |
| EW_eigenworm.json | _ew_eigenworm.py | modWorm + OWMD |
| cell_* (eleven-model panel, ladder, scale) | cell_panel.py, cell_panel3.py, cell_ladder.py, cell_scale.py | none (self-contained ODEs) |
| E1_pyloric_manifold.json | PCA over the published Prinz–Marder population; assembly script not retained | population data |
| E2_allen_variability.json | _e2_allen_variability.py | Allen table (not deposited; public) |
| E4_channel_conservation.json | Ensembl REST query; assembly script not retained | network access |
| E5_n2_manifold.json | OWMD Tierpsy aggregation; assembly script not retained | OWMD files |
| null_subspace_overlap.json | _null_subspace_overlap.py | none |

Files marked "not retained" predate this release's provenance rule; their
construction is specified in the Supplementary section that cites them.

Run-time output names that differ from the deposited names (the deposited file is
the run's output, renamed on deposit):

| script writes | deposited as |
|---|---|
| E33_stg_noise.json | E33_stg_fd_convergence.json |
| E44_baai_controls.json | E44_baai_fd_controls.json |
| E37_granularity.json | E37_granularity_modworm.json |
