# Which script produced which result file

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
