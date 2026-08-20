# Which script owns which figure

Several scripts in this directory can write the same output file, because later
figure revisions were made in new scripts without retiring the old ones. On
2026-08-20 rerunning `_redraw_remaining.py` for a one-line fix silently replaced
`fig_p2_b1_perchannel.png` and `Fig_connectome_curvature.png` with obsolete
versions that no longer matched their captions. **Regenerate figures only with
the owner listed here.** If you must run a non-owner script, rerun the owners
afterwards.

| figure (paper/figures/) | owner script | note |
|---|---|---|
| Fig1_concept.png | (external: generated image, prompt in chat) | replace when ≥2400 px version exists |
| Fig_connectome_curvature.png | `_redraw_fig2_three.py` | 3-panel; `_redraw_new_results.py` writes an obsolete 2-panel version |
| fig_p2_tiling_crossspecies.png | `_redraw_tiling.py` | |
| fig_p2_b1_perchannel.png | `_redraw_new_results.py` | BAAIWorm named conductances (IRK 123.5 top, SLO omitted); `_redraw_figures.py` and `_redraw_remaining.py` write obsolete versions |
| fig_p2_flygym_hessian48.png | `_redraw_new_results.py` | |
| Fig_biological_datasets.png | `_redraw_remaining.py` | |
| Fig_multistart_three_sim.png | `_redraw_remaining.py` | |
| fig_p2_b2_neural.png | `_fix_si_figs.py` | |
| fig_p2_b3_localise.png | `_fix_si_figs.py` | |
| fig_p2_crosssystem_real.png | `_fix_si_figs.py` | |
| fig_p2_flyvis_b1.png | `_fix_si_figs.py` | `_redraw_figures.py` writes an obsolete version |
| fig_p2_cell_groundtruth.png | `_fig_cell_groundtruth.py` | |
| fig_p2_cell_scale.png | `_fig_cell_scale.py` | |
| fig_p2_modworm_real_b1.png | `_redraw_figures.py` | |
| e1b_hessian.png, Fig_E*.png, Fig_crosscheck.png | archival, no active owner | do not regenerate |

Safe full rebuild order (later overwrites earlier where they collide):

    _redraw_figures.py
    _redraw_remaining.py
    _redraw_new_results.py
    _redraw_fig2_three.py
    _redraw_tiling.py
    _fix_si_figs.py
    _fig_cell_groundtruth.py
