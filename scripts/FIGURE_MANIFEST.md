# Which script owns which figure

## Current revision, 2026-10-04

Only these figures are included in the revised manuscript. Their PNG/PDF/SVG versions share the same plotted data and printed width.

| Main/SI figure | Basename | Owner | Evidence |
|---|---|---|---|
| Main 1 | output_concentration | redraw_output_sensitivity.py | Overview panel a with recorded E42 baseline voltage; unchanged full E42 spectra and class contributions, AB7 and E58 in b-e |
| Main 2 | measurement_dependence | redraw_output_sensitivity.py | AB4 matched steps/readouts; E1 output metadata |
| Main 3 | Fig_visual_sensitivity | redraw_visual_sensitivity.py | Flyvis common-weight, shared-coordinate comparison and step controls |
| Main 4 | context_dependence | redraw_output_sensitivity.py | Closed-loop contexts, full reconstructed joint spectrum, FlyGym subset curves |
| Main 5 | sensitivity_and_recovery | redraw_output_sensitivity.py | Corrected cell panels, HH assay grid, MAPK traces and verified time metadata |
| SI 1 | cell_network_scaling | redraw_output_sensitivity.py | Coupled HH family, including the single-cell control |
| SI 2 | stimulus_comparison | prospective/plot_formal.py | All 12 paired prospective cases and 48 outcomes; mean endpoints, paired ensemble-minus-control differences and independent-case bootstrap intervals; failure retained |

`output_sensitivity_style.py` defines shared appearance. `results/revision_20261004/figure_sources.json` records data-source hashes. Numerical rank and spectral-mass counts have different meanings. No incomplete spectrum is extended synthetically.




Earlier figures and their scripts remain in repository history. Use the current owners listed above to regenerate manuscript figures.
