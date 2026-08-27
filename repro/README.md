# One-command reproduction

Reproduces every quantitative claim and the cross-species hero figure from the
deposited result files. The forward-model simulations (BAAIWorm, modWorm,
Virtual Rodent, flyvis, the Prinz–Marder STG population) are
deposited as result JSONs; this bundle reproduces the **analysis layer** on top of them — effective dimension, participation ratio, complementary tiling, union
growth, the two-model agreement, and the figure — with no simulator, GPU, or
network access required.

## Run with Docker (no local Python needed)

From the repository root:

```bash
docker build -f repro/Dockerfile -t repro .
docker run --rm repro
```

## Run directly

```bash
pip install -r repro/requirements.txt
python repro/regenerate_all.py          # reads ../results/ by default
# or point it at the JSONs explicitly:
RESULTS_DIR=/path/to/jsons python repro/regenerate_all.py
```

## Expected output

- `=== NUMBER AUDIT: 72/72 PASS ===` — every number cited in the main text and
  SI is recomputed from its source JSON and checked (non-zero exit on any FAIL).
- The S6.9 threshold-robustness table (effective dimension at 80/90/95/99 % of
  spectral mass + participation ratio), recomputed from the saved eigenspectra.
- `fig_p2_tiling_crossspecies_CHECK.png` — the four-panel hero figure
    regenerated from the JSONs. (The publication-quality version is produced by
  `_fig_nc_tiling.py`; the check version verifies the data, not the styling.)

## Checking the manuscript

`regenerate_all.py` verifies the numbers. Two further scripts verify the document:

```bash
python repro/check_manuscript.py
```

checks the journal's Article limits (abstract, main text, display items, references,
title, subheading placement) and the manuscript's internal consistency: that every
Supplementary cross-reference resolves, that every result file cited in either document
exists in `results/`, and that every figure clears 300 dpi at its printed width. Exit
status is non-zero on any failure.

```bash
python repro/check_style.py
```

flags constructions the manuscript avoids: em-dashes in prose, hedging formulas,
changelog phrasing, bullet lists, and infrastructure detail. Not every hit is a fault;
em-dashes in section headings and in table cells marking a withdrawn row are legitimate.

## What maps to what

| Claim | Source JSON |
|---|---|
| eigenworm identity (3.79/4.14; cos 0.77/0.68) | `EW_eigenworm.json`, `EW_baai_local.json` |
| complementary tiling (7.15× context shift) | `BAAI_chemo.json` |
| union saturation (worm 4/7) | `SAT_saturation.json`, `MB_behaviour_specificity.json` |
| cross-species union (larva 8, rodent 12) | `LARVA_manifold.json`, `RODENT_manifold.json` |
| two-model agreement (wiring 0.83/0.87) | `G2_crosssim_align.json` |
| coupling (participation 3.42/4.68) | `CP_coupling.json` |
| robustness (D1 multipoint, D3 stats) | `mw_D1R6.json`, `D3_modworm_stats.json` |

External-reference datasets (Allen, WormBase/Ensembl, OWMD, Prinz–Marder) are
public; their provenance and access are listed in the Supplementary Information.
