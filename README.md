# How much of a nervous-system model does behaviour identify?

Code, result files and reproduction container for the manuscript:

> Chenxi He. *How much of a nervous-system model does behaviour identify?* (2026)

Archived at Zenodo: [https://doi.org/10.5281/zenodo.21594739](https://doi.org/10.5281/zenodo.21594739)

Whole-organism biophysical simulators reproduce animal behaviour from hundreds to thousands of
internal parameters. This work measures how many of those parameters behaviour actually constrains,
using a single Gauss-Newton curvature estimator applied to every system, and reports each figure
alongside the checks that make it meaningful: whether the observables outnumber the parameters,
whether the response is linear in the perturbation, and what the numerical noise floor is.

## Key findings

- **Eight directions of three thousand.** In the connectome-scale nematode model (BAAIWorm, 136
  cells, 3076 synaptic and gap-junction weights), behaviour constrains 8 directions at 90% of the
  curvature and 47 at 99%, participation ratio 2.55. Only 4 weights have no measurable effect, so
  the flat directions are degeneracy rather than unconnected parameters. Observables (9600)
  outnumber parameters, the response is proportional to the step to four significant figures over a
  twentyfold range, and repeat simulations are bit-identical, so the noise floor is zero.
- **A second connectome model agrees.** The flyvis *Drosophila* optic-lobe model (734 free
  parameters), probed across all three parameter groups and read through its visual response,
  gives effective dimension 14 (90%) / 32 (99%) of 330 sampled parameters. It is also the best
  conditioned model here: response proportional to the step across two decades, ratio 0.498.
- **The curvature is not spread evenly over the connectome.** Chemical synapses carry 87% of it from
  1992 connections, gap junctions 13% from 1084: per connection, a chemical synapse matters about
  3.6 times more, though the median elasticities differ by only 1.20x, so the split is carried by a
  heavy tail of strong synapses rather than by the typical connection. Per-connection elasticities span
  seven orders of magnitude.
- **What behaviour barely sees.** Over the model's named
  conductances, read through the full motor trajectory: effective dimension 3 (90%) / 6 (99%) over all
  sixteen named conductances, none of which is identically zero. The stiffest are IRK and the sodium
  leak NCA; the fast voltage-gated calcium channels EGL-19 (1.77) and UNC-2 (0.17) are among the
  sloppiest, together with SLO-2 (0.57-0.78) and SLO-1/EGL-19 (0.09), while the other
  calcium-activated potassium conductances KCNL (15.1) and SLO-1/UNC-2 (6.68) rank mid-table.
- **Behavioural richness helps, with diminishing returns.** Adding behaviours enlarges the
  identifiable subspace monotonically (nematode 2.50 to 3.00 of 7; fly 16.5 to 19.0 of 48) but
  saturates by the third or fourth behaviour. Four fly behaviours give 19 dimensions rather than
  four times the 16.5 that one gives, so behaviours overlap far more than they complement.
- **Ground truth verifies the geometry where it can be known.** Whole-organism models have no true
  parameter vector; eleven cell models do. Recovered from random starts drawn independently of the
  truth, the 2-4-parameter neuronal models come back exactly (median error below 1e-4), while the
  canonical MAPK cascade matches behaviour to 1e-3 with its parameters 40% from the truth, farther
  than they started. A perfect behavioural fit does not imply recovered parameters.
- **The low dimension is a property of the region, not of one point.** Displacing the entire 3076-
  weight vector to four new operating points leaves the effective dimension at 8-12, and the stiff
  subspaces coincide across points at principal cosine 0.89 against a random null of 0.145.
- **Effective dimension is not recoverable dimension.** On a parameter-count x observable grid over
  one neuron model the 90% effective dimension is 1 in every cell while the achieved recovery error
  spans 4% to 56%: it measures how concentrated the curvature is, not how many parameters a fit can
  pin down.
- **An effective dimension is a statement about the measurement.** A Gauss-Newton Hessian built from
  n observables has rank at most n. The same FlyGym controller reads as 1 of 48 dimensions through
  ten summary statistics and 16 of 48 through 4200 joint-angle observables, against a measured noise
  floor of 7.
- **The gates retire one of our own headline numbers.** Two widely used simulators fail the
  differentiability test. modWorm has a discontinuous response to its own parameters, with two of
  seven mechanism classes producing no change at machine precision. In the Prinz-Marder pyloric
  circuit any perturbation decorrelates the trajectory, so the response is flat in the step across a
  250-fold range and the Jacobian scales as 1/step; the phase-invariant summary statistics adopted in
  that literature turn out to saturate almost as strongly (median response-to-step ratio varying by
  23x across a 25-fold step range, against 1 for a proportional response, under two whitenings). The
  pyloric effective dimension is therefore withdrawn, and the circuit is reported through its
  enumerated population instead: the 2365 conductance sets that produce the same rhythm span 23.2 of
  31 dimensions, which needs no finite difference at all.

## Quick reproduction

Every quantity reported in the paper is recomputed from the deposited result files and checked:

```bash
# with Docker (no local Python needed)
docker build -f repro/Dockerfile -t repro .
docker run --rm repro

# or directly
pip install -r repro/requirements.txt
python repro/regenerate_all.py
```

Expected output: `=== NUMBER AUDIT: 46/46 PASS ===`, the threshold-robustness table, and the
regenerated figure. Exit status is non-zero if any number fails to reproduce.

## Contents

| Path | Description |
|---|---|
| `paper/` | Manuscript and Supplementary Information (LaTeX + PDF), bibliography, and the nine figures |
| `results/` | Result files (JSON) for every analysis; each Supplementary section names its file |
| `scripts/` | Analysis and experiment code (Hessian/manifold probes, eigenworm, tiling, cross-species, external references, figure generation) |
| `data/` | Input data used by the analyses: *C. elegans* c302 connectome matrices, muscle map, OpenWorm connectivity and the Prinz–Marder valid pyloric parameter set |
| `repro/` | One-command reproduction container and the number-audit script |

## Data sources

All external data are public and are redistributed here only where licences permit; otherwise the
retrieval scripts are provided.

- **Prinz–Marder pyloric model population** — Prinz, Bucher & Marder, *Nat. Neurosci.* **7**, 1345 (2004).
- **Allen Cell Types** electrophysiology — https://celltypes.brain-map.org
- **WormBase ParaSite / Ensembl Metazoa** ortholog identities — https://parasite.wormbase.org
- **OpenWorm Movement Database** (Tierpsy features, N2 and mutant strains) — http://movement.openworm.org

Simulators are the published releases of BAAIWorm, modWorm, flybody, NeuroMechFly v2 / FlyGym,
flyvis, larvaworld, and the Virtual Rodent of the `dm_control` suite; see the manuscript for
citations. Scripts that must run inside a simulator's own environment use that environment's paths
and are provided for reference rather than as a turnkey pipeline.

## Notes

- Scripts prefixed `_` were run on compute nodes with the relevant simulator installed; their
  absolute paths reflect those environments.
- The number audit in `repro/` is the authoritative check: it re-derives each reported quantity
  from its source file rather than from any cached value.

## Citation

See `CITATION.cff`.

## Licence

Code and result files: MIT (see `LICENSE`). The manuscript text and figures are © the author.
Third-party datasets retain their original licences and terms.
