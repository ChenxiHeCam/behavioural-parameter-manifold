# Behaviour identifies a low-dimensional parameter subspace tiled by the behavioural repertoire

Code, result files and reproduction container for the manuscript:

> Chenxi He. *Behaviour identifies a low-dimensional parameter subspace tiled by the behavioural
> repertoire.* (2026)

Whole-organism biophysical simulators reproduce animal behaviour from hundreds to thousands of
internal parameters. This work measures the **behavioural-equivalence manifold** — the set of
parameter configurations that yield indistinguishable behaviour — across simulators spanning three
phyla, and shows that behaviour constrains a low-dimensional stiff subspace whose extent is set by
the richness of the behavioural repertoire.

## Key findings

- **Low-dimensional manifold.** Behaviour constrains 2–11 effective directions at every parameter
  scale examined (5, 7, 20, 30, 48, 767 parameters).
- **Identity.** In the nematode the manifold is the neural substrate of the animal's
  ~4-dimensional eigenworm postural space; two independently built models occupy the empirically
  measured basis (cos 0.77 / 0.68, *p* < 10⁻⁴ against random subspaces).
- **Complementary tiling.** Distinct behaviours constrain complementary subspaces (chemotaxis
  vs. locomotion: 7.15× context shift; cross-behaviour overlap cos 0.54), so the identifiable
  dimension is the **union over the behavioural repertoire** and grows as behaviours are added.
- **Generality.** The same organisation — network connectivity and slow/passive properties stiff,
  fast voltage-gated inward currents sloppy — recurs in nematode, fly larva, rodent, an adult-fly
  visual circuit and the crustacean pyloric population, and in four external biological datasets.

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

Expected output: `=== NUMBER AUDIT: 23/23 PASS ===`, the threshold-robustness table, and the
regenerated figure. Exit status is non-zero if any number fails to reproduce.

## Contents

| Path | Description |
|---|---|
| `paper/` | Manuscript and Supplementary Information (LaTeX + PDF), bibliography, and all 14 figures |
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
