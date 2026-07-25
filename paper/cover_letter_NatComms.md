# Cover letter — Nature Communications

Chenxi He
Cavendish Laboratory, University of Cambridge
ch2067@cam.ac.uk

Dear Editor,

Please consider the enclosed manuscript, **"Behaviour identifies a low-dimensional parameter
subspace tiled by the behavioural repertoire"**, for publication as an Article in *Nature
Communications*.

**The problem.** Whole-organism simulators of nervous systems — connectome-constrained models of
*C. elegans* and *Drosophila*, and their mammalian counterparts — are now routinely fitted to
behaviour on the premise that doing so recovers the underlying biophysics. Decades of work on
smaller systems, from pyloric degeneracy to the theory of sloppy models, suggest this premise is
unsafe: many parameter sets produce the same dynamics. Whether, and how strongly, that degeneracy
holds at whole-organism scale had not been measured.

**What we did.** We measure the *behavioural-equivalence manifold* — the set of parameter
configurations that yield indistinguishable behaviour — across simulators spanning three phyla,
using two independent geometric probes with an inference-based control. Behaviour constrains a
low-dimensional stiff subspace and leaves the rest free, at every parameter scale from 5 to 767.

**What is new.** Beyond establishing the measurement, we characterise the subspace in a way that
changes how identifiability should be understood in this field. Distinct behaviours constrain
*complementary*, not overlapping, parameter subspaces: in the nematode, chemotaxis renders
identifiable the chemosensory parameters that locomotion leaves silent, a sevenfold context shift.
The identifiable dimension is therefore the **union over the behavioural repertoire**, and it grows
as behaviours are added — which we demonstrate in nematode, fly-larva and rodent models. This
reframes degeneracy from a fixed limit into a measurable resource: a model is identifiable up to the
diversity of behaviours one is willing to elicit. We further identify what the manifold *is* in the
nematode — the neural substrate of the animal's four-dimensional eigenworm postural space, with two
independently built models occupying the empirically measured basis — and show that the same
organisation, with network connectivity stiff and fast voltage-gated inward currents sloppy, recurs
in every system we examined.

**Why *Nature Communications*.** The result is a measurement with immediate consequences across
several communities that your readership spans: it tells modellers what behaviour-based calibration
can and cannot recover, tells experimentalists that assay diversity — not assay duration — sets what
is identifiable, and lifts the sloppy-models and neural-degeneracy literature from single circuits to
behaving whole organisms. It is a general, transferable framework rather than a result about one
model organism, which is why we believe it fits the broad scope of *Nature Communications* rather
than a specialist venue.

**Rigour and reproducibility.** The two geometric probes agree (Spearman *ρ* = 0.96, exact
permutation *p* = 0.0028); two independently constructed worm models agree on the biophysical
partition; the effective-dimension estimator is calibrated against spectra of known rank and is
insensitive to the spectral-mass threshold; and four external biological datasets that use none of
our simulators show the same geometry. All data are public, and a container reproduces every
reported quantity and figure from the deposited result files.

We confirm that this manuscript is not under consideration elsewhere and that the author declares no
competing interests.

Thank you for your consideration.

Yours sincerely,

Chenxi He

---

## Suggested reviewers (non-conflicted; to be confirmed)
- A researcher in sloppy-model / systems-biology identifiability theory (e.g. the
  Sethna/Transtrum line of work).
- A researcher in neural degeneracy and homeostatic compensation (e.g. the Marder/Prinz line).
- A developer of connectome-constrained whole-organism simulation.
- A researcher in quantitative behaviour and low-dimensional postural analysis.

## Editor-facing summary (2 sentences)
Behaviour constrains whole-organism nervous-system models on a low-dimensional subspace, and we
measure it directly across simulators spanning three phyla. Because distinct behaviours constrain
complementary subspaces, the identifiable dimension is the union over the behavioural repertoire and
grows with it, which reframes model degeneracy as a resource set by the diversity of behaviours one
measures.
