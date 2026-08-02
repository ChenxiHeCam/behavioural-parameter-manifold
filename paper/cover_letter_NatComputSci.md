# Cover letter — Nature Computational Science

Chenxi He
Cavendish Laboratory, University of Cambridge
ch2067@cam.ac.uk

Dear Editor,

Please consider the enclosed manuscript, **"Behavioural repertoires tile the identifiable parameter
subspaces of whole-organism nervous-system models"**, for publication as an Article in *Nature
Computational Science*.

**The computational problem.** Whole-organism simulators — connectome-constrained models of
*C. elegans* and *Drosophila*, musculoskeletal fly models, and mammalian controllers — are now
routinely fitted to behavioural data on the premise that the fit recovers the underlying biophysics.
That premise rests on an unexamined assumption about identifiability. Work on much smaller systems,
from pyloric degeneracy to sloppy-model theory, shows that many parameter sets produce
indistinguishable dynamics; whether that holds at whole-organism scale, and what governs it, had not
been measured. This is a question about the models themselves, and it determines what any
behaviour-based calibration pipeline can be expected to return.

**What we contribute.** We give a measurement framework and apply it uniformly across simulators
spanning three phyla. Behavioural curvature is computed in a single Gauss–Newton (Fisher) form, so
identifiability is read off a positive semi-definite matrix with an explicitly defined effective
dimension, and the same estimator is applied to every system from 7 to 767 parameters. Two
independent probes — a curvature probe and an optimisation-based manifold-drift probe — agree on
which axes are identifiable (Spearman *ρ* = 0.96), with an amortised posterior as a calibration
control.

**The main result.** Behaviour constrains only a low-dimensional stiff subspace. The informative part
is what governs its extent: distinct behaviours constrain *complementary*, not overlapping,
subspaces, so the identifiable dimension is the union over the behavioural repertoire and grows as
behaviours are added. Because a union grows whenever subspaces differ at all, we test this against
explicit redundant and random-orientation null models; the observed geometry is significantly more
aligned than random (*p* = 0.016–0.020) while remaining far from redundant, so the growth is neither
trivial nor mere redundancy. This reframes identifiability as a quantity set by the diversity of
behaviours one elicits, rather than a fixed property of a model — an actionable statement for anyone
building or calibrating these simulators.

**Why *Nature Computational Science*.** The contribution is methodological and transferable rather
than organism-specific: a way to measure what a behavioural objective can and cannot determine in a
mechanistic simulator, demonstrated to behave consistently across five systems and three phyla, with
calibration controls, null models and a container that re-derives every reported number from its
source file. It speaks directly to readers building digital organisms, to those doing
simulation-based inference on mechanistic models, and to the broader identifiability literature.

**Scope and honesty of claims.** We are explicit about the limits. The measurements are of models,
and the title and claims are scoped accordingly. The four biological datasets are reported as
convergent observations, not as validation, because they compare different levels of description. We
state where a low-dimensional behaviour need *not* bound identifiable rank in general, we report the
confounds that remain uncontrolled in the one high-dimensional contrasting system, and we name the
prospective out-of-sample test that current datasets cannot support. One preliminary probe that
evaluated a raw Hessian at a saddle point is documented and explicitly withdrawn from the reported
results.

All data are public, and the code, result files and reproduction container are deposited at
https://doi.org/10.5281/zenodo.21594746. We confirm that this manuscript is not under consideration
elsewhere and that the author declares no competing interests.

Thank you for your consideration.

Yours sincerely,

Chenxi He

---

## Editor-facing summary (2 sentences)
We measure what a behavioural objective can identify in whole-organism nervous-system simulators, and
find that it constrains only a low-dimensional subspace of parameters. Because distinct behaviours
constrain complementary subspaces, that identifiable dimension is the union over the behavioural
repertoire and grows with it — so identifiability is set by the diversity of behaviours one measures,
a result we establish across three phyla against explicit null models.

## Suggested reviewers (non-conflicted; to be confirmed)
- Sloppy-model / systems-biology identifiability theory (the Sethna–Transtrum line).
- Neural degeneracy and homeostatic compensation (the Marder–Prinz line).
- Developers of connectome-constrained whole-organism simulation.
- Simulation-based inference for mechanistic models.

## Related manuscripts by the author
A separate manuscript on behaviour-based parameter generation for whole-organism simulators is
available as a preprint (https://doi.org/10.21203/rs.3.rs-10486658/v1) and is cited here as related work.
The present study is independent of it: it measures identifiability in published simulators and
shares no analyses, datasets or claims with that manuscript.
