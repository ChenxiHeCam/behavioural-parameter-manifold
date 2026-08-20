# Cover letter — Nature Computational Science

Chenxi He
Cavendish Laboratory, University of Cambridge
ch2067@cam.ac.uk

Dear Editor,

Please consider the enclosed manuscript, **"How much of a nervous-system model does behaviour
identify?"**, for publication as an Article in *Nature Computational Science*.

**The problem.** Whole-organism simulators — connectome-constrained models of *C. elegans* and
*Drosophila*, musculoskeletal fly models, mammalian controllers — are now routinely fitted to
behavioural data on the premise that the fit recovers the underlying biophysics. That premise rests
on an assumption about identifiability that has not been measured at this scale. Work on much
smaller systems, from pyloric degeneracy to sloppy-model theory, shows that many parameter sets
produce indistinguishable dynamics. Whether that holds for a model with three thousand connectome
weights, and which parameters it spares, determines what any behaviour-based calibration can be
expected to return.

**The main result.** We measure it on the BAAIWorm model published in this journal in 2024. Every
one of its 3076 synaptic and gap-junction weights is perturbed in turn and the motor output read
through 9600 observables, so that observables outnumber parameters and the curvature rank is not
capped by the measurement. Behaviour constrains eight directions at 90 per cent of the curvature and
47 at 99 per cent: three thousand directions are flat. Only four weights have no measurable effect,
so the flat directions are genuine degeneracy rather than unconnected parameters. The curvature that
remains divides unevenly by connection type — chemical synapses carry 87 per cent of it and gap
junctions 13 per cent, from 1992 and 1084 connections — so per connection a chemical synapse matters
about five times more.

**What behaviour cannot see is biologically interpretable.** Re-measured over the model's named
conductances through the full motor trajectory, the stiffest axes are the inward rectifier IRK and
the sodium leak NCA, whereas the fast voltage-gated calcium channels EGL-19 and UNC-2 are among the
sloppiest and the calcium-activated potassium channels SLO-1 and SLO-2 produce no measurable change
at all. Those are precisely the currents known to be homeostatically compensated and variable
between individual animals: the directions behaviour cannot resolve are the directions biology
itself leaves free.

**A methodological point we think the field needs.** An effective dimension is a statement about a
measurement as much as about a model, and we show this concretely. A Gauss–Newton Hessian built from
*n* observables has rank at most *n*, so the same FlyGym controller reads as 1 of 48 dimensions
through ten summary statistics and 16 of 48 through 4200 joint-angle observables, against a noise
floor of 7 that we measure rather than assume. We therefore accompany every curvature figure with a
step-size scan, a noise floor, and a count of parameters that do anything at all — and report that
two widely used simulators fail this test outright: one has a discontinuous response to its own
parameters, and in another any perturbation decorrelates the trajectory, so the phase-invariant
summary statistics that literature uses are the correct choice rather than a shortcut. Separating
degeneracy in a model from a defect in it has not, to our knowledge, been done systematically.

**On behavioural richness.** Enriching the behavioural repertoire does enlarge the identifiable
subspace, monotonically and in every threshold, but the gain is bounded and saturates by the third
or fourth behaviour: four fly behaviours give 19 identifiable dimensions rather than four times the
16.5 that one gives. We report that as measured, with its saturation, rather than as an identity.

**Fit to the journal.** The work is a measurement of computational models rather than of animals,
its object is a simulator published in *Nature Computational Science*, and its practical output is a
diagnostic that any group fitting a mechanistic model to behaviour can run before committing to the
fit. All analysis code, the per-experiment result files, and a container that reproduces every
reported quantity from those files are deposited publicly.

We confirm that this manuscript is not under consideration elsewhere.

Yours sincerely,
Chenxi He
