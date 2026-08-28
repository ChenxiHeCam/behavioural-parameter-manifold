# Cover letter — Nature Computational Science

Chenxi He
Cavendish Laboratory, University of Cambridge
ch2067@cam.ac.uk

Dear Editors,

Please consider the enclosed manuscript, **"How much of a nervous-system model does behaviour
identify?"**, for publication as an Article in *Nature Computational Science*. I am the sole
author, and the work is not under consideration elsewhere.

**The question.** Whole-organism simulators are built against the animal's own data — virtual
flies imitate recorded trajectories, a virtual nematode tuned to electrophysiology is judged by
whether the assembled loop crawls and chemotaxes like the worm — and the calibrated model is then
read as the animal's biophysics. That reading assumes behaviour determines the parameters that
produce it. Classic work on small circuits, from pyloric degeneracy to sloppy-model theory, says
the map from parameters to dynamics is many-to-one; whether and how strongly that holds for a model
with thousands of connectome weights had never been measured. The answer decides what any
behaviour-based calibration of these models can be expected to return.

**The measurement.** On the BAAIWorm nematode model published in this journal (Zhao et al., *Nat.
Comput. Sci.* 4, 106–120), we perturb each of the 3076 connectome weights in turn and read the
motor command through 9600 observables, so observables outnumber parameters and the curvature rank
is not capped by the assay. Eight directions carry 90% of the Gauss–Newton curvature and 47 carry
99%; only four weights produce no measurable effect, so the weakly curved directions are
degeneracy, not disconnection. The number describes a region, not a point — four displaced
operating points give 8–12 with coinciding stiff subspaces — and a connectome-constrained
*Drosophila* visual model concentrates its curvature the same way, at 14 of 330 sampled
parameters.

**What the geometry means.** What is determined and what is left free has biophysical identity:
chemical synapses carry 87% of the curvature against 13% from gap junctions, and the conductances
behaviour leaves free — the calcium currents EGL-19 and UNC-2 — are the currents biology itself
lets vary between animals. Refining one action saturates (twelve behaviours never exceed the
subspace four of them already span), whereas a functionally different action extends the identified
set; assay diversity, not recording duration, is the quantity to optimise. And at the cell scale,
where truth is knowable, the geometry is verified against it: identified models are recovered
exactly, the canonical MAPK cascade matches behaviour with its parameters 40% wrong, and an
effective dimension is shown to be a statement about curvature concentration, not a bound on
recoverable parameters — twelve assay combinations on one neuron all give dimension one while
recovery error spans tenfold.

**Why we think this belongs in this journal.** The contribution is a measurement of a property of
computational models, made with the discipline such a measurement needs. Every reported dimension
carries three admission conditions — observables outnumbering parameters, response proportional to
the step, signal above the simulator's own noise floor — and the conditions have teeth: they
retired our own headline number for the pyloric circuit, whose response saturates in the step for
every observable set we tried, and they corrected the fly walking figure from 16 to 11 once it was
measured where the response is proportional. The release deposits one result file per analysis, a
72-check audit that re-derives the audited quantities from those files in a container, and a
specification table giving every assay's step, rollout, observables, whitening and seeds.

For programmes that fit mechanistic models to behaviour — digital twins included — the practical
statement is short: the stiff subspace is the part of a fitted model the assay pins down and that
supports biological interpretation; the free directions name what it leaves open and which further
behaviours would close it.

Thank you for your consideration.

Chenxi He
