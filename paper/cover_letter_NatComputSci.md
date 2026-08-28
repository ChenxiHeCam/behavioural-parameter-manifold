# Cover letter — Nature Computational Science

Chenxi He
Cavendish Laboratory, University of Cambridge
ch2067@cam.ac.uk

Dear Editors,

Please consider the manuscript "How much of a nervous-system model does behaviour identify?" for
publication as an Article in *Nature Computational Science*. I am the sole author, and the work is
not under consideration elsewhere.

Whole-organism simulators are calibrated against behaviour and the calibrated model is then read as
the animal's biophysics, yet which parameter combinations behaviour determines is unknown. This
matters because behaviour-based calibration can recover a working parameter set without recovering
unique biophysical values. The manuscript measures what behaviour can and cannot identify in
published whole-organism nervous-system models.

In the connectome-based nematode model BAAIWorm, published in this journal (Zhao et al., *Nat.
Comput. Sci.* 4, 106–120; 2024), perturbing each of the 3,076 connectome weights with more
observables than parameters shows that behaviour constrains eight directions at 90% of the
curvature and 47 at 99%. A connectome-constrained *Drosophila* optic-lobe model gives a
corresponding result, 14 of 330 sampled directions at 90%: across a motor and a visual model in two
phyla, behaviour leaves the great majority of a connectome-scale parameter space free. The
constrained space has biological structure: chemical synapses carry 87% of the curvature and gap
junctions 13%, while the calcium currents EGL-19 and UNC-2, documented as homeostatically
compensated in small circuits, are left free. Cell models with known truth provide a direct check:
small neuronal models are recovered exactly, whereas the degenerate MAPK cascade matches behaviour
with its parameters 40% from the truth.

The analysis yields an assay-design principle. Further aspects of one action add little, because
the subspaces they determine overlap; an action with different functional demands reaches
directions the first cannot. Assay diversity rather than recording duration is therefore the
quantity to optimise. Every reported dimension carries three admission conditions — observable
count, step-size proportionality, and the simulator's own noise floor — and these conditions have
teeth: they retired our own headline number for the pyloric circuit.

The study suits *Nature Computational Science* because it makes behavioural identifiability a
measurable property of mechanistic simulation and connects model geometry to assay design. Its
scope spans connectome-scale motor and visual models and cell models with known parameters, and it
should interest the journal's readership across computational modelling, biological simulation and
parameter inference.

A related manuscript describing the behaviour-based parameter-generation method, used here as one
independent probe and cited in the text, is available as a preprint (Research Square,
doi:10.21203/rs.3.rs-10486658); its claims do not overlap with this study's.

Thank you for your consideration.

Yours sincerely,

Chenxi He
