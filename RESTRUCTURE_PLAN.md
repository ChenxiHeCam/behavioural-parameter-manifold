# Paper 2 restructure, from the supplementary experiments

## What the measurements now say

### The anchor
BAAIWorm, the connectome-constrained biophysical model published in this journal
in 2024: **3076 synaptic and gap-junction weights, effective dimension 8 (90%) /
47 (99%), participation ratio 2.55.**

It survives every check that the other systems fail:

| check | result |
|---|---|
| observables exceed parameters | 9600 against 3076, no rank ceiling |
| response linear in the step | ratio constant to four significant figures over a 20x range |
| simulation noise | zero; repeat runs are bit-identical |
| dead parameters | 4 of 3076 |

So: **behaviour, read from the motor output, constrains eight directions out of
three thousand.** That is the paper's central number and it is now unassailable.

### What behaviour identifies, biologically
Re-measured without the observable rank cap, over the model's own named
conductances (12 with a measurable effect, thousands of observables):
effective dimension 3 (90%) / 6 (99%), elasticities spanning three orders.

Stiff: IRK (Kir), NCA (Na-leak), EGL-36 (Kv), EGL-2 (Kv-eag), CCA-1 (T-Ca).
Sloppy: KVS-1, EGL-19 (L-Ca), SHL-1, UNC-2 (PQ-Ca).
No measurable effect at all: SLO-1, SLO-2 (calcium-activated potassium).

The paper's biological reading survives: the fast voltage-gated calcium
conductances and the calcium-activated potassium conductances are the ones
behaviour cannot see, and those are exactly the currents known to be
homeostatically compensated and variable between animals.

At connection level the same measurement splits the connectome: chemical synapses
carry 87% of the curvature and gap junctions 13%, from 1992 and 1084 connections
respectively -- per connection, synapses matter about five times more.

## What has to be withdrawn

### The repertoire claim: supported where it was actually tested, and bounded

Enriching the behavioural repertoire does increase the identifiable dimension. The
effect is real, monotone and reproducible, but it is smaller than the submitted
claim implies and it saturates quickly.

| model | were the behaviours genuinely different? | 1 -> 4 behaviours |
|---|---|---|
| FlyGym | yes: straight walking and two turns, visibly different gaits | 90%: 16.5 -> 19.0; 99%: 37 -> 40; PR 8.18 -> 10.33 |
| modWorm | marginally: variants of one escape response | 90%: 2.50 -> 3.00 |
| BAAIWorm | no: driving different sensory neurons left the motor output almost unchanged | independent variable did not move |
| flyvis | no: four full-field visual patterns | independent variable did not move |

Both systems in which the behaviours really differed gave an increase, in every
threshold and in the threshold-free participation ratio, with the spread across
behaviour subsets shrinking to zero as behaviours are added. A shuffled control,
in which the behaviour-specific structure of each Jacobian is destroyed, gives a
flat and higher value, so the measured low dimensionality is structure rather than
artefact.

Three quantitative statements replace the submitted one:

1. Behaviours overlap far more than they complement. Four behaviours give a union
   of 19 of 48, not four times the 16.5 that one behaviour gives.
2. There is nevertheless a net gain, consistent across thresholds: about two and a
   half dimensions of 48.
3. The return diminishes sharply and saturates by the third or fourth behaviour.

The practical reading is the useful part: adding behavioural paradigms helps, but
runs out quickly, and is worth less than measuring a single behaviour densely.

What remains open is whether behaviours that recruit genuinely different motor
programmes -- forward locomotion, reversal, omega turn in the connectome-scale
model -- extend the curve further than the same-programme variants tested here.

### Numbers that are wrong
1. Supplementary S5.3 concludes the connection-weight space is "collectively
   stiff" with "essentially no direction flat". Measured: 3068 of 3076 directions
   are flat. The conclusion is inverted.
2. Gap junctions are described as among the stiffest mechanisms. Per connection,
   chemical synapses carry about five times more curvature.
3. modWorm: three of the seven mechanisms (leak, capacitance, sigmoid gain) change
   the posture by 1e-12 to 1e-7, machine precision. They are unconnected, not
   sloppy. Any "2 of 7" is really "2 of 4".
4. FlyGym "3 of 48" is set by using ten observables; with 4200 the figure is 16.
5. flyvis "43 of 65" tracks the readout count (64 readouts; at 6 readouts it is 5)
   and probes only the 65 biases of 734 free parameters.
6. The solution-cloud dimension of 21.7 of 126 is not evidence of a manifold: a
   problem with a unique optimum and no manifold gives 20.2 under the same
   estimator. The informative quantity is the dispersion magnitude, which differs
   by 78x between the two.
7. The spectral span of "19.5 orders of magnitude" is measured to eigenvalues at
   1e-19; above the stated finite-difference floor the span is 3.85 orders.
8. The eigenworm claim: the model's own posture is 1.66-dimensional against 3.79
   for the animal, and per-mode alignments are 0.30, 0.02, 0.57, 0.24. Only the
   subspace-averaged 0.77 is reported.

## New section the paper needs

**Finite-difference validity.** No curvature number means anything without it, and
it separates biological degeneracy from modelling artefacts:

| model | diagnosis |
|---|---|
| BAAIWorm | linear over a 20x step range, zero noise |
| flyvis | linear over two decades, ratio constant at 0.498 |
| FlyGym | noise floor dominates below step 0.1; linear above 0.25 |
| STG | any perturbation decorrelates the voltage trace (r = 0.42 at 0.1%), so
  trajectory observables are unusable; the summary statistics the literature uses
  are the correct response to this, not a shortcut |
| modWorm | discontinuous: zero below 1%, jumps at 5%, identical at 5% and 10% |

## Proposed thesis

Behaviour constrains a small, biophysically interpretable subspace of a
whole-organism model. In the connectome-scale nematode model, eight of three
thousand connection directions are identifiable, and the conductances behaviour
cannot see are the fast calcium and calcium-activated potassium currents, which
are the currents biology itself leaves variable between animals.

Enriching the behavioural repertoire does enlarge that subspace, monotonically and
in every measure, but the behaviours overlap heavily and the gain saturates after
three or four of them: the identifiable dimension is bounded well below the sum of
what each behaviour constrains alone.

That is narrower than the submitted claim, and every part of it is measured on a
model that passes the differentiability and noise controls.
