"""Convert Results narration to Nature-style 'we' + past tense."""
import os
f = os.path.join(os.path.dirname(__file__), '..', 'paper', 'paper2_main.tex')
s = open(f, encoding='utf-8').read()

subs = [
 # --- Results 1 ---
 ("We first establish that the manifold has positive dimension. Repeating optimisation from independent starting points against a single fixed behavioural target, the behavioural loss converges while the recovered parameter vectors remain dispersed.",
  "We first asked whether the manifold has positive dimension. Repeating optimisation from independent starting points against a single fixed behavioural target, the behavioural loss converged while the recovered parameter vectors remained dispersed."),
 ("Behavioural convergence with persistent parameter dispersion is direct evidence that the equivalence manifold has positive dimension",
  "Behavioural convergence with persistent parameter dispersion is direct evidence that the equivalence manifold has positive dimension"),
 # --- Results 2 ---
 ("The behavioural-loss Hessian at the optimum quantifies the dimensionality directly. On the $767$-parameter flybody policy, a Lanczos spectrum yields",
  "We next quantified the dimensionality directly from the behavioural-loss Hessian at the optimum. On the $767$-parameter flybody policy, a Lanczos spectrum yielded"),
 ("On the full coupled real modWorm model, the per-mechanism behavioural Hessian has effective dimension $2$ of $7$",
  "On the full coupled real modWorm model, the per-mechanism behavioural Hessian had effective dimension $2$ of $7$"),
 ("On the $48$-dimensional FlyGym walking controller, the behavioural Hessian has effective dimension $3$ ($90\\%$) / $6$ ($99\\%$) of $48$ and spans",
  "On the $48$-dimensional FlyGym walking controller, the behavioural Hessian had effective dimension $3$ ($90\\%$) / $6$ ($99\\%$) of $48$ and spanned"),
 ("On the BAAIWorm five-mechanism probe, the synaptic, gap-junction and motor-output scalings carry the curvature while the fast single-cell conductances do not.",
  "On the BAAIWorm five-mechanism probe, the synaptic, gap-junction and motor-output scalings carried the curvature while the fast single-cell conductances did not."),
 # --- Results 3 (eigenworm) ---
 ("Real \\emph{C.~elegans} posture from the OpenWorm Movement Database occupies an effective $3.79$--$4.14$ dimensions, with four eigenworms capturing",
  "Real \\emph{C.~elegans} posture from the OpenWorm Movement Database occupied an effective $3.79$--$4.14$ dimensions, with four eigenworms capturing"),
 ("Two independently constructed worm models generate behaviour that occupies this same empirical eigenworm space: the modWorm posture subspace aligns",
  "Two independently constructed worm models generated behaviour occupying this same empirical eigenworm space: the modWorm posture subspace aligned"),
 ("These alignments substantially exceed those of random four-dimensional subspaces",
  "These alignments substantially exceeded those of random four-dimensional subspaces"),
 ("and within each model it is the stiff mechanisms, not the sloppy ones, that displace the eigenworm trajectory",
  "and within each model it was the stiff mechanisms, not the sloppy ones, that displaced the eigenworm trajectory"),
 ("as re-measuring with $7{,}200$--$14{,}400$ posture observables leaves the effective dimension at $2$--$3$ of $7$",
  "as re-measuring with $7{,}200$--$14{,}400$ posture observables left the effective dimension at $2$--$3$ of $7$"),
 # --- Results 4 (tiling) ---
 ("In BAAIWorm, perturbing the chemosensory parameters alters behaviour $7.15\\times$ more during chemotaxis",
  "In BAAIWorm, perturbing the chemosensory parameters altered behaviour $7.15\\times$ more during chemotaxis"),
 ("whereas perturbing the motor parameters shows the opposite contrast",
  "whereas perturbing the motor parameters showed the opposite contrast"),
 ("Within a single worm model this union increases as behaviours are added: four distinct locomotor presets each constrain",
  "Within a single worm model this union increased as behaviours were added: four distinct locomotor presets each constrained"),
 ("their behaviour-specific stiff directions span an effective $3.0$ of $4$ dimensions",
  "their behaviour-specific stiff directions spanned an effective $3.0$ of $4$ dimensions"),
 ("Their union reaches $5$; across twelve behaviours it saturates at $4$ of $7$",
  "Their union reached $5$; across twelve behaviours it saturated at $4$ of $7$"),
 # --- Results 5 (cross-phyla) ---
 ("We measure it also in a fly larva (larvaworld) and a mammal",
  "We measured it also in a fly larva (larvaworld) and a mammal"),
 ("and in each case the identifiable dimension increases as distinct behaviours are added",
  "and in each case the identifiable dimension increased as distinct behaviours were added"),
 ("so that their union ($8$ of $10$) is approximately double the single-behaviour dimension",
  "so that their union ($8$ of $10$) was approximately double the single-behaviour dimension"),
 ("In the rodent, a six-behaviour repertoire drives a still-increasing identifiability curve",
  "In the rodent, a six-behaviour repertoire drove a still-increasing identifiability curve"),
 # --- Results 6 (probes) ---
 ("the two geometric probes concur on which axes are identifiable: per-mechanism Hessian stiffness and negative manifold drift are rank-correlated",
  "the two geometric probes concurred on which axes are identifiable: per-mechanism Hessian stiffness and negative manifold drift were rank-correlated"),
 ("A third probe serves as a calibration control rather than a parallel identifiability ranking: an amortised neural posterior (NPE/MAF, \\texttt{sbi}) trained in the same region on $N{=}300$ real simulations is well-calibrated",
  "A third probe served as a calibration control rather than a parallel identifiability ranking: an amortised neural posterior (NPE/MAF, \\texttt{sbi}) trained in the same region on $N{=}300$ real simulations was well-calibrated"),
 # --- Results 7 (biophysics) ---
 ("At the BAAIWorm optimum the stiff axes are the synaptic ($331.97$), gap-junction ($299.64$) and motor-output ($270.50$) weights.",
  "At the BAAIWorm optimum the stiff axes were the synaptic ($331.97$), gap-junction ($299.64$) and motor-output ($270.50$) weights."),
 ("A complementary probe over the full $3076$-weight connectome finds the connection weights collectively stiff",
  "A complementary probe over the full $3076$-weight connectome found the connection weights collectively stiff"),
 # --- Results 8 (external) ---
 ("Four external datasets, none of which involves our forward models, exhibit the same low-dimensional stiff/sloppy structure",
  "Four external datasets, none of which involves our forward models, exhibited the same low-dimensional stiff/sloppy structure"),
 ("yields a $31$-parameter eigenspectrum spanning approximately ten orders of magnitude",
  "yielded a $31$-parameter eigenspectrum spanning approximately ten orders of magnitude"),
]
n = 0
for a, b in subs:
    if a in s:
        s = s.replace(a, b, 1); n += 1
    else:
        print("NOT FOUND:", a[:65])
open(f, 'w', encoding='utf-8').write(s)
print(f"tense conversions: {n}/{len(subs)}")
