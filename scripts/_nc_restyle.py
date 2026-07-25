"""Restyle paper2_main.tex into Nature Communications house style."""
import re, os
f = os.path.join(os.path.dirname(__file__), '..', 'paper', 'paper2_main.tex')
s = open(f, encoding='utf-8').read()

# 1. TITLE: 17 -> 12 words
s = s.replace(
    "\\title{\\textbf{The behavioural-equivalence manifold of whole-organism nervous systems: behaviour identifies a low-dimensional subspace tiled by the behavioural repertoire}}",
    "\\title{\\textbf{Behaviour identifies a low-dimensional parameter subspace tiled by the behavioural repertoire}}")

# 2. ABSTRACT: NC style ~195 words, no citations
old_ab = re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}', s, re.S).group(0)
new_ab = (
"\\begin{abstract}\n\\noindent\n"
"Whole-organism biophysical simulators reproduce animal behaviour from hundreds to thousands of internal "
"parameters, yet how many of those parameters behaviour actually constrains is unknown. Here we measure the "
"behavioural-equivalence manifold, the set of parameter configurations that yield indistinguishable behaviour, "
"across simulators spanning three phyla. We find that behaviour constrains a low-dimensional stiff subspace and "
"leaves the remaining directions free, and we characterise that subspace in three ways. Its identity: in the "
"nematode it is the neural substrate of the animal's four-dimensional postural repertoire, and two independently "
"built models generate behaviour that occupies the measured eigenworm space. Its dependence on behaviour: distinct "
"behaviours constrain complementary rather than overlapping subspaces, so that the identifiable dimension is the "
"union over the behavioural repertoire and grows as behaviours are added, which we demonstrate in nematode, "
"fly-larva and rodent models. Its generality: the same organisation, with network connectivity and slow membrane "
"properties stiff and fast voltage-gated inward currents sloppy, recurs in every system examined and in four "
"external biological datasets that use none of our simulators. Behaviour therefore constrains whole-organism "
"nervous systems on a low-dimensional subspace whose extent is set by behavioural richness, leaving an "
"irreducible degenerate core.\n\\end{abstract}")
s = s.replace(old_ab, new_ab)

# 3. RESULTS SUBHEADINGS: NC uses declarative sentences as subsections
heads = [
 ("\\paragraph{A positive-dimensional behavioural-equivalence manifold.}",
  "\\subsection*{Behaviour converges while parameters remain dispersed}"),
 ("\\paragraph{The manifold is low-dimensional.}",
  "\\subsection*{The behavioural-equivalence manifold is low-dimensional}"),
 ("\\paragraph{The manifold is the neural control of the eigenworm postural space.}",
  "\\subsection*{The manifold is the neural control of eigenworm posture}"),
 ("\\paragraph{Complementary tiling of the identifiable subspace by the behavioural repertoire.}",
  "\\subsection*{Distinct behaviours constrain complementary subspaces}"),
 ("\\paragraph{Recurrence of the geometry across three phyla.}",
  "\\subsection*{The same geometry recurs across three phyla}"),
 ("\\paragraph{Convergence of two geometric probes, and calibration of the amortised posterior.}",
  "\\subsection*{Two independent probes agree on the identifiable axes}"),
 ("\\paragraph{Biophysical interpretation of the stiff and sloppy directions.}",
  "\\subsection*{Stiff and sloppy directions are biophysically interpretable}"),
 ("\\paragraph{Recurrence of the geometry in external biological data.}",
  "\\subsection*{The same geometry appears in external biological data}"),
]
n = 0
for a, b in heads:
    if a in s:
        s = s.replace(a, b, 1); n += 1
    else:
        print("HEAD NOT FOUND:", a[:60])
print(f"headings converted: {n}/{len(heads)}")

# 4. Author contributions (NC requires it)
if "Author contributions" not in s:
    s = s.replace("\\section*{Competing interests}",
        "\\section*{Author contributions}\nC.H. conceived the study, designed and performed all analyses, "
        "and wrote the manuscript.\n\n\\section*{Competing interests}")

open(f, 'w', encoding='utf-8').write(s)
print("restyle written")
