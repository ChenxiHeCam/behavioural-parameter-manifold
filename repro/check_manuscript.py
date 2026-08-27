"""Check the manuscript against the journal's Article limits and against itself.

Journal limits (Nature Computational Science, Article): abstract up to 150 words and
unreferenced, main text up to 3,500 words excluding abstract, Methods, references and
figure legends, up to 6 display items, about 50 references, title up to 10 words or
90 characters, Results carries topical subheadings and the Discussion does not.

Self-consistency: every Supplementary cross-reference resolves, every result file
cited in either document exists, and every figure clears 300 dpi at its printed width.

    python repro/check_manuscript.py

Exit status is non-zero if anything fails, so it can gate a commit.
"""
import io, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_TEXT_MM = 183.0
LIM = {"abstract": 150, "main": 3500, "displays": 6, "refs": 50,
       "title_words": 10, "title_chars": 90, "dpi": 300}


def words(t):
    """Count as the rendered page would: a maths group is one word."""
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", t, flags=re.S)
    t = re.sub(r"\\citep?\[[^]]*\]\{[^}]*\}", "", t)
    t = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", "", t)
    t = re.sub(r"\$[^$]*\$", "N", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}\\~&]", " ", t)
    return len(t.split())


def read(*parts):
    return io.open(os.path.join(ROOT, *parts), encoding="utf-8").read()


fail = []


def check(label, got, limit=None, ok=None):
    if ok is None:
        ok = got <= limit
    print("  %-42s %-22s %s" % (label, got if limit is None else "%s / %s" % (got, limit),
                                "" if ok else "OVER"))
    if not ok:
        fail.append(label)


M = read("paper", "manuscript.tex")
ab_src = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", M, re.S).group(1)
title = re.sub(r"\\textbf\{|\}", "", re.search(r"\\title\{(.*?)\}\s*\n", M, re.S).group(1)).strip()
intro = words(M.split("Introduction}")[1].split("\\section*{Results}")[0])
resdisc = words(M.split("\\section*{Results}")[1].split("\\section*{Methods}")[0])
disc = M.split("\\section*{Discussion}")[1].split("\\section*{Methods}")[0]

print("journal limits")
check("title, words", len(title.split()), LIM["title_words"])
check("title, characters", len(title), LIM["title_chars"])
check("abstract, words", words(ab_src), LIM["abstract"])
check("abstract is unreferenced", "yes" if "\\cite" not in ab_src else "NO",
      None, "\\cite" not in ab_src)
check("main text, words", intro + resdisc, LIM["main"])
check("display items", M.count("begin{figure}") + M.count("begin{table}"), LIM["displays"])
cites = set()
for m in re.findall(r"\\cite[a-zA-Z]*\{([^}]*)\}", M):
    cites.update(x.strip() for x in m.split(","))
check("references", len(cites), LIM["refs"])
check("Discussion has no subheadings", disc.count("\\subsection"), 0)

print("\nself-consistency")
toc_path = os.path.join(ROOT, "paper", "supplementary.toc")
if os.path.exists(toc_path):
    nums = set(re.findall(r"numberline \{([0-9]+(?:\.[0-9]+)?)\}", read("paper", "supplementary.toc")))
    si = sorted(set(re.findall(r"\\S([0-9]+(?:\.[0-9]+)?)", M)),
                key=lambda x: [int(v) for v in x.split(".")])
    miss = [c for c in si if c not in nums]
    check("Supplementary references resolve", "%d of %d" % (len(si) - len(miss), len(si)),
          None, not miss)
else:
    print("  (supplementary.toc absent; compile the supplement to check its references)")

res = set(os.listdir(os.path.join(ROOT, "results")))
cited = set()
for name in ("manuscript.tex", "supplementary.tex"):
    cited.update(re.findall(r"([A-Za-z0-9_\-]+\.json)", read("paper", name).replace("\\_", "_")))
missing = sorted(c for c in cited if c not in res)
check("cited result files exist", "%d of %d" % (len(cited) - len(missing), len(cited)),
      None, not missing)
if missing:
    print("      missing: " + ", ".join(missing))

try:
    from PIL import Image
    low = []
    figs = {}
    for name in ("manuscript.tex", "supplementary.tex"):
        for m in re.finditer(r"\\includegraphics\[width=([0-9.]+)\\textwidth\]\{([^}]+)\}",
                             read("paper", name)):
            figs[m.group(2)] = float(m.group(1))
    for f, frac in figs.items():
        path = os.path.join(ROOT, "paper", "figures", f)
        if not os.path.exists(path):
            low.append(f + " (absent)")
        elif Image.open(path).size[0] / (W_TEXT_MM * frac / 25.4) < LIM["dpi"]:
            low.append(f)
    check("figures at 300 dpi or better", "%d of %d" % (len(figs) - len(low), len(figs)),
          None, not low)
    if low:
        print("      below: " + ", ".join(low))
except ImportError:
    print("  (Pillow absent; figure resolution not checked)")

print("\n=== MANUSCRIPT: %s ===" % ("PASS" if not fail else "FAIL -- " + "; ".join(fail)))
sys.exit(1 if fail else 0)
