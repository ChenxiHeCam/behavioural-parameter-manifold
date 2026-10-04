"""Scan both documents for constructions the manuscript deliberately avoids.

The paper states what was measured and concentrates caveats in the Limitations
paragraph. This flags the constructions that erode that: em-dashes in prose,
hedging formulas, changelog phrasing left over from drafting, bullet lists,
and any leak of infrastructure detail into the text.

Not every hit is a fault. Em-dashes inside section headings and in table cells
that mark a withdrawn row are legitimate, as is "is reported as" in a statement
of what a quantity is. Read the context each flags.

    python repro/check_style.py
"""
import io, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERNS = [
    ("em-dash in prose",        r"---"),
    ("hedge: we do not claim",  r"we do not claim"),
    ("hedge: it should be noted", r"[Ii]t should be noted"),
    ("hedge: we note that",     r"[Ww]e note that"),
    ("hedge: may / might",      r"\b(?:may|might) (?:be|reflect|indicate|explain)\b"),
    ("hedge: suggests",         r"\bsuggests? that\b"),
    ("changelog: previously",   r"previously reported|previously stated|used to (?:say|report|read)"),
    ("changelog: earlier version", r"[Aa]n earlier version|earlier draft|in a previous"),
    ("changelog: superseded",   r"superseded"),
    ("bullet list",             r"\\begin\{itemize\}|\\begin\{enumerate\}"),
    ("infrastructure leak",     r"/root/|ssh -p"),
    ("not X but Y",             r"not (?:a|an|the) [a-z ]{3,25} but (?:a|an|the)"),
]

total = 0
for name in ("manuscript.tex", "supplementary.tex"):
    t = io.open(os.path.join(ROOT, "paper", name), encoding="utf-8").read()
    print("=" * 92)
    print(name)
    hit_here = 0
    for tag, pat in PATTERNS:
        hits = list(re.finditer(pat, t))
        if not hits:
            continue
        hit_here += len(hits)
        print("  %-28s %d" % (tag, len(hits)))
        for m in hits[:5]:
            a, b = max(0, m.start() - 85), min(len(t), m.end() + 85)
            print("      ..." + re.sub(r"\s+", " ", t[a:b]) + "...")
    if not hit_here:
        print("  clean")
    total += hit_here
print("=" * 92)
print("flagged: %d (read the context; headings and withdrawn-row dashes are legitimate)" % total)
