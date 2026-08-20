"""Audit the supplementary captions and remaining content gaps."""
import re, os
P = os.path.join(os.path.dirname(__file__), "..", "paper")


def strip(t):
    t = re.sub(r"\\citep?\[[^\]]*\]\{[^}]*\}", "", t)
    t = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$\\~&]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


s = open(os.path.join(P, "supplementary.tex"), encoding="utf-8").read()
caps = re.findall(r"\\caption\{(.*?)\}\s*\n\\label\{([^}]*)\}", s, re.S)
print(f"SI captions: {len(caps)}")
for c, l in caps:
    print(f"  {l:20s} {len(strip(c).split()):3d} words | bold-title={'textbf' in c[:40]} | {strip(c)[:60]}...")

m = open(os.path.join(P, "manuscript.tex"), encoding="utf-8").read()
res = m.split("section*{Results}")[1]
lead = res.split("\\subsection*")[0]
print(f"\nResults lead paragraph before first subheading: {len(strip(lead).split())} words")
print("  (Nature Communications papers open Results with an unnumbered lead paragraph)")

print("\nResults subheadings:")
for h in re.findall(r"\\subsection\*\{([^}]*)\}", m):
    print(f"  - {h}")
