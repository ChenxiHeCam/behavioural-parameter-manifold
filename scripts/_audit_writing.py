"""Quantify the manuscript's writing characteristics for comparison against
published Nature Communications papers."""
import re, os

P = os.path.join(os.path.dirname(__file__), "..", "paper", "manuscript.tex")
s = open(P, encoding="utf-8").read()


def clean(t):
    t = re.sub(r"%.*", "", t)
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", t, flags=re.S)
    t = re.sub(r"\\citep?\[[^\]]*\]\{[^}]*\}", "", t)
    t = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$\\~&]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


print("=== OUR MANUSCRIPT ===\n")

heads = re.findall(r"\\subsection\*\{([^}]*)\}", s)
print(f"Results subheadings ({len(heads)}):")
for h in heads:
    print(f"   - {h}")

res = s.split("section*{Results}")[1].split("section*{Discussion}")[0]
chunks = [c for c in re.split(r"\\subsection\*\{[^}]*\}", res) if len(clean(c)) > 100]
print(f"\nOpening sentence of each Results section:")
for c in chunks:
    print(f"   - {clean(c).split('.')[0][:100]}...")

print(f"\nInline statistics usage:")
for pat, name in [(r"p\s*\{?=\}?\s*[\d.]", "p values"),
                  (r"rho\s*\{?=\}?", "Spearman rho"),
                  (r"95\\?%\s*(interval|CI)", "95% interval"),
                  (r"\bn\s*\{?=\}?\s*\d", "n=")]:
    print(f"   {name}: {len(re.findall(pat, s))}")

caps = re.findall(r"\\caption\{(.*?)\}\s*\n\\label", s, re.S)
print(f"\nFigure captions: {len(caps)}, word counts = {[len(clean(c).split()) for c in caps]}")
if caps:
    print(f"   first caption opens: {clean(caps[0])[:120]}...")

disc = clean(s.split("section*{Discussion}")[1].split("paragraph{Limitations}")[0])
print(f"\nDiscussion: {len(disc.split())} words")
print(f"   opens: {disc.split('.')[0][:120]}...")
lim = clean(s.split("paragraph{Limitations}")[1].split("section*{Methods}")[0])
print(f"Limitations: dedicated paragraph, {len(lim.split())} words")

body = clean(s.split("Introduction}")[1].split("\\section*{Methods}")[0])
print(f"\nMain text total: {len(body.split())} words")
print(f"Past-tense 'we' constructions: {len(re.findall(r'[Ww]e (asked|measured|quantified|tested|computed|found|applied|used|report)', s))}")
