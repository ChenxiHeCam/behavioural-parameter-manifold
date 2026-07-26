"""Count words/items in manuscript.tex against Nature Communications limits."""
import re, sys, os
p = os.path.join(os.path.dirname(__file__), '..', 'paper', 'manuscript.tex')
s = open(p, encoding='utf-8').read()

def strip_tex(t):
    t = re.sub(r'%.*', '', t)
    t = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', t, flags=re.S)
    t = re.sub(r'\\citep?\[[^]]*\]\{[^}]*\}', '', t)
    t = re.sub(r'\\cite[a-zA-Z]*\{[^}]*\}', '', t)
    t = re.sub(r'\\[a-zA-Z]+\*?', ' ', t)
    t = re.sub(r'[{}$\\~&]', ' ', t)
    return t

ab = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', s, re.S).group(1)
print(f"Abstract words   : {len(strip_tex(ab).split()):5d}   (NC: 150-200)")
print(f"Abstract has cites: {'YES - must remove' if 'cite' in ab else 'no'}")

ti = re.search(r'\\title\{(.*?)\}\s*\n', s, re.S).group(1)
print(f"Title words      : {len(strip_tex(ti).split()):5d}   (NC: <15)")

body = s.split('Introduction}')[1].split('\\section*{Methods}')[0]
print(f"Main text words  : {len(strip_tex(body).split()):5d}   (NC: <5000)")

meth = s.split('\\section*{Methods}')[1].split('\\section*{Data availability}')[0]
print(f"Methods words    : {len(strip_tex(meth).split()):5d}   (NC: <3000)")

print(f"Display items    : {s.count('begin{figure}') + s.count('begin{table}'):5d}   (NC: <=10)")
cites = set()
for m in re.findall(r'\\cite[a-zA-Z]*\{([^}]*)\}', s):
    cites.update(x.strip() for x in m.split(','))
print(f"Unique refs(main): {len(cites):5d}   (NC guide: <=70)")
