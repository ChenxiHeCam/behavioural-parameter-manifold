"""Re-check the Nature Computational Science submission limits."""
import re, os

BS = chr(92)
m = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "manuscript.tex"),
         encoding="utf-8").read()


def strip(t):
    t = re.sub(BS + r"cite[a-zA-Z]*(\[[^\]]*\])?\{[^}]*\}", "", t)
    t = re.sub(BS + r"[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$" + BS + r"~&%]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


ab = re.search(BS + r"begin\{abstract\}(.*?)" + BS + r"end\{abstract\}", m, re.S).group(1)
body = m.split("section*{Results}")[1].split(BS + "bibliography")[0]
groups = re.findall(BS + r"cite[a-zA-Z]*(?:\[[^\]]*\])?\{([^}]*)\}", m)
keys = {k.strip() for g in groups for k in g.split(",")}
disp = len(re.findall(BS + r"begin\{figure", m)) + len(re.findall(BS + r"begin\{table", m))

print(f"abstract   {len(strip(ab).split()):5d} w   (NCS limit 150)")
print(f"main text  {len(strip(body).split()):5d} w   (NCS limit ~3000)")
print(f"references {len(keys):5d}       (NCS limit 50)")
print(f"display    {disp:5d}       (NCS limit 8)")
