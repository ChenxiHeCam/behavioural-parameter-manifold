"""Compose the measurement figure: the connectome-scale curvature (panels a-c)
stacked over the named-conductance and FlyGym panels (d, e).

Sources are the owner-script outputs; regenerate them first if they changed:
    python scripts/_redraw_fig2_three.py      # Fig_connectome_curvature.png  (a-c)
    python scripts/_compose_panels.py         # Fig_channels_flygym.png       (d, e)
"""
import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "paper", "figures")

top = Image.open(os.path.join(FIG, "Fig_connectome_curvature.png"))
bot = Image.open(os.path.join(FIG, "Fig_channels_flygym.png"))

W = max(top.width, bot.width)
def fit(im):
    if im.width == W:
        return im
    return im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)

top, bot = fit(top), fit(bot)
GAP = round(W * 0.012)
out = Image.new("RGB", (W, top.height + GAP + bot.height), "white")
out.paste(top, (0, 0))
out.paste(bot, (0, top.height + GAP))
p = os.path.join(FIG, "Fig_measurement.png")
out.save(p, dpi=(600, 600))
print("Fig_measurement.png  %dx%d  ratio %.2f  = %.0f mm at 300 dpi"
      % (out.width, out.height, out.width / out.height, out.width / 300 * 25.4))
