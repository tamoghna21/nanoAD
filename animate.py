"""
animate.py -- build the animated hero image for the README.

    python animate.py runs/nanoad/ckpt.pt runs/egoonly/ckpt.pt --out docs/hero.gif

What this is: a smooth, seamlessly looping sweep through the toy world's scene
parameters -- curvature bends left, straightens, bends right; a lead vehicle
approaches and recedes; speed rises and falls. Every frame is an independent
forward pass on a fresh scene.

What this is NOT: closed-loop driving. The ego vehicle is not being steered by
the model, and the scene does not respond to the predictions. nanoAD is an
open-loop planner and this animation is honest about that -- it shows what each
model *predicts*, frame by frame, not what would happen if you let it drive.
Making the world reactive is exercise 4 in the README.

Watch the orange (nanoad) line track the bend while pink (egoonly) and grey
(constvel) drive straight off the road. That divergence is the whole repo.

One caveat worth stating: this sweep spends far more time on curved road than
the random scene distribution does (`sample_scene` draws straight road 35% of
the time). So the on-screen lateral errors run larger than the numbers in the
README's evaluation table -- on curved frames, roughly 0.6 m for nanoad against
8.6 m for the blind models. The animation is chosen to be legible, not
representative. The table is the honest measurement; this is the trailer.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import model as M
import toyworld as T
from data import to_tensor
from viz import COLORS, load, project


def sweep(t):
    """Scene parameters at loop position t in [0, 1).

    Every term is periodic in t so the GIF loops without a seam. The dash
    phase advances 36 m over the loop (a multiple of the 6 m dash period),
    which is what sells the illusion of forward motion.
    """
    ph = 2 * np.pi * t
    return dict(
        kappa=0.011 * np.sin(2 * ph),              # left bend -> straight -> right bend
        y0=0.7 * np.sin(3 * ph),                   # weaving within the lane
        psi=0.030 * np.sin(2 * ph + 0.6),
        v0=9.0 + 5.0 * np.sin(ph + 1.2),
        a0=0.0,
        lead_x=60.0 - 45.0 * max(0.0, np.sin(ph)), # a car closes in, then drops back
        dash_phase=(36.0 * t) % 6.0,
        sun=1.0,                                   # constant: brightness flicker looks awful
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="*", help="path(s) to ckpt.pt")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--colors", type=int, default=96, help="GIF palette size; lower = smaller file")
    p.add_argument("--out", default="docs/hero.gif")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    dev = torch.device(a.device)
    nets = dict(load(c, dev) for c in a.ckpts)
    nets.setdefault("constvel", M.build("constvel").to(dev))

    fig, (ax_img, ax_bev) = plt.subplots(
        1, 2, figsize=(8.0, 2.8), dpi=90, gridspec_kw={"width_ratios": [1.75, 1]})
    fig.patch.set_facecolor("#12141a")

    frames = []
    for i in range(a.frames):
        scene = sweep(i / a.frames)
        img = T.render(scene, rng=None)
        wps, _ = T.expert(scene)
        ego = np.array([scene["v0"] / 10.0, scene["a0"] / 3.0], np.float32)

        with torch.no_grad():
            x = to_tensor(img)[None].to(dev)
            e = torch.from_numpy(ego)[None].to(dev)
            preds = {k: n(x, e)[0].float().cpu().numpy() for k, n in nets.items()}
        preds["expert"] = wps

        ax_img.clear()
        ax_img.imshow(img)
        for k, tr in preds.items():
            uv = project(tr)
            if len(uv):
                ax_img.plot(uv[:, 0], uv[:, 1], color=COLORS.get(k, "w"), lw=2.4,
                            ls="--" if k == "expert" else "-", alpha=0.95)
        ax_img.set_xlim(0, T.IMG_W)
        ax_img.set_ylim(T.IMG_H, 0)
        ax_img.axis("off")

        ax_bev.clear()
        ax_bev.set_facecolor("#181b22")
        for k, tr in preds.items():
            ax_bev.plot(tr[:, 1], tr[:, 0], marker="o", ms=3, color=COLORS.get(k, "w"),
                        ls="--" if k == "expert" else "-")
        ax_bev.set_xlim(9, -9)                       # +y is LEFT, so invert
        ax_bev.set_ylim(0, 70)
        ax_bev.set_xlabel("y (m, left+)", color="#9aa0ad", fontsize=7)
        ax_bev.set_ylabel("x (m fwd)", color="#9aa0ad", fontsize=7)
        ax_bev.tick_params(colors="#5a6070", labelsize=6)
        ax_bev.grid(alpha=0.15)
        for sp in ax_bev.spines.values():
            sp.set_color("#2a2f3a")

        # live lateral error at the 4 s waypoint -- the number the repo is about
        bits = [f"{k} {abs(preds[k][-1, 1] - wps[-1, 1]):.2f} m"
                for k in ("nanoad", "egoonly", "constvel") if k in preds]
        ax_img.set_title("lateral error @4s:   " + "   |   ".join(bits),
                         color="#c8ccd6", fontsize=8, loc="left", pad=6)

        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[..., :3]).quantize(colors=a.colors, method=2))
        if (i + 1) % 12 == 0:
            print(f"  frame {i+1}/{a.frames}")

    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / a.fps), loop=0, optimize=True, disposal=2)
    import os
    print(f"wrote {a.out}  ({a.frames} frames, {os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
