"""
viz.py -- look at your predictions. Metrics lie; pictures lie less.

    python viz.py runs/nanoad/ckpt.pt --out preds.png
    python viz.py runs/nanoad/ckpt.pt runs/egoonly/ckpt.pt --out compare.png

Left panel: the camera view with the trajectory projected back onto the road
plane. Right panel: bird's-eye view. The projection uses the same pinhole
model the renderer used, so if the overlay does not land on the road you have
a coordinate-frame bug -- which is the single most common bug in this field.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import model as M
import toyworld as T
from data import to_tensor

COLORS = {"expert": "#00d68f", "nanoad": "#ff8c42", "egoonly": "#e04b6e", "constvel": "#8a8fa3"}


def project(traj, n_interp=60):
    """Ground-plane trajectory (T, 2) in metres -> image pixels (N, 2)."""
    x, y = traj[:, 0], traj[:, 1]
    s = np.linspace(0, 1, n_interp)
    xs = np.interp(s, np.linspace(0, 1, len(x)), x)
    ys = np.interp(s, np.linspace(0, 1, len(y)), y)
    keep = xs > 3.0                                   # behind/too near the camera
    xs, ys = xs[keep], ys[keep]
    u = T.CX - T.FOCAL * ys / xs
    v = T.CY + T.FOCAL * T.CAM_HEIGHT / xs
    return np.stack([u, v], axis=1)


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    kw = {} if a["model"] != "nanoad" else {"backbone": a["backbone"]}
    net = M.build(a["model"], **kw).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return a["model"], net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="*")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="preds.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    dev = torch.device(a.device)
    nets = dict(load(c, dev) for c in a.ckpts)
    nets.setdefault("constvel", M.build("constvel").to(dev))

    rng = np.random.default_rng(a.seed)
    fig, axes = plt.subplots(a.n, 2, figsize=(11, 2.7 * a.n),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    fig.patch.set_facecolor("#12141a")

    for r in range(a.n):
        img, ego, wps, _ = T.make_sample(rng)
        with torch.no_grad():
            x = to_tensor(img)[None].to(dev)
            e = torch.from_numpy(ego)[None].to(dev)
            preds = {k: n(x, e)[0].float().cpu().numpy() for k, n in nets.items()}
        preds["expert"] = wps

        ax = axes[r, 0]
        ax.imshow(img)
        for k, tr in preds.items():
            uv = project(tr)
            ax.plot(uv[:, 0], uv[:, 1], color=COLORS.get(k, "w"), lw=2.4,
                    label=k if r == 0 else None,
                    ls="--" if k == "expert" else "-", alpha=0.95)
        ax.set_xlim(0, T.IMG_W); ax.set_ylim(T.IMG_H, 0); ax.axis("off")

        ax = axes[r, 1]
        ax.set_facecolor("#181b22")
        for k, tr in preds.items():
            ax.plot(tr[:, 1], tr[:, 0], marker="o", ms=3, color=COLORS.get(k, "w"),
                    ls="--" if k == "expert" else "-")
        ax.set_xlim(8, -8); ax.set_ylim(0, 70)          # +y is LEFT, so invert x-axis
        ax.set_xlabel("y (m, left+)", color="#9aa0ad", fontsize=8)
        ax.set_ylabel("x (m fwd)", color="#9aa0ad", fontsize=8)
        ax.tick_params(colors="#5a6070", labelsize=7)
        ax.grid(alpha=0.15)
        for sp in ax.spines.values():
            sp.set_color("#2a2f3a")
        ax.text(0.03, 0.94, f"v={ego[0]*10:.1f} m/s", transform=ax.transAxes,
                color="#9aa0ad", fontsize=8, va="top")

    fig.legend(loc="upper center", ncol=4, frameon=False,
               labelcolor="#c8ccd6", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(a.out, dpi=130, facecolor=fig.get_facecolor())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
