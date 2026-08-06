"""
drive.py -- closed-loop evaluation. Same checkpoints, a harder question.

    python drive.py runs/nanoad/ckpt.pt runs/egoonly/ckpt.pt

Everything else in this repo is open-loop: a scene is drawn, the model
predicts, the prediction is scored against the expert, the scene is thrown
away. animate.py's GIF *looks* like driving but every frame is an independent
sample -- the model steers nothing and errors never compound.

This file is the real thing. A persistent road is generated once per episode.
The model's prediction is fed through a controller into a kinematic bicycle
model, the car actually moves, and next frame's observation is a consequence
of this frame's mistake. That compounding is "covariate shift" -- the classic
failure mode of imitation learning, and the reason open-loop numbers can be
misleadingly rosy.

No model is retrained here. runs/nanoad/ckpt.pt is the exact checkpoint that
scores 0.090 m laterally in eval.py; drive.py just asks it a different
question.

Mandatory sanity check before trusting anything below: run the loop with
toyworld.expert() standing in for the network (--model expert, included by
default). The expert sees the true road and MUST complete every episode. If
it doesn't, the bug is in this file's dynamics/controller/frame projection,
not in a learned model.
"""

import argparse
import os
from types import SimpleNamespace

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

WHEELBASE = 2.7
DELTA_MAX = 0.5           # rad
A_MIN, A_MAX = -3.0, 2.0  # m/s^2


# ---------------------------------------------------------------------
#  The persistent road: a curvature profile, integrated once per episode.
# ---------------------------------------------------------------------

def build_road(rng, ds=0.25, s_max=1300.0):
    """Sum of three sinusoids in kappa(s), amplitude-matched to sample_scene's
    training range (|kappa| up to ~0.012) so the closed loop stays in-
    distribution -- we want to measure covariate shift from *driving*, not
    from handing the model curvature it never saw in training.
    """
    n = int(s_max / ds) + 1
    s = np.arange(n) * ds
    ph = rng.uniform(0, 2 * np.pi, 3)
    kappa = (0.006 * np.sin(2 * np.pi * s / 220 + ph[0])
             + 0.004 * np.sin(2 * np.pi * s / 130 + ph[1])
             + 0.003 * np.sin(2 * np.pi * s / 70 + ph[2]))
    psi = np.concatenate(([0.0], np.cumsum(kappa[:-1]) * ds))
    X = np.concatenate(([0.0], np.cumsum(np.cos(psi[:-1])) * ds))
    Y = np.concatenate(([0.0], np.cumsum(np.sin(psi[:-1])) * ds))
    return SimpleNamespace(s=s, ds=ds, kappa=kappa, psi=psi, X=X, Y=Y)


def nearest_s_idx(road, pos, guess_idx, window_m=15.0):
    """Windowed nearest-point search. The car can't teleport, so a small
    window around last step's index is both correct and cheap."""
    half = int(window_m / road.ds)
    lo, hi = max(0, guess_idx - half), min(len(road.s), guess_idx + half + 1)
    dx, dy = road.X[lo:hi] - pos[0], road.Y[lo:hi] - pos[1]
    return lo + int(np.argmin(dx * dx + dy * dy))


def offset_curve(road, offset):
    """Points at constant lateral `offset` (metres, +left) from the centreline."""
    return road.X - offset * np.sin(road.psi), road.Y + offset * np.cos(road.psi)


def on_road(y0):
    """Same two-lane geometry render() draws: far/left kerb at +1.5 lane
    widths, near/right kerb at -0.5 (ego drives in the right lane). Asymmetric
    on purpose -- see toyworld.render()'s comment on u_l/u_r."""
    return (y0 - 0.5 * T.LANE_WIDTH <= 0.0) and (y0 + 1.5 * T.LANE_WIDTH >= 0.0)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def path_length(trace):
    if len(trace) < 2:
        return 0.0
    d = np.diff(np.asarray(trace), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


# ---------------------------------------------------------------------
#  Controller: waypoints -> (steering, acceleration)
# ---------------------------------------------------------------------

def controller(traj, v):
    """Pure pursuit for steering, trajectory spacing for speed. traj is (8,2)
    in the ego frame (x forward, y left), same convention as the expert."""
    pts = np.vstack([[0.0, 0.0], traj])
    seg = np.diff(pts, axis=0)
    cum = np.concatenate(([0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))))

    Ld = float(np.clip(1.2 * v, 4.0, 15.0))
    look_at = min(Ld, cum[-1]) if cum[-1] > 0 else 0.0
    y_look = np.interp(look_at, cum, pts[:, 1])
    delta = float(np.clip(np.arctan2(2 * WHEELBASE * y_look, Ld ** 2), -DELTA_MAX, DELTA_MAX))

    v_target = float(np.clip(traj[0, 0] / 0.5, 0.0, T.V_MAX))
    a = float(np.clip((v_target - v) / 0.5, A_MIN, A_MAX))
    return delta, a


# ---------------------------------------------------------------------
#  One closed-loop episode
# ---------------------------------------------------------------------

def get_traj(name, net, scene, device):
    img = T.render(scene, rng=None)
    if name == "expert":
        wps, _ = T.expert(scene)
        return wps, img
    ego = np.array([scene["v0"] / 10.0, scene["a0"] / 3.0], np.float32)
    with torch.no_grad():
        x = to_tensor(img)[None].to(device)
        e = torch.from_numpy(ego)[None].to(device)
        traj = net(x, e)[0].float().cpu().numpy()
    return traj, img


def rollout(name, net, seed, device, dt=T.DT, max_time=60.0, record=False, stride=3):
    rng = np.random.default_rng(seed)
    road = build_road(rng)

    y0_init = float(rng.uniform(-1.1, 1.1))
    heading = float(rng.uniform(-0.06, 0.06))          # psi_init
    v, a = float(rng.uniform(4.0, 12.0)), 0.0
    pos = np.array([0.0, -y0_init])                    # so that y0 == y0_init at s=0
    dash_phase = float(rng.uniform(0.0, 6.0))
    guess_idx = 0

    trace, lat_offsets, frames = [pos.copy()], [], ([] if record else None)
    n_steps = int(round(max_time / dt))
    departed, t_end = False, max_time

    for step in range(n_steps):
        idx = nearest_s_idx(road, pos, guess_idx)
        guess_idx = idx
        dx, dy = road.X[idx] - pos[0], road.Y[idx] - pos[1]
        y0 = -np.sin(heading) * dx + np.cos(heading) * dy
        psi = wrap(road.psi[idx] - heading)          # road heading in the ego frame

        if not on_road(y0):
            departed, t_end = True, step * dt
            if record:
                scene = dict(y0=y0, psi=psi, kappa=road.kappa[idx], dash_phase=dash_phase,
                             lead_x=np.inf, sun=1.0, v0=v, a0=a)
                frames.append(dict(img=T.render(scene, rng=None), traj=None, trace=np.array(trace),
                                    t=t_end, v=v, y0=y0, departed=True))
            break

        lat_offsets.append(y0)
        scene = dict(y0=y0, psi=psi, kappa=road.kappa[idx], dash_phase=dash_phase,
                     lead_x=np.inf, sun=1.0, v0=v, a0=a)
        traj, img = get_traj(name, net, scene, device)
        delta, a_cmd = controller(traj, v)

        if record and step % stride == 0:
            frames.append(dict(img=img, traj=traj, trace=np.array(trace),
                                t=step * dt, v=v, y0=y0, departed=False))

        pos = pos + v * dt * np.array([np.cos(heading), np.sin(heading)])
        heading = heading + (v / WHEELBASE) * np.tan(delta) * dt
        v = max(0.0, v + a_cmd * dt)
        a = a_cmd
        dash_phase = (dash_phase + v * dt) % 6.0
        trace.append(pos.copy())
    else:
        pass  # loop completed without break -> reached timeout

    metrics = dict(
        success=not departed, t=t_end, dist=path_length(trace),
        mean_abs_y0=float(np.mean(np.abs(lat_offsets))) if lat_offsets else float("nan"),
        max_abs_y0=float(np.max(np.abs(lat_offsets))) if lat_offsets else float("nan"),
    )
    return metrics, road, frames


# ---------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------

def summarize(name, episodes):
    succ = [e["success"] for e in episodes]
    dist = [e["dist"] for e in episodes]
    t = [e["t"] for e in episodes]
    lat = [e["mean_abs_y0"] for e in episodes if not np.isnan(e["mean_abs_y0"])]
    mx = [e["max_abs_y0"] for e in episodes if not np.isnan(e["max_abs_y0"])]
    return dict(name=name, success_rate=100.0 * sum(succ) / len(succ),
                mean_dist=np.mean(dist), median_dist=np.median(dist), mean_t=np.mean(t),
                mean_lat=np.mean(lat) if lat else float("nan"),
                max_lat=np.max(mx) if mx else float("nan"))


def print_table(rows):
    head = (f"\n{'model':<9} {'success':>8} {'mean dist':>10} {'median dist':>12} "
            f"{'mean t':>8} {'mean |y0|':>10} {'max |y0|':>9}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['name']:<9} {r['success_rate']:7.1f}% {r['mean_dist']:9.1f}m "
              f"{r['median_dist']:11.1f}m {r['mean_t']:7.1f}s {r['mean_lat']:9.3f}m "
              f"{r['max_lat']:8.3f}m")
    print("(dist/t = distance/time survived per episode, capped at timeout; "
          "|y0| = lateral offset from lane centre while on-road)")


# ---------------------------------------------------------------------
#  The closed-loop GIF: camera view + top-down traced path
# ---------------------------------------------------------------------

def render_gif(frames, road, name, out, fps, colors, window_m=30.0):
    left, right = offset_curve(road, 1.5 * T.LANE_WIDTH), offset_curve(road, -0.5 * T.LANE_WIDTH)

    fig, (ax_img, ax_map) = plt.subplots(
        1, 2, figsize=(8.0, 2.9), dpi=90, gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor("#12141a")
    color = COLORS.get(name, "w")

    gif_frames = []
    for f in frames:
        ax_img.clear()
        ax_img.imshow(f["img"])
        if f["traj"] is not None:
            uv = project(f["traj"])
            if len(uv):
                ax_img.plot(uv[:, 0], uv[:, 1], color=color, lw=2.4, alpha=0.95)
        ax_img.set_xlim(0, T.IMG_W); ax_img.set_ylim(T.IMG_H, 0); ax_img.axis("off")
        title = f"{name}   t={f['t']:.1f}s   v={f['v']:.1f} m/s   |y0|={abs(f['y0']):.2f} m"
        if f["departed"]:
            title = f"{name}   LEFT ROAD at t={f['t']:.1f}s"
        ax_img.set_title(title, color=("#ff5a5a" if f["departed"] else "#c8ccd6"),
                         fontsize=8, loc="left", pad=6)

        ax_map.clear()
        ax_map.set_facecolor("#181b22")
        px, py = f["trace"][-1]
        near = (np.abs(road.X - px) < window_m + 10) & (np.abs(road.Y - py) < window_m + 10)
        ax_map.plot(road.X[near], road.Y[near], "--", color="#3a4050", lw=1.0)
        ax_map.plot(left[0][near], left[1][near], "-", color="#5a6070", lw=1.0)
        ax_map.plot(right[0][near], right[1][near], "-", color="#5a6070", lw=1.0)
        ax_map.plot(f["trace"][:, 0], f["trace"][:, 1], "-", color=color, lw=2.0)
        ax_map.plot([px], [py], "o", color=color, ms=5)
        ax_map.set_xlim(px - window_m, px + window_m)
        ax_map.set_ylim(py - window_m, py + window_m)
        ax_map.set_aspect("equal")
        ax_map.tick_params(colors="#5a6070", labelsize=6)
        for sp in ax_map.spines.values():
            sp.set_color("#2a2f3a")

        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        gif_frames.append(Image.fromarray(rgba[..., :3]).quantize(colors=colors, method=2))

    gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    print(f"wrote {out}  ({len(gif_frames)} frames, {os.path.getsize(out)/1e6:.1f} MB)")


# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="*", help="path(s) to ckpt.pt, e.g. runs/nanoad/ckpt.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-time", type=float, default=60.0)
    p.add_argument("--gif-model", default="nanoad")
    p.add_argument("--gif-episode", type=int, default=0)
    p.add_argument("--out", default="docs/closed_loop.gif")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--colors", type=int, default=96)
    p.add_argument("--stride", type=int, default=3, help="physics steps per GIF frame")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    dev = torch.device(a.device)
    nets = dict(load(c, dev) for c in a.ckpts)
    nets.setdefault("constvel", M.build("constvel").to(dev))
    names = ["expert"] + [n for n in nets]           # expert first: the control experiment

    rows, gif_frames, gif_road = [], None, None
    for name in names:
        net = nets.get(name)
        episodes = []
        for ep in range(a.episodes):
            record = (name == a.gif_model and ep == a.gif_episode)
            metrics, road, frames = rollout(name, net, a.seed + ep, dev,
                                            max_time=a.max_time, record=record, stride=a.stride)
            episodes.append(metrics)
            if record:
                gif_frames, gif_road = frames, road
        rows.append(summarize(name, episodes))
        if name == "expert" and rows[0]["success_rate"] < 100.0:
            print(f"\n*** WARNING: the EXPERT control experiment did not reach 100% "
                  f"success ({rows[0]['success_rate']:.0f}%). This means drive.py's dynamics, "
                  f"controller, or frame projection has a bug -- the model results below "
                  f"cannot be trusted until this is fixed. ***\n")

    print_table(rows)

    if gif_frames:
        render_gif(gif_frames, gif_road, a.gif_model, a.out, a.fps, a.colors)
    else:
        print(f"\n(no GIF written -- --gif-model {a.gif_model!r} was not among "
              f"{names}, or --gif-episode {a.gif_episode} >= --episodes {a.episodes})")


if __name__ == "__main__":
    main()
