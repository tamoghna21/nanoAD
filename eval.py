"""
eval.py -- the file that is supposed to make you suspicious of your own results.

Everyone reports ADE (average displacement error) and FDE (final displacement
error). Both are dominated by the longitudinal axis, because in 4 s a car
travels ~60 m forward and ~2 m sideways. So a model that nails the speed
profile and steers dead straight gets a respectable ADE while being completely
unable to drive.

So we always report:

    lat   lateral error   -- did it understand the road geometry?
    lon   longitudinal    -- did it understand the speed profile?

and we always print the blind baselines next to the model. If your ADE looks
good but `lat` is no better than EgoOnly, your camera branch is decorative.
"""

import torch

HORIZONS = [1.0, 2.0, 3.0, 4.0]      # seconds, assuming 0.5 s waypoint spacing


@torch.no_grad()
def evaluate(net, loader, device, dt=0.5):
    net.eval()
    n, sums = 0, torch.zeros(4, device=device)          # ade, fde, lat, lon
    per_h = {h: torch.zeros(2, device=device) for h in HORIZONS}

    for img, ego, wps in loader:
        img, ego, wps = img.to(device), ego.to(device), wps.to(device)
        pred = net(img, ego).float()
        err = pred - wps                                 # (B, T, 2) metres
        dist = err.norm(dim=-1)                          # (B, T)
        b = img.shape[0]

        sums[0] += dist.mean(dim=1).sum()                # ADE
        sums[1] += dist[:, -1].sum()                     # FDE
        sums[2] += err[..., 1].abs().mean(dim=1).sum()   # lateral   (y)
        sums[3] += err[..., 0].abs().mean(dim=1).sum()   # longitudinal (x)

        for h in HORIZONS:
            k = int(round(h / dt)) - 1
            if k < wps.shape[1]:
                per_h[h][0] += err[:, k, 1].abs().sum()
                per_h[h][1] += err[:, k, 0].abs().sum()
        n += b

    net.train()
    ade, fde, lat, lon = (sums / n).tolist()
    return {
        "ADE": ade, "FDE": fde, "lat": lat, "lon": lon,
        "per_h": {h: (per_h[h] / n).tolist() for h in HORIZONS},
        "n": n,
    }


def format_report(results):
    """results: {name: metrics dict} -> a table you can paste into an issue."""
    w = max(len(k) for k in results) + 1
    head = (f"\n{'model':<{w}} {'ADE':>7} {'FDE':>7} {'lat':>7} {'lon':>7}   "
            + " ".join(f"lat@{h:.0f}s".rjust(7) for h in HORIZONS))
    lines = [head, "-" * len(head)]
    for name, m in results.items():
        lines.append(
            f"{name:<{w}} {m['ADE']:7.3f} {m['FDE']:7.3f} {m['lat']:7.3f} {m['lon']:7.3f}   "
            + " ".join(f"{m['per_h'][h][0]:7.3f}" for h in HORIZONS))
    lines.append(f"(metres, lower is better; n={next(iter(results.values()))['n']})")
    return "\n".join(lines)


def compare(checkpoints, loader, device):
    """Evaluate several trained checkpoints side by side, plus the free baseline."""
    import model as M
    results = {"constvel": evaluate(M.build("constvel").to(device), loader, device)}
    for name, path in checkpoints.items():
        ck = torch.load(path, map_location=device, weights_only=False)
        a = ck["args"]
        kw = {} if a["model"] != "nanoad" else {"backbone": a["backbone"]}
        net = M.build(a["model"], **kw).to(device)
        net.load_state_dict(ck["model"])
        results[name] = evaluate(net, loader, device)
    return results


if __name__ == "__main__":
    import argparse
    from data import ToyDriveVal
    from torch.utils.data import DataLoader

    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="*", help="name=path/to/ckpt.pt")
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    loader = DataLoader(ToyDriveVal(a.n_val), batch_size=64, num_workers=4)
    ck = dict(c.split("=", 1) for c in a.ckpts)
    print(format_report(compare(ck, loader, torch.device(a.device))))
