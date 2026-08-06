"""
train.py -- imitation learning, the whole thing.

    python train.py --compare                # BOTH models + the free floor, one table
    python train.py                          # just nanoad (tiny CNN, toy world)
    python train.py --model egoonly          # just the blind baseline
    python train.py --backbone resnet18 --pretrained
    python train.py --data nuscenes --dataroot /path/to/nuscenes

--compare is the one you want. It trains nanoad and egoonly on the *same* data
stream, evaluates both plus the zero-parameter constvel floor on the same
fixed-seed validation set, and prints the comparison that this repo exists to
make. Roughly 9 minutes end to end on a 4 GB laptop GPU.

Memory: the default fits in well under 2 GB of VRAM. resnet18 at batch 64 sits
around 3 GB. If you are on a 4 GB card and it complains, halve --batch-size;
AMP (on by default when CUDA is present) roughly halves activation memory.
Every run prints its own measured peak, so you never have to trust that claim.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import model as M
from data import loaders
from eval import evaluate, format_report


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", action="store_true",
                   help="train nanoad AND egoonly, then report both against constvel")
    p.add_argument("--model", default="nanoad", choices=["nanoad", "egoonly", "constvel"])
    p.add_argument("--backbone", default="tiny", choices=["tiny", "resnet18"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--data", default="toy", choices=["toy", "nuscenes"])
    p.add_argument("--dataroot", default="./data/nuscenes")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/nanoad",
                   help="checkpoint dir; in --compare mode this is the PARENT dir")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def lr_at(step, args):
    """Linear warmup then cosine decay to 10% -- the boring choice that works."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def waypoint_loss(pred, target):
    """Smooth-L1 in metres.

    Note what we are NOT doing: no per-horizon weighting, no separate lateral
    term. The cumulative-sum decoding in the model already balances near and
    far waypoints. Add weighting only after you have measured that you need it.
    """
    return F.smooth_l1_loss(pred, target, beta=0.5)


def train_one(args, model_name, out_dir, train_loader, val_loader, dev, amp):
    """Train a single model. Returns (metrics, stats)."""
    torch.manual_seed(args.seed)            # same init/data order for every model
    out_dir.mkdir(parents=True, exist_ok=True)

    kw = {"backbone": args.backbone, "pretrained": args.pretrained} if model_name == "nanoad" else {}
    net = M.build(model_name, **kw).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    label = f"{model_name} ({args.backbone})" if model_name == "nanoad" else model_name
    print(f"\n=== {label}: {n_params/1e6:.2f}M params on {dev} ===")

    if n_params == 0:                       # constvel has nothing to learn
        return evaluate(net, val_loader, dev), {"params": 0, "minutes": 0.0,
                                                "it_s": float("nan"), "loss": float("nan"),
                                                "peak_gb": 0.0}

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    it, t0, run_loss, last = iter(train_loader), time.time(), 0.0, float("nan")
    for step in range(args.steps):
        try:
            img, ego, wps = next(it)
        except StopIteration:               # map-style loaders run dry; restart
            it = iter(train_loader)
            img, ego, wps = next(it)

        img, ego, wps = img.to(dev, non_blocking=True), ego.to(dev), wps.to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)

        with torch.amp.autocast("cuda", enabled=amp):
            loss = waypoint_loss(net(img, ego), wps)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        run_loss += loss.item()
        if (step + 1) % 100 == 0:
            last = run_loss / 100
            print(f"step {step+1:6d}/{args.steps}  loss {last:.4f}  "
                  f"lr {lr_at(step, args):.2e}  {(step+1)/(time.time()-t0):.1f} it/s")
            run_loss = 0.0

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics = evaluate(net, val_loader, dev)
            print(format_report({model_name: metrics}))
            saved_args = {**vars(args), "model": model_name}
            torch.save({"model": net.state_dict(), "args": saved_args}, out_dir / "ckpt.pt")

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0.0
    stats = {"params": n_params, "minutes": elapsed / 60, "it_s": args.steps / elapsed,
             "loss": last, "peak_gb": peak}
    print(f"saved -> {out_dir/'ckpt.pt'}   "
          f"[{stats['minutes']:.1f} min, {stats['it_s']:.1f} it/s, "
          f"peak VRAM {peak:.2f} GB]")
    return metrics, stats


def headline(results):
    """Print the ratios the README is built on.

    Deliberately prints ADE alongside lateral rather than instead of it. The
    interesting quantity is the *difference* between the two ratios: ADE is
    dominated by the longitudinal axis, so it systematically understates how
    much better a model that can actually see the road is.
    """
    if not {"nanoad", "egoonly"} <= set(results):
        return ""
    n, e = results["nanoad"], results["egoonly"]
    lines = ["", "headline (x = times better; >1 means the first model wins)", "-" * 70,
             f"  nanoad  vs egoonly   ADE {e['ADE']/n['ADE']:6.1f}x"
             f"   lateral {e['lat']/n['lat']:6.1f}x"
             f"   lateral@4s {e['per_h'][4.0][0]/n['per_h'][4.0][0]:6.1f}x"]

    if "constvel" in results:
        c = results["constvel"]
        ade_r, lat_r = c["ADE"] / e["ADE"], c["lat"] / e["lat"]
        lines.append(f"  egoonly vs constvel  ADE {ade_r:6.1f}x   lateral {lat_r:6.2f}x")
        # Only assert the interpretation when the numbers actually show it.
        if ade_r > 1.2 and lat_r < 1.05:
            lines.append(f"    -> the blind model looks {ade_r:.1f}x better than a zero-parameter"
                         f" straight line on ADE,")
            lines.append(f"       but laterally it is {abs(c['lat']-e['lat'])*1000:.0f} mm from it."
                         f" Its entire ADE gain is speed, not steering.")
        elif ade_r <= 1.2:
            lines.append("    -> egoonly has not separated from constvel yet; train longer.")
    lines.append("-" * 70)
    return "\n".join(lines)


def main():
    args = parse()
    dev = torch.device(args.device)
    amp = (not args.no_amp) and dev.type == "cuda"
    train_loader, val_loader = loaders(args)

    if not args.compare:
        train_one(args, args.model, Path(args.out), train_loader, val_loader, dev, amp)
        return

    # In compare mode --out is a parent directory. Map the single-run default
    # "runs/nanoad" back to "runs" so the flag works with no other arguments.
    base = Path(args.out)
    if base.name in ("nanoad", "egoonly", "constvel"):
        base = base.parent

    results, stats = {}, {}
    for name in ("nanoad", "egoonly", "constvel"):
        results[name], stats[name] = train_one(
            args, name, base / name, train_loader, val_loader, dev, amp)

    print("\n" + "=" * 62)
    print("COMPARISON" + (f"  ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else "  (CPU)"))
    print("=" * 62)
    print(format_report(results))
    print(headline(results))
    print(f"\n{'model':<10} {'params':>9} {'minutes':>8} {'it/s':>7} {'peak VRAM':>10}")
    for name, s in stats.items():
        if s["params"] == 0:                       # constvel is never trained
            print(f"{name:<10} {0:8.2f}M {'-':>8} {'-':>7} {'-':>10}")
            continue
        peak = f"{s['peak_gb']:.2f} GB" if dev.type == "cuda" else "n/a"
        print(f"{name:<10} {s['params']/1e6:8.2f}M {s['minutes']:8.1f} "
              f"{s['it_s']:7.1f} {peak:>10}")
    print(f"\ncheckpoints -> {base}/<model>/ckpt.pt")


if __name__ == "__main__":
    main()
