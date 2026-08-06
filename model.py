"""
model.py -- the whole planner, in one readable file.

An end-to-end driving policy is, stripped of ceremony, this:

    (what the camera sees, how fast I am going)  ->  where I will be in 4 s

That is all UniAD, VAD and the NAVSIM baselines are doing at the outermost
level. They differ in what they put in the middle (BEV features, occupancy,
detection, tracking) and in how they supervise it. Start here, then add.
"""

import torch
import torch.nn as nn

N_WAYPOINTS = 8
EGO_DIM = 2          # [speed / 10, accel / 3]


# ---------------------------------------------------------------------
#  Backbones
# ---------------------------------------------------------------------

class TinyCNN(nn.Module):
    """~300 k params. Trains from scratch on a laptop GPU in minutes.

    Five stride-2 blocks take 144x256 down to 5x8, then global-average-pool.
    """

    def __init__(self, width=32, out_dim=256):
        super().__init__()
        chans = [3, width, width * 2, width * 4, width * 4, out_dim]
        layers = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            layers += [
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).mean(dim=(2, 3))


def resnet18_backbone(pretrained=False):
    """The standard choice in the literature. Needs torchvision."""
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = resnet18(weights=weights)
    net.fc = nn.Identity()
    net.out_dim = 512
    return net


# ---------------------------------------------------------------------
#  The planner
# ---------------------------------------------------------------------

class NanoAD(nn.Module):
    """Camera + ego state -> N_WAYPOINTS future (x, y) positions in metres.

    Two design choices worth understanding, because they are the ones the
    real systems agonise over too:

    1. We predict *offsets between consecutive waypoints*, not absolute
       positions. Absolute targets grow to ~60 m at the 4 s mark while the
       first waypoint is ~5 m, so an unweighted L1 on absolute positions is
       dominated by the far future. Cumulative-sum decoding fixes the scale
       imbalance for free.

    2. The ego state enters late, by concatenation. Feed it in early and the
       network learns to solve the task from speed alone and lets the visual
       pathway rot -- the "ego status is all you need" failure mode.
    """

    def __init__(self, backbone="tiny", pretrained=False, hidden=256):
        super().__init__()
        self.visual = resnet18_backbone(pretrained) if backbone == "resnet18" else TinyCNN()
        self.ego = nn.Sequential(
            nn.Linear(EGO_DIM, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(self.visual.out_dim + 64, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, N_WAYPOINTS * 2),
        )

    def forward(self, img, ego):
        f = torch.cat([self.visual(img), self.ego(ego)], dim=1)
        deltas = self.head(f).view(-1, N_WAYPOINTS, 2)
        return torch.cumsum(deltas, dim=1)


class EgoOnly(nn.Module):
    """The baseline that is not allowed to look.

    Keep this around and report it on every plot. If your fancy vision model
    does not clearly beat a network that only knows its own speed, your fancy
    vision model has not learned to see -- it has learned to extrapolate. This
    is not a hypothetical: it is the finding of "Is Ego Status All You Need for
    Open-Loop End-to-End Autonomous Driving?" (Li et al., CVPR 2024), which
    showed several published nuScenes planners were largely doing exactly this.
    """

    def __init__(self, hidden=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(EGO_DIM, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, N_WAYPOINTS * 2),
        )

    def forward(self, img, ego):          # img accepted and ignored, on purpose
        return torch.cumsum(self.head(ego).view(-1, N_WAYPOINTS, 2), dim=1)


class ConstantVelocity(nn.Module):
    """Not learned at all. Drives straight at the current speed.

    The floor. If a trained model does not beat this, something is broken.
    """

    def __init__(self, dt=0.5):
        super().__init__()
        self.register_buffer("t", torch.arange(1, N_WAYPOINTS + 1).float() * dt)

    def forward(self, img, ego):
        v = ego[:, :1] * 10.0                                  # undo normalisation
        x = v * self.t[None, :]
        return torch.stack([x, torch.zeros_like(x)], dim=-1)


def build(name, **kw):
    return {"nanoad": NanoAD, "egoonly": EgoOnly, "constvel": ConstantVelocity}[name](**kw)


if __name__ == "__main__":
    for name, kw in [("nanoad", {}), ("nanoad", {"backbone": "resnet18"}),
                     ("egoonly", {}), ("constvel", {})]:
        m = build(name, **kw)
        n = sum(p.numel() for p in m.parameters())
        y = m(torch.randn(2, 3, 144, 256), torch.randn(2, EGO_DIM))
        print(f"{name:9s} {str(kw):32s} params={n/1e6:6.2f}M  out={tuple(y.shape)}")
