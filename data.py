"""
data.py -- where the (image, ego, trajectory) triplets come from.

Two sources:

  ToyDriveTrain / ToyDriveVal   procedural, zero download, runs anywhere
  NuScenesDrive                 the real thing, once you are ready

The toy training set is an IterableDataset that never repeats a sample. That
is a deliberate luxury of synthetic data: there is no train/val leakage and no
overfitting to memorise, so every number you see is generalisation within the
generative distribution. When you switch to nuScenes you lose this and the
usual discipline applies.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

import toyworld as T

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(img_uint8):
    """HWC uint8 -> CHW float, ImageNet-normalised."""
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class ToyDriveTrain(IterableDataset):
    """An endless stream of freshly generated scenes."""

    def __init__(self, seed=0):
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng(self.seed * 9973 + wid)
        while True:
            img, ego, wps, _ = T.make_sample(rng)
            yield to_tensor(img), torch.from_numpy(ego), torch.from_numpy(wps)


class ToyDriveVal(Dataset):
    """A fixed set, so that metrics are comparable across runs and models."""

    def __init__(self, n=2000, seed=1234):
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)
        img, ego, wps, _ = T.make_sample(rng)
        return to_tensor(img), torch.from_numpy(ego), torch.from_numpy(wps)


# ---------------------------------------------------------------------
#  Real data
# ---------------------------------------------------------------------
#  NOTE: this adapter is written against the nuscenes-devkit API but was NOT
#  executed while writing (no dataset in the authoring sandbox). Treat it as a
#  starting point and check the first few samples with viz.py before trusting
#  it. `pip install nuscenes-devkit`, then grab the 4 GB v1.0-mini split.
# ---------------------------------------------------------------------

def _yaw(q):
    """Yaw angle from a nuScenes [w, x, y, z] quaternion."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class NuScenesDrive(Dataset):
    """CAM_FRONT + ego speed -> 4 s of future ego positions.

    nuScenes keyframes are annotated at 2 Hz, which lands exactly on the 0.5 s
    waypoint spacing the toy world uses, so nothing needs resampling. We keep
    only samples with a full 4 s of future inside the same scene.
    """

    def __init__(self, dataroot, version="v1.0-mini", n_waypoints=T.N_WAYPOINTS,
                 img_hw=(T.IMG_H, T.IMG_W)):
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.n_wp, self.img_hw, self.dataroot = n_waypoints, img_hw, dataroot

        self.index = []
        for s in self.nusc.sample:
            chain, cur = [s], s
            for _ in range(n_waypoints):
                if not cur["next"]:
                    break
                cur = self.nusc.get("sample", cur["next"])
                chain.append(cur)
            if len(chain) == n_waypoints + 1:
                self.index.append([c["token"] for c in chain])

    def _pose(self, sample_token):
        sd = self.nusc.get("sample_data", self.nusc.get("sample", sample_token)["data"]["CAM_FRONT"])
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        return np.array(ep["translation"][:2]), _yaw(ep["rotation"]), ep["timestamp"], sd["filename"]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        from PIL import Image
        chain = self.index[i]
        p0, yaw0, t0, fname = self._pose(chain[0])

        # future global positions -> current ego frame (x forward, y left)
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        R = np.array([[c, -s], [s, c]])
        wps = np.stack([R @ (self._pose(tok)[0] - p0) for tok in chain[1:]]).astype(np.float32)

        # instantaneous speed from the first 0.5 s step
        v0 = float(np.linalg.norm(wps[0]) / max((self._pose(chain[1])[2] - t0) / 1e6, 1e-3))

        img = Image.open(f"{self.dataroot}/{fname}").resize((self.img_hw[1], self.img_hw[0]))
        ego = np.array([v0 / 10.0, 0.0], np.float32)     # accel: fill in if you want it
        return to_tensor(np.asarray(img.convert("RGB"))), torch.from_numpy(ego), torch.from_numpy(wps)


def loaders(args):
    from torch.utils.data import DataLoader
    if args.data == "nuscenes":
        full = NuScenesDrive(args.dataroot)
        n_val = max(1, len(full) // 10)
        val, train = torch.utils.data.random_split(
            full, [n_val, len(full) - n_val], generator=torch.Generator().manual_seed(0))
        tl = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=True)
    else:
        tl = DataLoader(ToyDriveTrain(args.seed), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
        val = ToyDriveVal(args.n_val)
    vl = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    return tl, vl
