"""
toyworld.py -- a 200-line procedural driving world.

Why this exists: every end-to-end driving repo makes you download 300 GB before
you can see a single gradient step. This file generates (image, ego_state,
expert_trajectory) triplets on the fly, in ~2 ms each, with no dataset at all.

The world is deliberately simple but NOT trivial:
  - the road has curvature that you can only perceive from the image
  - the target speed depends on curvature and on a lead vehicle
  - the expert converges back to lane centre over ~2 s rather than teleporting

That last point matters. It means a model that ignores the camera and just
extrapolates the current speed can still score well on average displacement
error -- which is exactly the pathology that plagues real open-loop driving
benchmarks. See eval.py, where we measure lateral and longitudinal error
separately so the cheat is visible.

Coordinate frame (ISO 8855, the one the AD world uses):
    x = forward, y = LEFT, z = up. Ego is at the origin, heading +x.
"""

import numpy as np

# --- camera intrinsics / extrinsics --------------------------------------
IMG_H, IMG_W = 144, 256
FOCAL = 250.0
CX, CY = IMG_W / 2.0, 50.0      # CY is the horizon row
CAM_HEIGHT = 1.6                # metres above the road plane

# --- world constants ------------------------------------------------------
LANE_WIDTH = 3.6
V_MAX = 16.0                    # m/s, ~58 km/h
HORIZON_S = 4.0                 # seconds of future we predict
N_WAYPOINTS = 8                 # one every 0.5 s
DT = 0.1                        # integration step for the expert

# colours (B, G, R order does not matter here -- we keep RGB)
C_SKY = np.array([135, 170, 210], np.float32)
C_GRASS = np.array([ 70, 105,  60], np.float32)
C_ROAD = np.array([ 78,  78,  82], np.float32)
C_LINE = np.array([225, 225, 210], np.float32)
C_EDGE = np.array([200, 200, 195], np.float32)
C_CAR = np.array([160,  55,  50], np.float32)


def centreline(x, y0, psi, kappa):
    """Lateral position of the lane centre at longitudinal distance x.

    A clothoid is the honest model; a quadratic is the useful one.
    y0   -- lateral offset of the lane centre at the ego (metres, +left)
    psi  -- heading error (radians)
    kappa-- curvature (1/m)
    """
    return y0 + np.tan(psi) * x + 0.5 * kappa * x ** 2


def sample_scene(rng):
    """Draw the latent parameters of one scene."""
    return dict(
        y0=rng.uniform(-1.1, 1.1),                 # off-centre in the lane
        psi=rng.uniform(-0.06, 0.06),              # heading error
        kappa=rng.choice([0.0, 1.0], p=[0.35, 0.65]) * rng.uniform(-0.012, 0.012),
        v0=rng.uniform(2.0, V_MAX),                # current speed
        a0=rng.uniform(-1.5, 1.5),                 # current acceleration
        dash_phase=rng.uniform(0, 6.0),
        lead_x=rng.choice([np.inf, 1.0], p=[0.5, 0.5]) * rng.uniform(8.0, 55.0),
        sun=rng.uniform(0.75, 1.25),               # global brightness jitter
    )


# ------------------------------------------------------------------------
#  Rendering
# ------------------------------------------------------------------------

def render(scene, rng=None):
    """Project the scene into a forward-facing camera image (H, W, 3) uint8.

    We iterate over image ROWS rather than over world distance. For a flat
    ground plane each row below the horizon maps to exactly one distance,
        v = CY + FOCAL * CAM_HEIGHT / x   ->   x = FOCAL * CAM_HEIGHT / (v - CY)
    which guarantees we never leave a gap between scanlines.
    """
    img = np.empty((IMG_H, IMG_W, 3), np.float32)
    img[:int(CY)] = C_SKY
    img[int(CY):] = C_GRASS

    y0, psi, kappa = scene["y0"], scene["psi"], scene["kappa"]

    rows = np.arange(int(CY) + 1, IMG_H)
    xs = FOCAL * CAM_HEIGHT / (rows - CY)                    # distance per row
    yc = centreline(xs, y0, psi, kappa)
    # pinhole: u = CX - FOCAL * y / x   (+y is left -> smaller u)
    # Two-lane road, ego in the RIGHT lane. The dashed divider sits half a lane
    # to the left of where the ego should be -- so the target is never painted
    # directly on the ground; the model has to infer it from road geometry.
    u_l = CX - FOCAL * (yc + 1.5 * LANE_WIDTH) / xs          # far kerb
    u_r = CX - FOCAL * (yc - 0.5 * LANE_WIDTH) / xs          # near kerb
    u_c = CX - FOCAL * (yc + 0.5 * LANE_WIDTH) / xs          # dashed divider
    half_line = np.maximum(FOCAL * 0.08 / xs, 0.6)           # 16 cm paint
    dash_on = (np.mod(xs + scene["dash_phase"], 6.0) < 3.0)

    for i, v in enumerate(rows):
        a, b = int(np.floor(u_l[i])), int(np.ceil(u_r[i]))
        if b < 0 or a >= IMG_W:
            continue
        img[v, max(a, 0):min(b, IMG_W)] = C_ROAD
        # solid edge lines
        for u_edge in (u_l[i], u_r[i]):
            e0, e1 = int(u_edge - half_line[i]), int(np.ceil(u_edge + half_line[i]))
            if e1 > 0 and e0 < IMG_W:
                img[v, max(e0, 0):min(e1, IMG_W)] = C_EDGE
        # dashed centre line
        if dash_on[i]:
            c0, c1 = int(u_c[i] - half_line[i]), int(np.ceil(u_c[i] + half_line[i]))
            if c1 > 0 and c0 < IMG_W:
                img[v, max(c0, 0):min(c1, IMG_W)] = C_LINE

    # lead vehicle: a box on the lane centre
    lx = scene["lead_x"]
    if np.isfinite(lx) and lx > 4.0:
        ly = centreline(lx, y0, psi, kappa)
        uc = CX - FOCAL * ly / lx
        hw = FOCAL * 0.9 / lx                                # 1.8 m wide
        v_bot = CY + FOCAL * CAM_HEIGHT / lx
        v_top = CY + FOCAL * (CAM_HEIGHT - 1.5) / lx         # 1.5 m tall
        a, b = int(uc - hw), int(np.ceil(uc + hw))
        t, bo = int(v_top), int(np.ceil(v_bot))
        if b > 0 and a < IMG_W and bo > 0 and t < IMG_H:
            img[max(t, 0):min(bo, IMG_H), max(a, 0):min(b, IMG_W)] = C_CAR

    img *= scene["sun"]
    if rng is not None:
        img += rng.normal(0, 4.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------
#  The expert
# ------------------------------------------------------------------------

def expert(scene):
    """Roll out a privileged expert and sample N_WAYPOINTS from its path.

    Returns (N_WAYPOINTS, 2) array of (x, y) in metres, ego frame.

    The expert knows the true scene parameters -- that is what makes it
    'privileged'. The student only sees pixels.
    """
    y0, psi, kappa = scene["y0"], scene["psi"], scene["kappa"]
    v, lx = scene["v0"], scene["lead_x"]

    # target speed: slow for curves, slow for a close lead vehicle
    v_tgt = V_MAX / (1.0 + 45.0 * abs(kappa))
    if np.isfinite(lx):
        v_tgt = min(v_tgt, max(0.0, (lx - 6.0) / 1.4))
    v_tgt = float(np.clip(v_tgt, 0.0, V_MAX))

    n_steps = int(round(HORIZON_S / DT))
    xs = np.empty(n_steps + 1)
    ts = np.arange(n_steps + 1) * DT
    xs[0] = 0.0
    for k in range(n_steps):                         # rate-limited speed tracking
        dv = np.clip(v_tgt - v, -3.0 * DT, 2.0 * DT)
        v = max(0.0, v + dv)
        xs[k + 1] = xs[k] + v * DT

    # lateral: converge onto the centreline with a ~12 m space constant
    L = 12.0
    ys = centreline(xs, y0, psi, kappa) - y0 * np.exp(-xs / L)

    idx = np.round(np.linspace(HORIZON_S / N_WAYPOINTS, HORIZON_S, N_WAYPOINTS) / DT).astype(int)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32), ts[idx]


def make_sample(rng):
    """One (image, ego_state, waypoints) triplet."""
    scene = sample_scene(rng)
    img = render(scene, rng)
    wps, _ = expert(scene)
    ego = np.array([scene["v0"] / 10.0, scene["a0"] / 3.0], np.float32)
    return img, ego, wps, scene


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    import time
    t = time.time()
    for _ in range(200):
        make_sample(rng)
    print(f"{(time.time() - t) / 200 * 1000:.2f} ms per sample")
