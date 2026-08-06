"""
test_geometry.py -- run this before you trust anything else.

    python test_geometry.py

Coordinate-frame errors are the most common bug in autonomous driving code and
the most expensive, because they do not crash. They quietly degrade your
metrics and you spend a week blaming the model. These checks assert the
invariants that a sign flip or an axis swap would violate.
"""

import numpy as np

import toyworld as T


def test_expert_stays_on_road(n=500, seed=3):
    """The privileged expert must never drive off the road surface.

    The road spans [yc - 0.5*W, yc + 1.5*W] because ego occupies the right
    lane of two. If someone flips the sign of `y` (forgetting that +y is LEFT
    in ISO 8855) this fires immediately.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        sc = T.sample_scene(rng)
        wps, _ = T.expert(sc)
        x = np.linspace(max(wps[0, 0], 3.1), wps[-1, 0], 60)
        y = np.interp(x, wps[:, 0], wps[:, 1])
        yc = T.centreline(x, sc["y0"], sc["psi"], sc["kappa"])
        excursion = np.maximum(yc - 0.5 * T.LANE_WIDTH - y, y - (yc + 1.5 * T.LANE_WIDTH))
        worst = max(worst, float(excursion.max()))
    assert worst <= 1e-6, f"expert left the road by {worst:.3f} m"
    return f"expert on-road over {n} scenes (worst excursion {worst:.2e} m)"


def test_projection_roundtrip(n=2000, seed=5):
    """Ground plane -> pixels -> ground plane must be the identity.

    v = CY + FOCAL*h/x  and  u = CX - FOCAL*y/x  invert to
    x = FOCAL*h/(v - CY) and y = (CX - u)*x/FOCAL.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(4.0, 90.0, n)
    y = rng.uniform(-12.0, 12.0, n)
    u = T.CX - T.FOCAL * y / x
    v = T.CY + T.FOCAL * T.CAM_HEIGHT / x
    x2 = T.FOCAL * T.CAM_HEIGHT / (v - T.CY)
    y2 = (T.CX - u) * x2 / T.FOCAL
    err = float(max(np.abs(x - x2).max(), np.abs(y - y2).max()))
    assert err < 1e-6, f"projection round-trip error {err:.2e} m"
    return f"pinhole round-trip exact to {err:.1e} m over {n} points"


def test_left_is_positive_y():
    """A road curving LEFT must project to the LEFT half of the image."""
    sc = dict(y0=0.0, psi=0.0, kappa=0.01, v0=10.0, a0=0.0,
              dash_phase=0.0, lead_x=np.inf, sun=1.0)
    x = 50.0
    yc = T.centreline(x, sc["y0"], sc["psi"], sc["kappa"])
    assert yc > 0, "positive curvature should bend left (+y)"
    u = T.CX - T.FOCAL * yc / x
    assert u < T.CX, "a left bend must appear left of image centre"
    return f"left bend at 50 m -> y={yc:+.1f} m, u={u:.0f} px (centre {T.CX:.0f})"


def test_expert_respects_lead_vehicle():
    """A close lead vehicle must slow the expert down."""
    base = dict(y0=0.0, psi=0.0, kappa=0.0, v0=14.0, a0=0.0,
                dash_phase=0.0, sun=1.0)
    free, _ = T.expert({**base, "lead_x": np.inf})
    blocked, _ = T.expert({**base, "lead_x": 12.0})
    assert blocked[-1, 0] < free[-1, 0] - 5.0, "lead vehicle did not slow the expert"
    return f"4 s travel: {free[-1,0]:.1f} m free vs {blocked[-1,0]:.1f} m behind a lead"


def test_curvature_is_invisible_to_ego_state():
    """Ego state must carry no curvature information.

    This is what makes EgoOnly a fair, informative baseline: it fails on the
    lateral axis by construction, not by undertraining.
    """
    rng = np.random.default_rng(11)
    kappas, speeds = [], []
    for _ in range(4000):
        sc = T.sample_scene(rng)
        kappas.append(sc["kappa"])
        speeds.append(sc["v0"])
    r = float(np.corrcoef(np.abs(kappas), speeds)[0, 1])
    assert abs(r) < 0.05, f"curvature leaks into ego speed (r={r:.3f})"
    return f"corr(|kappa|, v0) = {r:+.4f} -- no leakage"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        try:
            print(f"  PASS  {t.__name__:38s} {t()}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__:38s} {e}")
    print(f"\n{ok}/{len(tests)} passed")
