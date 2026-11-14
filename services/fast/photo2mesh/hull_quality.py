"""Calibrate inferred silhouettes and rasterize mesh evidence."""
from dataclasses import replace

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.optimize import least_squares

from .camera import Camera
from .silhouette import analyze


def recover_mask(photo, mask):
    """Separate a saturated subject from gray background attached to the border."""
    h, w = mask.shape
    lab = photo.lab
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]))
    chroma = np.linalg.norm(lab[..., 1:] - np.median(border[:, 1:], axis=0), axis=-1)
    seed = cv2.morphologyEx(np.uint8(chroma > 14), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed)
    if count <= 1:
        return mask
    clean = binary_fill_holes(labels == 1 + stats[1:, 4].argmax())
    if not .62 * mask.sum() < clean.sum() < .78*h*w:
        return mask
    return clean


def profile_prior(mask, rounded):
    """Derive a rotational envelope and softly fit broad apple sections to an ellipsoid."""
    profile = analyze(mask, samples=160, smoothing=.025, rounded=rounded)
    y, r = profile.heights, profile.sample_radii
    if not rounded:
        return y, r
    broad = r > .28*r.max()
    def model(p):
        return p[0]*np.sqrt(np.maximum(0, 1-((y-p[2])/p[1])**2))
    fit = least_squares(lambda p: (model(p)-r)[broad], [min(1.99,r.max()), .52, 0],
                        bounds=([.001,.32,-.13],[2.,.72,.13]), loss="soft_l1", f_scale=.025)
    return y, np.where(broad, .85*r+.15*model(fit.x), r)


def align(view, points):
    """Fit bounded image offset and orbit distance to a projected reference envelope."""
    y, x = np.where(view.mask)
    target = np.array([x.min(), x.max(), y.min(), y.max()])
    camera = view.camera
    length = max(1, np.ptp(y))
    def candidate(p):
        return Camera.perspective(camera.azimuth, camera.elevation, camera.focal_length,
                                  camera.cx+p[1]*length, camera.cy+p[2]*length,
                                  camera.distance*np.exp(p[0]))
    def residual(p):
        u, v, _ = candidate(p).project(points)
        bounds = np.array([np.nanmin(u), np.nanmax(u), np.nanmin(v), np.nanmax(v)])
        return np.r_[(bounds-target)/length, .16*p]
    fit = least_squares(residual, [0,0,0], bounds=([-.22,-.08,-.08],[.22,.08,.08]))
    return replace(view, camera=candidate(fit.x))


def rasterize(vertices, faces, view):
    """Rasterize silhouette coverage and the near surface in camera depth."""
    x, y, depth = view.camera.project(vertices)
    buffer = np.full(view.mask.shape, -np.inf, dtype=np.float32)
    h, w = buffer.shape
    f = np.asarray(faces)
    if not len(f):
        return buffer
    ux, vy, dz = x[f], y[f], depth[f]
    lefts = np.maximum(0, np.ceil(ux.min(axis=1))).astype(int)
    rights = np.minimum(w - 1, np.floor(ux.max(axis=1))).astype(int)
    tops = np.maximum(0, np.ceil(vy.min(axis=1))).astype(int)
    bottoms = np.minimum(h - 1, np.floor(vy.max(axis=1))).astype(int)
    dets = (ux[:, 1] - ux[:, 0]) * (vy[:, 2] - vy[:, 0]) - (ux[:, 2] - ux[:, 0]) * (vy[:, 1] - vy[:, 0])
    valid = (
        np.isfinite(ux).all(axis=1)
        & np.isfinite(vy).all(axis=1)
        & (rights >= lefts)
        & (bottoms >= tops)
        & (np.abs(dets) >= 1e-9)
    )
    distance = view.camera.distance
    for u, v, d, left, right, top, bottom, det in zip(
        ux[valid], vy[valid], dz[valid], lefts[valid], rights[valid], tops[valid], bottoms[valid], dets[valid]
    ):
        yy, xx = np.ogrid[top:bottom + 1, left:right + 1]
        a = ((xx - u[0]) * (v[2] - v[0]) - (yy - v[0]) * (u[2] - u[0])) / det
        b = ((yy - v[0]) * (u[1] - u[0]) - (xx - u[0]) * (v[1] - v[0])) / det
        c = 1 - a - b
        if distance is None:
            z = c * d[0] + a * d[1] + b * d[2]
        else:
            inv = 1 / (distance - d)
            z = distance - 1 / np.maximum(1e-8, c * inv[0] + a * inv[1] + b * inv[2])
        patch = buffer[top:bottom + 1, left:right + 1]
        np.maximum(patch, z, out=patch, where=(a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6))
    return buffer


def polyhedral_prior(view, normal_rgb, basis=(1, 1, 1)):
    """Recover an oriented rectangular cuboid from a front normal map with two orthogonal vertical walls."""
    from .surface_refinement import decode

    pred, valid = decode(normal_rgb, view.mask)
    if np.count_nonzero(valid) < 120:
        return None
    pred = pred * np.asarray(basis, dtype=float)
    vecs_raw = pred[valid]
    if float(np.mean(vecs_raw[:, 2])) < -0.1:
        pred = pred * np.array([-1.0, 1.0, -1.0])
        vecs_raw = pred[valid]

    ys, xs = np.where(valid)
    vecs = pred[ys, xs]
    top_m = vecs[:, 1] > 0.65
    left_m = (np.abs(vecs[:, 1]) < 0.45) & (vecs[:, 0] < -0.20)
    right_m = (np.abs(vecs[:, 1]) < 0.45) & (vecs[:, 0] > 0.15)
    n_tot = len(vecs)
    if np.count_nonzero(left_m) < 0.12 * n_tot or np.count_nonzero(right_m) < 0.12 * n_tot:
        return None
    coverage = (np.count_nonzero(left_m) + np.count_nonzero(right_m) + np.count_nonzero(top_m)) / n_tot
    if coverage < 0.82:
        return None

    n_left = vecs[left_m].mean(axis=0)
    n_left /= max(np.linalg.norm(n_left), 1e-9)
    n_right = vecs[right_m].mean(axis=0)
    n_right /= max(np.linalg.norm(n_right), 1e-9)
    d_left = 1.0 - vecs[left_m] @ n_left
    d_right = 1.0 - vecs[right_m] @ n_right
    if (
        np.mean(d_left) >= 0.030
        or np.mean(d_right) >= 0.030
        or np.median(d_left) >= 0.010
        or np.median(d_right) >= 0.010
    ):
        return None

    e1_xz = np.array([n_right[0], 0.0, n_right[2]])
    n_left_xz = np.array([n_left[0], 0.0, n_left[2]])
    if np.linalg.norm(e1_xz) < 0.5 or np.linalg.norm(n_left_xz) < 0.5:
        return None
    e1_xz /= np.linalg.norm(e1_xz)
    n_left_xz /= np.linalg.norm(n_left_xz)
    if abs(float(np.dot(e1_xz, n_left_xz))) >= 0.18:
        return None

    e3_xz = np.array([-e1_xz[2], 0.0, e1_xz[0]])
    if np.dot(e3_xz, n_left_xz) < 0:
        e3_xz = -e3_xz

    scale = max(float(view.camera.pixels_per_unit), 1e-6)
    u = (xs - view.camera.cx) / scale
    v = (view.camera.cy - ys) / scale
    vert_m = left_m | right_m
    u_min, u_max = float(np.percentile(u, 0.5)), float(np.percentile(u, 99.5))
    u_crease = 0.5 * (float(np.percentile(u[left_m], 97.5)) + float(np.percentile(u[right_m], 2.5)))
    v_bot = float(np.percentile(v[vert_m], 0.5))
    v_crease = float(np.percentile(v[vert_m], 97.5))
    v_top = float(np.percentile(v, 99.5))
    if u_crease <= u_min + 1e-3 or u_max <= u_crease + 1e-3 or v_crease <= v_bot + 1e-3:
        return None

    a1 = 0.5 * (u_crease - u_min) / max(abs(float(e1_xz[0])), 1e-6)
    a3 = 0.5 * (u_max - u_crease) / max(abs(float(e3_xz[0])), 1e-6)
    sin_p = 0.0
    if np.count_nonzero(top_m) >= 20:
        denom = max(a1 * abs(float(e1_xz[2])) + a3 * abs(float(e3_xz[2])), 1e-6)
        sin_p = float(np.clip((v_top - v_crease) / denom, 0.0, 0.30))
    pitch_rad = float(np.arcsin(sin_p))
    e2 = np.array([0.0, np.cos(pitch_rad), np.sin(pitch_rad)])
    e1 = e1_xz - np.dot(e1_xz, e2) * e2
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e3 = np.cross(e1, e2)
    e3 /= max(np.linalg.norm(e3), 1e-9)
    if np.dot(e3, n_left) < 0:
        e3 = -e3
    rot = np.column_stack((e1, e2, e3))
    a2 = 0.5 * (v_crease - v_bot) / max(abs(float(e2[1])), 1e-6)
    x0 = u_crease - (a1 * float(e1[0]) + a3 * float(e3[0]))
    y0 = 0.5 * (v_bot + v_top)
    center = np.array([x0, y0, 0.0])
    half_ext = np.array([a1, a2, a3])
    return rot, center, half_ext


def agreement(projected, mask):
    """Measure silhouette intersection over union on the image grid."""
    return float(np.count_nonzero(projected & mask)/max(1,np.count_nonzero(projected | mask)))

