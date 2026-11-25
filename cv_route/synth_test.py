# -*- coding: utf-8 -*-
"""
synth_test.py — 对 carve.py 的合成数据验证（含鲁棒性扫描）

回答四个问题：
  Q1 全视角下，剪影雕刻能否还原形状？（实现正确性）
  Q2 只有"正面 ±40° + 少量背面"（本项目真实覆盖）时，外壳错在哪？错多少？
  Q3 对称补全能救回多少？
  Q4 相机位姿有误差 / 掩膜有膨胀时，方法有多脆？

输出：
  cv_route/results/synthetic_results.md
  cv_route/results/synthetic_results.json
  cv_route/results/slices.png
"""

import json
import os

import numpy as np
from PIL import Image

import carve as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

RES = 128
BBOX_MIN = np.array([-9.5, -8.0, -2.5])
BBOX_MAX = np.array([9.5, 8.0, 12.5])
IMG_W = IMG_H = 512
FOCAL = 900.0
CAM_R = 45.0

AX_X, AX_Y, AX_Z = 2, 1, 0          # vol[z, y, x]


# ----------------------------------------------------------------------
def ellipsoid(X, center, radii):
    d = (X - np.asarray(center)) / np.asarray(radii)
    return (d ** 2).sum(axis=1) <= 1.0


def make_truth(res, bmin, bmax):
    pts, n, step = C.make_grid(bmin, bmax, res)
    parts = [
        ((0.0, 0.0, 3.2), (6.6, 5.6, 4.2)),     # 躯干
        ((0.0, 0.0, 7.6), (6.0, 5.0, 4.0)),     # 头
        ((0.0, -4.6, 6.6), (2.6, 2.2, 1.7)),    # 吻
        ((-4.6, 0.6, 10.2), (1.1, 0.7, 1.5)),   # 左耳
        ((4.6, 0.6, 10.2), (1.1, 0.7, 1.5)),    # 右耳
        ((-3.4, -2.2, 0.9), (2.7, 2.7, 2.7)),   # 左腿
        ((3.4, -2.2, 0.9), (2.7, 2.7, 2.7)),    # 右腿
        ((-6.4, -0.4, 4.4), (1.7, 1.7, 2.4)),   # 左臂
        ((6.4, -0.4, 4.4), (1.7, 1.7, 2.4)),    # 右臂
    ]
    inside = np.zeros(len(pts), dtype=bool)
    for c, r in parts:
        inside |= ellipsoid(pts, c, r)
    return inside.reshape((res, res, res)), step


def look_at(C_, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    C_ = np.asarray(C_, float)
    f = np.asarray(target, float) - C_
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f], axis=0)          # world→camera，x右 y下 z前
    return R, -R @ C_


def ring(az_list, el_list):
    cams = []
    K = np.array([[FOCAL, 0, IMG_W / 2.0], [0, FOCAL, IMG_H / 2.0], [0, 0, 1.0]])
    for el in el_list:
        for az in az_list:
            a, e = np.radians(az), np.radians(el)
            C_ = CAM_R * np.array([np.sin(a) * np.cos(e),
                                   -np.cos(a) * np.cos(e),
                                   np.sin(e)])
            R, t = look_at(C_)
            cams.append(("az%+04d_el%+03d" % (az, el), K, R, t, IMG_W, IMG_H))
    return cams


def render_masks(truth, bmin, bmax, cams):
    res = truth.shape[0]
    pts, n, step = C.make_grid(bmin, bmax, res)
    obj = pts[truth.ravel()]
    out = []
    for (name, K, R, t, w, h) in cams:
        m = np.zeros((h, w), dtype=bool)
        u, v, z = C.project(K, R, t, obj)
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        m[np.clip(v[ok], 0, h - 1).astype(np.int64),
          np.clip(u[ok], 0, w - 1).astype(np.int64)] = True
        out.append(m)
    return out


def dilate_mask(m, it):
    for _ in range(it):
        p = np.pad(m, 1, constant_values=False)
        m = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
             | p[1:-1, 1:-1])
    return m


def rot_axis_angle(axis, ang):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(ang), np.sin(ang)
    Cc = 1 - c
    return np.array([
        [c + x * x * Cc, x * y * Cc - z * s, x * z * Cc + y * s],
        [y * x * Cc + z * s, c + y * y * Cc, y * z * Cc - x * s],
        [z * x * Cc - y * s, z * y * Cc + x * s, c + z * z * Cc],
    ])


def perturb(cams, deg, rel, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for (name, K, R, t, w, h) in cams:
        ax = rng.normal(size=3)
        Rn = rot_axis_angle(ax, np.radians(deg * rng.normal())) @ R
        tn = t + rel * np.linalg.norm(t) * rng.normal(size=3)
        out.append((name, K, Rn, tn, w, h))
    return out


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(u) if u else 0.0


def ymax(vol):
    _, yy, _ = np.nonzero(vol)
    return int(yy.max()) if yy.size else -1


def carve_set(cams, masks, verbose=False, tolerance=0):
    pts, n, st = C.make_grid(BBOX_MIN, BBOX_MAX, RES)
    keep = C.carve(pts, cams, masks, tolerance=tolerance, verbose=verbose)
    return keep.reshape((RES, RES, RES)), st


# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    truth, step = make_truth(RES, BBOX_MIN, BBOX_MAX)
    vox = step[0] * step[1] * step[2]
    print("真值：%d 体素 ≈ %.0f cm³" % (truth.sum(), truth.sum() * vox))

    sets = {
        "A_360": ring(list(range(0, 360, 10)), [-10, 0, 20]),
        "B_hemi": ring(list(range(-90, 91, 15)), [-10, 0, 20]),
        "C_real": (ring(list(range(-40, 41, 4)), [-10, 0, 15])
                   + ring([160, 170, 180, 190, 200], [0])),
        "D_narrow": ring(list(range(-40, 41, 4)), [-10, 0, 15]),
    }

    rows, slices = [], {}
    for tag, cams in sets.items():
        masks = render_masks(truth, BBOX_MIN, BBOX_MAX, cams)
        vol, st = carve_set(cams, masks)
        rec = dict(set=tag, n_cam=len(cams), iou=round(iou(vol, truth), 3),
                   vol=round(vol.sum() * vox), ymax_carved=ymax(vol),
                   ymax_truth=ymax(truth))
        v_e = C.erode3d(vol, 1)
        rec["iou_erode1"] = round(iou(v_e, truth), 3)
        v_s = C.symmetric_complete(vol, AX_X)
        rec["iou_sym"] = round(iou(v_s, truth), 3)
        rec["iou_sym_erode1"] = round(iou(C.erode3d(v_s, 1), truth), 3)
        rec["vol_sym"] = round(v_s.sum() * vox)
        rows.append(rec)
        slices[tag] = vol
        print("[%s] %d cams | IoU %.3f | +erode %.3f | +sym %.3f | +sym+erode %.3f"
              % (tag, len(cams), rec["iou"], rec["iou_erode1"], rec["iou_sym"], rec["iou_sym_erode1"]))

    # ---------------- 鲁棒性扫描（用 C_real）----------------
    base_cams = sets["C_real"]
    base_masks = render_masks(truth, BBOX_MIN, BBOX_MAX, base_cams)
    rob = []
    for deg in (0.0, 0.1, 0.2, 0.5):
        for tol in (0, 1, 2, 3):
            cams_p = perturb(base_cams, deg, deg * 0.002) if deg > 0 else base_cams
            vol, st = carve_set(cams_p, base_masks, tolerance=tol)
            i1 = iou(vol, truth)
            rob.append(dict(pose_deg=deg, mask_dilate_px=0, tolerance=tol,
                            iou=round(i1, 3)))
            print("[robust] pose±%.1f° tol=%d | IoU %.3f" % (deg, tol, i1))

    # 掩膜膨胀 + 容忍 的组合（模拟"分割过松"）
    for dm, tol in ((2, 0), (2, 2), (4, 2)):
        masks_p = [dilate_mask(m, dm) for m in base_masks]
        vol, st = carve_set(base_cams, masks_p, tolerance=tol)
        i1 = iou(vol, truth)
        rob.append(dict(pose_deg=0.0, mask_dilate_px=dm, tolerance=tol, iou=round(i1, 3)))
        print("[robust] mask+%dpx tol=%d | IoU %.3f" % (dm, tol, i1))

    # ---------------- 切片图 ----------------
    order = ["truth", "A_360", "B_hemi", "C_real", "D_narrow"]
    mid = RES // 2
    tiles = []
    for tag in order:
        v = truth if tag == "truth" else slices[tag]
        sl = v[:, :, mid]                       # z 中轴：yx 平面
        im = Image.fromarray((sl * 255).astype(np.uint8)).resize((220, 220), Image.NEAREST)
        tiles.append((tag, im))
    sheet = Image.new("L", (220 * len(tiles), 220 + 18), 30)
    for i, (tag, im) in enumerate(tiles):
        sheet.paste(im, (220 * i, 18))
    sheet.save(os.path.join(OUT, "slices.png"))

    # ---------------- 写报告 ----------------
    md = ["# 合成实验：剪影雕刻 + 对称补全", "",
          "真值：9 个椭球拼成的小熊，%d 体素 ≈ %.0f cm³；网格 %d³（体素 %.2f mm）；"
          "相机焦距 %d px，图像 %d×%d。" % (truth.sum(), truth.sum() * vox, RES,
                                            step[0] * 10, FOCAL, IMG_W, IMG_H), "",
          "## 1. 不同覆盖度", "",
          "| 配置 | 相机数 | IoU | +腐蚀1 | +对称补全 | +对称+腐蚀1 | 体积(cm³) | 背面最大 y |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %d | %d |" % (
            r["set"], r["n_cam"], r["iou"], r["iou_erode1"], r["iou_sym"],
            r["iou_sym_erode1"], r["vol"], r["ymax_carved"]))
    md += ["", "真值体积 ≈ %d cm³，背面最大 y = %d（体素索引）。" % (
        round(truth.sum() * vox), ymax(truth)), "",
        "## 2. 鲁棒性（配置 C_real）", "",
        "| 位姿误差(std) | 掩膜膨胀 | 容忍 k | IoU |", "|---|---|---|---|"]
    for r in rob:
        md.append("| ±%.1f° | +%d px | %d | %.3f |" % (
            r["pose_deg"], r["mask_dilate_px"], r["tolerance"], r["iou"]))
    md += ["", "（位姿误差：每个相机绕随机轴加高斯角度扰动；掩膜膨胀：模拟分割过松；"
           "容忍 k：允许 k 个视图反证而不删体素）", ""]
    with open(os.path.join(OUT, "synthetic_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(OUT, "synthetic_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sets=rows, robust=rob), f, ensure_ascii=False, indent=2)
    print("\n写入：", OUT)


if __name__ == "__main__":
    main()
