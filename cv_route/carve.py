# -*- coding: utf-8 -*-
"""
carve.py — 视觉外壳（Space Carving）+ 对称补全

用途
    给定一组已标定的相机（COLMAP 稀疏重建）和二值剪影掩膜，
    在体素网格上做"剪影一致性"雕刻，得到物体的视觉外壳；
    可选：按对称面镜像补全（解决"只有正面照片"的问题）。

理论
    V = { x ∈ R^3 |  ∀i: x 投影在第 i 幅图像内 ⇒ mask_i(π_i(x)) = 1 }
    即：只要有一张照片说"这里是背景"，体素就被删掉。
    这是 Kutulakos & Seitz (2000) 的 space carving 的纯剪影版本，
    等价于 visual hull（Laurentini, 1994）。

用法（命令行）
    python carve.py --colmap path/to/sparse_txt --masks path/to/masks \
                    --out carved.npz --ply carved_points.ply \
                    --res 256 --symmetry x --erode 1

    需要 COLMAP 文本模型：
        colmap model_converter --input_path sparse/0 --output_path sparse_txt --output_type TXT

    掩膜：与图片同名（png/jpg 均可），白色=物体，黑色=背景。
    剪影一致性只依赖掩膜，不依赖颜色/光照 —— 这是本方法对"手持生活照"
    鲁棒的根本原因。

作者：给这个项目的实现原型；只依赖 numpy + Pillow。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
# COLMAP 文本模型读取
# ----------------------------------------------------------------------
def _read_cameras_txt(path):
    cams = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            cam_id = int(tok[0])
            model = tok[1]
            w, h = int(tok[2]), int(tok[3])
            p = [float(x) for x in tok[4:]]
            if model == "SIMPLE_PINHOLE":
                f_, cx, cy = p[0], p[1], p[2]
                K = np.array([[f_, 0, cx], [0, f_, cy], [0, 0, 1]], float)
            elif model == "PINHOLE":
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
            else:  # SIMPLE_RADIAL / RADIAL / OPENCV / FULL_OPENCV ...
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
            cams[cam_id] = dict(model=model, w=w, h=h, K=K, params=p)
    return cams


def _qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,     1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w,     1 - 2 * x * x - 2 * y * y],
    ], float)


def read_colmap_text(sparse_dir):
    """返回 [(name, K, R, t, w, h), ...]，约定 X_cam = R @ X_world + t（COLMAP 约定）"""
    cams = _read_cameras_txt(os.path.join(sparse_dir, "cameras.txt"))
    out = []
    with open(os.path.join(sparse_dir, "images.txt"), "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        tok = line.split()
        q = [float(x) for x in tok[1:5]]
        tvec = np.array([float(x) for x in tok[5:8]])
        cam_id = int(tok[8])
        name = " ".join(tok[9:])
        R = _qvec_to_rotmat(q)
        c = cams[cam_id]
        out.append((name, c["K"], R, tvec, c["w"], c["h"]))
        i += 2  # images.txt 每个图像后跟一行 2D 点
    return out


def read_points3D(sparse_dir):
    path = os.path.join(sparse_dir, "points3D.txt")
    pts = []
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            tok = line.split()
            if len(tok) < 4:
                continue
            pts.append([float(tok[1]), float(tok[2]), float(tok[3])])
    return np.array(pts, float) if pts else None


# ----------------------------------------------------------------------
# 掩膜
# ----------------------------------------------------------------------
def load_mask(path, w, h):
    """读成 bool 数组，(h, w)；白色(>127)=物体。文件不存在时返回 None（该视图不约束）"""
    if path is None or not os.path.exists(path):
        return None
    im = Image.open(path).convert("L")
    if im.size != (w, h):
        im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im) > 127


def find_mask(mask_dir, image_name):
    if not mask_dir:
        return None
    stem = os.path.splitext(image_name)[0]
    for ext in (".png", ".PNG", ".jpg", ".jpeg", ".bmp"):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    # 允许掩膜放在子目录里（与 images 同结构）
    hits = glob.glob(os.path.join(mask_dir, "**", stem + ".*"), recursive=True)
    return hits[0] if hits else None


# ----------------------------------------------------------------------
# 体素网格
# ----------------------------------------------------------------------
def make_grid(bbox_min, bbox_max, res):
    """返回 (N,3) 体素中心坐标，以及形状信息"""
    bmin = np.asarray(bbox_min, float)
    bmax = np.asarray(bbox_max, float)
    span = bmax - bmin
    step = span / res
    xs = bmin[0] + (np.arange(res) + 0.5) * step[0]
    ys = bmin[1] + (np.arange(res) + 0.5) * step[1]
    zs = bmin[2] + (np.arange(res) + 0.5) * step[2]
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")   # 注意顺序：vol[z, y, x]
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return pts, res, step


def project(K, R, t, X):
    """X: (N,3) world → u, v, depth"""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * Xc[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / z_safe + K[1, 2]
    return u, v, z


def carve(pts, cams, masks, chunk=250_000, tolerance=0, verbose=True):
    """
    剪影一致性雕刻（带鲁棒容忍）。

    tolerance = k：允许最多 k 个视图"反证"（投影落在背景上）而不删，
    用于吸收 SfM 位姿误差与掩膜边缘误差。k=0 即经典 visual hull。

    返回 bool 数组（与 pts 等长）
    """
    N = len(pts)
    viol = np.zeros(N, dtype=np.int16)
    for idx, ((name, K, R, t, w, h), mask) in enumerate(zip(cams, masks)):
        if mask is None:
            continue
        alive = np.flatnonzero(viol <= tolerance)
        if alive.size == 0:
            break
        for s in range(0, alive.size, chunk):
            sel = alive[s:s + chunk]
            u, v, z = project(K, R, t, pts[sel])
            inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not inside.any():
                continue
            uu = np.clip(u[inside].astype(np.int64), 0, w - 1)
            vv = np.clip(v[inside].astype(np.int64), 0, h - 1)
            outside_object = ~mask[vv, uu]
            bad = sel[inside][outside_object]
            np.add.at(viol, bad, 1)
        if verbose:
            print("  [carve] %-40s kept %8d / %d" % (name[-40:], (viol <= tolerance).sum(), N))
    return viol <= tolerance


# ----------------------------------------------------------------------
# 后处理
# ----------------------------------------------------------------------
def to_volume(keep, res):
    return keep.reshape((res, res, res))


def erode3d(vol, iters=1):
    for _ in range(iters):
        p = np.pad(vol, 1, constant_values=False)
        nb = (p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1]
              & p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1]
              & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:])
        vol = vol & nb
    return vol


def dilate3d(vol, iters=1):
    for _ in range(iters):
        p = np.pad(vol, 1, constant_values=False)
        nb = (p[:-2, 1:-1, 1:-1] | p[2:, 1:-1, 1:-1]
              | p[1:-1, :-2, 1:-1] | p[1:-1, 2:, 1:-1]
              | p[1:-1, 1:-1, :-2] | p[1:-1, 1:-1, 2:])
        vol = vol | nb
    return vol


def symmetric_complete(vol, axis=0, op="union"):
    """
    关于网格中心平面镜像补全。
    要求网格在 axis 方向居中（本函数用翻转近似中心对称）。
    """
    v = np.flip(vol, axis=axis)
    return (vol | v) if op == "union" else (vol & v)


def largest_component(vol):
    """6-连通的最大连通域（纯 numpy 的迭代膨胀，网格小的时候够用）"""
    lab = vol.copy()
    # 用多次膨胀+与运算收敛（比 union-find 简单，收敛次数 = 直径体素数）
    for _ in range(int(max(vol.shape)) * 2):
        new = lab & dilate3d(lab)
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def boundary_points(vol, bbox_min, step):
    """取边界体素中心，导出为点云（给 MeshLab/Poisson 用）"""
    p = np.pad(vol, 1, constant_values=False)
    interior = (p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1]
                & p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1]
                & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:])
    surf = vol & ~interior
    zz, yy, xx = np.nonzero(surf)
    bmin = np.asarray(bbox_min, float)
    pts = np.stack([bmin[0] + (xx + 0.5) * step[0],
                    bmin[1] + (yy + 0.5) * step[1],
                    bmin[2] + (zz + 0.5) * step[2]], axis=1)
    return pts


def write_ply(path, pts):
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(pts))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write("%.5f %.5f %.5f\n" % (p[0], p[1], p[2]))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run(colmap_dir, mask_dir, out_npz, out_ply, res=256,
        symmetry=None, erode=0, margin=0.12, bbox=None, tolerance=0, verbose=True):
    cams_all = read_colmap_text(colmap_dir)
    if verbose:
        print("[carve] 读到 %d 个已标定相机" % len(cams_all))

    masks = [load_mask(find_mask(mask_dir, n), w, h) for (n, K, R, t, w, h) in cams_all]
    n_masked = sum(m is not None for m in masks)
    if verbose:
        print("[carve] 其中 %d 张有掩膜，其余视为'不约束'" % n_masked)
    if n_masked == 0:
        raise SystemExit("没有任何掩膜可用")

    if bbox is not None:
        bmin, bmax = bbox
    else:
        P = read_points3D(colmap_dir)
        if P is None or len(P) < 10:
            raise SystemExit("没有 points3D.txt，请用 --bbox 指定包围盒")
        bmin = P.min(axis=0)
        bmax = P.max(axis=0)
        pad = (bmax - bmin) * margin
        bmin, bmax = bmin - pad, bmax + pad
    # 体素数组的布局是 vol[z, y, x]，所以轴映射是反的
    AXIS = {"x": 2, "y": 1, "z": 0}
    if symmetry:
        # 让网格在对称轴方向关于 0 对称，镜像才是严格对应的
        ax = AXIS[symmetry]
        m = max(abs(bmin[ax]), abs(bmax[ax]))
        bmin[ax], bmax[ax] = -m, m
    if verbose:
        print("[carve] bbox %s → %s, res=%d" % (np.round(bmin, 3), np.round(bmax, 3), res))

    pts, n, step = make_grid(bmin, bmax, res)
    keep = carve(pts, cams_all, masks, tolerance=tolerance, verbose=verbose)
    vol = to_volume(keep, res)
    if verbose:
        print("[carve] 原始外壳体素数：%d" % vol.sum())

    if erode:
        vol = erode3d(vol, erode)
        if verbose:
            print("[carve] 腐蚀 %d 次后：%d" % (erode, vol.sum()))

    if symmetry:
        vol = symmetric_complete(vol, AXIS[symmetry])
        if erode:
            vol = erode3d(vol, 1)   # 镜像并集会在对称面处凸出一层，收一下
        if verbose:
            print("[carve] 对称补全后：%d" % vol.sum())

    np.savez_compressed(out_npz, volume=vol, bbox_min=bmin, bbox_max=bmax, res=res)
    if out_ply:
        P = boundary_points(vol, bmin, step)
        write_ply(out_ply, P)
        if verbose:
            print("[carve] 表面点 %d → %s" % (len(P), out_ply))
    return vol


def main(argv=None):
    ap = argparse.ArgumentParser(description="视觉外壳 + 对称补全")
    ap.add_argument("--colmap", required=True, help="COLMAP 文本模型目录（含 cameras.txt/images.txt/points3D.txt）")
    ap.add_argument("--masks", default=None, help="掩膜目录")
    ap.add_argument("--out", default="carved.npz", help="输出体素 npz")
    ap.add_argument("--ply", default="carved_points.ply", help="输出表面点云")
    ap.add_argument("--res", type=int, default=256, help="体素分辨率（单轴）")
    ap.add_argument("--symmetry", default=None, choices=["x", "y", "z"], help="沿哪个世界轴镜像补全")
    ap.add_argument("--erode", type=int, default=0, help="腐蚀次数（视觉外壳会偏胖）")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="允许几个视图反证（吸收位姿/掩膜误差，建议 1~3）")
    ap.add_argument("--bbox", nargs=6, type=float, default=None,
                    metavar=("xmin", "ymin", "zmin", "xmax", "ymax", "zmax"))
    args = ap.parse_args(argv)

    bbox = (np.array(args.bbox[:3]), np.array(args.bbox[3:])) if args.bbox else None
    run(args.colmap, args.masks, args.out, args.ply, res=args.res,
        symmetry=args.symmetry, erode=args.erode, bbox=bbox, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
