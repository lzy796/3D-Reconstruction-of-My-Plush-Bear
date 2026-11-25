# -*- coding: utf-8 -*-
"""
make_figure.py — 生成方法对比图（不带标签的旧图换成带标签的）

图中每一格是体素体积的一层切片（z 中轴，yx 平面；右边=背面）。
可以直观看到：±40° 覆盖时，背面被"合理但错误"地外扩了多少。
"""

import os

import numpy as np
from PIL import Image, ImageDraw

import carve as C
import synth_test as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def main():
    os.makedirs(OUT, exist_ok=True)
    truth, step = S.make_truth(S.RES, S.BBOX_MIN, S.BBOX_MAX)
    mm = step[1] * 10.0

    sets = {
        "truth\n(1121 cm3)": None,
        "A 360deg\n108 cams": S.ring(list(range(0, 360, 10)), [-10, 0, 20]),
        "B 180deg\n39 cams": S.ring(list(range(-90, 91, 15)), [-10, 0, 20]),
        "C +-40deg+rear\n68 cams": (S.ring(list(range(-40, 41, 4)), [-10, 0, 15])
                                    + S.ring([160, 170, 180, 190, 200], [0])),
        "D +-40deg only\n63 cams": S.ring(list(range(-40, 41, 4)), [-10, 0, 15]),
    }

    tiles = []
    for label, cams in sets.items():
        if cams is None:
            v = truth
            iou = 1.0
        else:
            masks = S.render_masks(truth, S.BBOX_MIN, S.BBOX_MAX, cams)
            v, _ = S.carve_set(cams, masks)
            iou = S.iou(v, truth)
        tiles.append((label, v, iou))

    T = 200
    pad = 6
    W = len(tiles) * (T + pad) + pad
    H = T + 46
    sheet = Image.new("L", (W, H), 22)
    d = ImageDraw.Draw(sheet)
    for i, (label, v, iou) in enumerate(tiles):
        sl = v[:, :, S.RES // 2]
        im = Image.fromarray((sl * 255).astype(np.uint8)).resize((T, T), Image.NEAREST)
        x = pad + i * (T + pad)
        sheet.paste(im, (x, pad))
        d.text((x + 2, T + 10), label.split("\n")[0], fill=230)
        d.text((x + 2, T + 22), label.split("\n")[1], fill=170)
        d.text((x + 2, T + 34), "IoU %.3f" % iou, fill=120)
    sheet.save(os.path.join(OUT, "slices.png"))
    print("saved", os.path.join(OUT, "slices.png"), sheet.size)


if __name__ == "__main__":
    main()
