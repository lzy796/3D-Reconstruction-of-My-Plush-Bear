# -*- coding: utf-8 -*-
"""
setup_scene.py — 小熊重建「一键搭骨架」（Blender 4.x）

做什么：
  1. 清空当前场景
  2. 按实测尺寸放好 7 个部件（头、吻、躯干、双耳、双臂、双腿）
  3. 合并 + 体素重构 + 平滑，得到一个"面团"
  4. 放好前 / 后两块参考板（只在视口显示，不参与渲染）
  5. 放好正交相机 + 3 盏灯
  6. 把单位设成 1 单位 = 1 厘米，熊高 10 单位（= 10 cm）

怎么用：
  1) 打开 Blender，顶部工作区选 "Scripting"
  2) 点 "Open"，选中本文件
  3) 点右上角 ▶（Run Script），或按 Alt+P
  4) 切回 "Layout" 工作区，按 数字键盘 1 看前视图

注意：
  * 请在「新建的空白工程」里运行（它会删除场景里所有物体）
  * 参考板需要 REF_DIR 里那 4 张 png；找不到会跳过（不报错），
    只影响对照，不影响建模
  * 尺寸单位：1 Blender 单位 = 1 厘米
"""

import bpy
import os
import math

# ---------------------------------------------------------------- 配置
REF_DIR = r"D:\3D-Reconstruction-of-My-Plush-Bear\traditional_route\ref_boards"
BEAR_H = 10.0          # 熊高 10 cm
VOXEL = 0.12           # 体素重构的颗粒大小（cm），越小越细越慢
USE_REFS = True        # 找不到参考图时自动跳过


def log(msg):
    print("[bear] " + msg)


# ---------------------------------------------------------------- 工具
def purge_scene():
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
    except Exception:
        pass
    log("场景已清空")


def ellipsoid(name, size, loc, rot=(0.0, 0.0, 0.0), seg=32, ring=16):
    """size = (X 宽, Y 厚, Z 高) 单位 cm；loc = 中心点"""
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=seg, ring_count=ring, radius=0.5,
        location=loc, rotation=rot)
    ob = bpy.context.object
    ob.name = name
    ob.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    log("  加入 %s  尺寸 %s  位置 %s" % (name, size, loc))
    return ob


def add_ref_plane(fname, name, height, loc, rot):
    path = os.path.join(REF_DIR, fname)
    if not USE_REFS or not os.path.exists(path):
        log("  跳过参考板（找不到）：" + path)
        return None
    img = bpy.data.images.load(path, check_existing=True)
    w, h = img.size
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc, rotation=rot)
    ob = bpy.context.object
    ob.name = name
    ob.scale = (height * w / float(h), height, 1.0)

    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    ob.data.materials.append(mat)

    ob.hide_render = True          # 只当背景板，不参与渲染
    try:
        ob.visible_shadow = False
    except Exception:
        pass
    log("  加入参考板：" + name)
    return ob


# ---------------------------------------------------------------- 主流程
def main():
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    sc.unit_settings.length_unit = 'CENTIMETERS'
    sc.unit_settings.scale_length = 0.01      # 1 BU = 1 cm

    purge_scene()

    log("开始放部件 ……")
    parts = []

    # 尺寸已按"含毛剪影 − 毛长补偿"换算过（毛会再长回来 4~5 mm）
    parts.append(ellipsoid("head",   (6.2, 5.3, 4.0), (0.0, 0.0, 7.78)))
    parts.append(ellipsoid("muzzle", (2.6, 2.2, 1.7), (0.0, -2.4, 6.90)))
    parts.append(ellipsoid("body",   (6.6, 5.6, 3.9), (0.0, 0.0, 4.00)))

    for s in (-1, 1):
        parts.append(ellipsoid(
            "ear_%s" % ("L" if s < 0 else "R"),
            (1.0, 0.6, 1.3),
            (s * 2.6, 0.5, 8.60),
            rot=(math.radians(-15.0), 0.0, s * math.radians(18.0))))

    for s in (-1, 1):
        parts.append(ellipsoid(
            "arm_%s" % ("L" if s < 0 else "R"),
            (1.5, 1.5, 2.3),
            (s * 2.9, -0.2, 4.40),
            rot=(0.0, s * math.radians(12.0), 0.0)))

    for s in (-1, 1):
        parts.append(ellipsoid(
            "leg_%s" % ("L" if s < 0 else "R"),
            (2.6, 2.6, 2.6),
            (s * 1.6, -1.2, 1.40)))

    # ---- 合并 ----
    bpy.ops.object.select_all(action='DESELECT')
    for ob in parts:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    bear = bpy.context.object
    bear.name = "Bear_Blockout"
    log("已合并为：Bear_Blockout")

    # ---- 体素重构：把 7 个球变成一张连续的面 ----
    m = bear.modifiers.new(name="Remesh", type='REMESH')
    m.mode = 'VOXEL'
    m.voxel_size = VOXEL
    m.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=m.name)
    log("体素重构完成（颗粒 %.2f cm）" % VOXEL)

    s = bear.modifiers.new(name="Smooth", type='SMOOTH')
    s.factor = 0.5
    s.iterations = 6
    bpy.ops.object.modifier_apply(modifier=s.name)
    log("平滑完成")

    # ---- 参考板 ----
    # 前视板：整张图按 11.70 cm 高摆放，使"熊"刚好 10 cm；
    # 熊在图里的包围盒是 x 120~1090 / y 85~1145（图 1327x1240），
    # 据此把图中心挪到 (0.55, +8, 5.05)，让熊的中线对准 x=0。
    log("放参考板 ……")
    add_ref_plane("ref_front_105.png", "REF_front", 11.70,
                  (0.55, 8.0, 5.05), (math.radians(90.0), 0.0, 0.0))
    # 背视板：来自另一张照片、且左边被裁掉，只作"徽章长什么样"的参照，
    # 不需要和模型严丝合缝（要用时自己按 S / G 微调）。
    add_ref_plane("ref_back_80.png", "REF_back", 11.70,
                  (0.0, -8.0, 5.05), (math.radians(90.0), 0.0, math.radians(180.0)))

    # ---- 相机（正交前视）----
    cd = bpy.data.cameras.new("Cam_Ortho")
    cd.type = 'ORTHO'
    cd.ortho_scale = 12.0
    cam = bpy.data.objects.new("Cam_Ortho", cd)
    sc.collection.objects.link(cam)
    cam.location = (0.0, -25.0, 5.0)
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    sc.camera = cam
    log("相机就位（正交，视野 12 cm）")

    # ---- 三盏灯（阴天柔光的大致效果）----
    def area(name, loc, rot, size, energy):
        bpy.ops.object.light_add(type='AREA', location=loc, rotation=rot)
        ob = bpy.context.object
        ob.name = name
        ob.data.size = size
        ob.data.energy = energy
        return ob

    area("Key",  (-8.0, -10.0, 14.0), (math.radians(55.0), 0.0, math.radians(-40.0)), 6.0, 60.0)
    area("Fill", (10.0, -8.0, 6.0),  (math.radians(75.0), 0.0, math.radians(55.0)),  6.0, 18.0)
    area("Rim",  (0.0, 9.0, 13.0),   (math.radians(140.0), 0.0, 0.0),                5.0, 35.0)
    log("灯光就位（Key / Fill / Rim）")

    # ---- 环境光（浅灰，避免暗部死黑）----
    try:
        w = sc.world
        if w is None:
            w = bpy.data.worlds.new("World")
            sc.world = w
        w.use_nodes = True
        bg = w.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.55, 0.57, 0.62, 1.0)
            bg.inputs[1].default_value = 0.6
    except Exception as e:
        log("环境光设置跳过：" + str(e))

    # ---- 视口友好设置 ----
    try:
        sc.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        try:
            sc.render.engine = 'BLENDER_EEVEE'
        except Exception:
            pass
    sc.render.resolution_x = 1600
    sc.render.resolution_y = 1600
    sc.render.film_transparent = True

    bpy.ops.object.select_all(action='DESELECT')
    bear.select_set(True)
    bpy.context.view_layer.objects.active = bear

    log("完成。下一步：数字键盘 1 看前视图 → Tab 之外的 Z 键切线框 → 对照前后参考板调轮廓")


if __name__ == "__main__":
    main()
