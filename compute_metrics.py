import os
import json
import numpy as np
from pathlib import Path

# ====================== 配置项（你已经给全了）======================
ROOT_DIR = "/root/project/data/KRONC"
BENCHMARK_DIR = os.path.join(ROOT_DIR, "benchmark")

# 要计算平均的两个版本
VERSION_CONFIGS = [
    {
        "subdir": "recon_difix3d_kronc_v3_mask_ref",
        "save_name": "average_recon_difix3d_kronc_v3_mask_ref.json"
    },
    {
        "subdir": "recon_normal_kronc_v1",
        "save_name": "average_recon_normal_kronc_v1.json"
    },
]

# 要平均的指标
KEYS = ["obj_psnr", "obj_ssim", "obj_lpips", "ellipse_time", "num_GS"]

# =================================================================

# 创建 benchmark 文件夹（不存在自动创建）
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# 遍历两个版本
for cfg in VERSION_CONFIGS:
    subdir = cfg["subdir"]
    save_name = cfg["save_name"]
    save_path = os.path.join(BENCHMARK_DIR, save_name)

    print(f"\n===== 正在计算：{subdir} =====")

    # 收集所有场景路径
    scene_dirs = []
    for scene_name in os.listdir(ROOT_DIR):
        scene_path = os.path.join(ROOT_DIR, scene_name)
        if not os.path.isdir(scene_path):
            continue
        # 检查是否存在该版本的结果
        stats_path = os.path.join(
            scene_path, subdir, "stats", "val_step29999.json"
        )
        if os.path.exists(stats_path):
            scene_dirs.append(stats_path)

    print(f"找到 {len(scene_dirs)} 个有效场景")

    # 读取所有json
    all_metrics = {k: [] for k in KEYS}
    for p in scene_dirs:
        try:
            with open(p, "r") as f:
                data = json.load(f)
            for k in KEYS:
                if k in data:
                    all_metrics[k].append(data[k])
        except:
            print(f"跳过：{p}")

    # 计算平均
    avg = {}
    for k in KEYS:
        vals = all_metrics[k]
        if len(vals) == 0:
            avg[k] = 0.0
        else:
            avg[k] = float(np.mean(vals))

    # 保存
    with open(save_path, "w") as f:
        json.dump(avg, f, indent=4)

    print(f"✅ 平均结果已保存到：{save_path}")
    print(f"📊 结果：{avg}")

print("\n🎉 全部完成！")