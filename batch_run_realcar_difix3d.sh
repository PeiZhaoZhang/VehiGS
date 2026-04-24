#!/bin/bash

# 1. 环境变量与 Python 路径设置
export PYTHONPATH=$PYTHONPATH:/root/project/gsplat
export PYTHONPATH=$PYTHONPATH:/root/project

# 🛡️ 杀手锏：强行禁止 Hugging Face 联网 (防止网络抖动导致批量任务中断)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export DISABLE_TELEMETRY=1

# 🧹 神级外挂：允许 PyTorch 动态拓展显存段，彻底消灭碎片化 OOM！
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU 与 CUDA 设置
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# 2. 公共参数（所有场景共用）
RESULT_TAG=recon_difix3d_realcar_v3_mask_ref  # 输出文件夹名
DATA_FACTOR=1

# 3. 定义需要遍历的两个“主阵地”
PARENT_DIRS=(
    "/root/project/data/HQ200_select/ground"
    "/root/project/data/HQ200_select/under_ground"
)

# 4. 双重循环自动扫盘遍历
echo "🚀 开始执行 HQ200_select 全量自动化批处理任务..."

for PARENT_DIR in "${PARENT_DIRS[@]}"
do
    echo "=================================================="
    echo " 📂 正在扫描大区目录：$PARENT_DIR"
    echo "=================================================="

    # 健壮性检查：如果目录不存在，直接跳过，防止脚本崩溃
    if [ ! -d "$PARENT_DIR" ]; then
        echo "⚠️ 目录不存在，跳过: $PARENT_DIR"
        continue
    fi

    # 遍历该大区下的所有子文件夹 (即具体的场景)
    for DATA in "$PARENT_DIR"/*/
    do
        # 去掉路径最后的斜杠 (例如把 .../scene1/ 变成 .../scene1)
        DATA=${DATA%/}
        
        # 自动提取场景名称 (提取最后一个斜杠后面的内容)
        SCENE_ID=$(basename "$DATA")

        echo "--------------------------------------------------"
        echo " 🎯 开始处理场景：$SCENE_ID"
        echo " 📁 数据集路径：$DATA"
        echo "--------------------------------------------------"

        OUTPUT_DIR=${DATA}/${RESULT_TAG}

        # 执行 3DGS + Difix3D 训练脚本
        python examples/gsplat/simple_trainer_difix3d.py mcmc \
            --data_dir ${DATA} \
            --data_factor ${DATA_FACTOR} \
            --result_dir ${OUTPUT_DIR} \
            --no-normalize-world-space \
            --disable_viewer \
            --test_every 8

        echo -e "\n✅ 场景 $SCENE_ID 处理完成！\n"
    done
done

echo "🎉 恭喜！所有 ground 和 under_ground 中的场景已全部通关！可以去查看对比视频了！"