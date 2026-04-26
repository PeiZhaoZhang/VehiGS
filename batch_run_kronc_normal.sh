#!/bin/bash

# 2. 添加Python路径 (完全保持你原来的配置)
export PYTHONPATH=$PYTHONPATH:/root/project/gsplat
export PYTHONPATH=$PYTHONPATH:/root/project

# 3. 定义 uCO3D 数据的根目录
UCO3D_ROOT="/root/project/data/KRONC"  # 请根据实际情况修改路径，确保它指向包含所有场景的父目录

# 4. 公共参数
RESULT_TAG="recon_normal_kronc_v1"
DATA_FACTOR=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# 5. 全自动搜索所有已经预处理好的序列
echo "🔍 正在扫描全量有效序列..."
ALL_SCENES=$(find "$UCO3D_ROOT" -type d -name "0" | grep "sparse/0" | sed 's/\/sparse\/0//g')

SCENE_COUNT=$(echo "$ALL_SCENES" | wc -l)
echo "🚀 扫描完成，共发现 $SCENE_COUNT 个可训练场景。"

# 6. 开始遍历执行
CURRENT_IDX=1
for DATA in $ALL_SCENES
do
    SCENE_NAME=$(basename "$DATA")
    
    echo "=================================================="
    echo " 🟢 [$CURRENT_IDX/$SCENE_COUNT] 正在训练: $SCENE_NAME"
    echo " 📂 路径: $DATA"
    echo "=================================================="

    OUTPUT_DIR="${DATA}/${RESULT_TAG}"

    # 执行训练脚本 (此时所在目录是 /root/project/Difix3D，路径绝对能找到)
    python examples/gsplat/simple_trainer_normal.py mcmc \
        --data_dir "${DATA}" \
        --data_factor ${DATA_FACTOR} \
        --result_dir "${OUTPUT_DIR}" \
        --no-normalize-world-space \
        --disable_viewer \
        --test_every 8

    echo -e "\n✅ 场景 $SCENE_NAME 训练完成！"
    echo -e "--------------------------------------------------\n"

    ((CURRENT_IDX++))
done

echo "🎉 所有 $SCENE_COUNT 个 kronc 场景全部处理完毕！"