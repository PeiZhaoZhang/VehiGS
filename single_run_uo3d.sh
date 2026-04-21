#!/bin/bash

# 1. 强制切换到工作目录
cd /root/project/Difix3D || exit 1

# 2. 添加Python路径
export PYTHONPATH=$PYTHONPATH:/root/project/gsplat:/root/project

# 3. 接收命令行参数作为路径，如果没有传参，则默认使用你指定的书本场景
TARGET_DATA=${1:-"/root/project/data/uco3d/books_and_reading_materials/book/28006-14019-9638"}

# 4. 公共参数
RESULT_TAG="recon_difix3d_uco3d_v1"
DATA_FACTOR=1
export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1

OUTPUT_DIR="${TARGET_DATA}/${RESULT_TAG}"
SCENE_NAME=$(basename "$TARGET_DATA")

echo "=================================================="
echo " 🎯 正在单点重建场景: $SCENE_NAME"
echo " 📂 路径: $TARGET_DATA"
echo "=================================================="

# 检查该路径下是否有 sparse/0 目录，避免路径填错直接崩
if [ ! -d "${TARGET_DATA}/sparse/0" ]; then
    echo "❌ 错误：在目标路径下未找到 sparse/0 位姿数据，请检查路径！"
    exit 1
fi

# 执行训练脚本
python examples/gsplat/simple_trainer_difix3d.py mcmc \
    --data_dir "${TARGET_DATA}" \
    --data_factor ${DATA_FACTOR} \
    --result_dir "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test_every 8

echo -e "\n✅ 单场景 $SCENE_NAME 训练完成！结果保存在 ${OUTPUT_DIR}\n"