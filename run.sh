#!/bin/bash

# 1. 添加Python路径
export PYTHONPATH=$PYTHONPATH:/root/project/gsplat
# 指向 Difix3D 的上一级目录，这样 Python 才能识别 Difix3D 作为一个 package
export PYTHONPATH=$PYTHONPATH:/root/project
# 2. 定义需要批量处理的场景列表（你可以自由增删）
SCENE_LIST=(
"2024_09_10_18_14_13"
"2024_09_11_12_52_22"
"2024_09_12_10_28_22"
"2024_09_21_10_59_31_anonymous_SUV"
"2024_10_09_10_17_02_anonymous_special_vehicles"
)

# 3. 公共参数（所有场景共用）
RESULT_TAG=test
DATA_FACTOR=1
# CKPT_PATH=CKPT_DIR/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt  # 循环内会自动替换SCENE_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# 4. 循环遍历所有场景
for SCENE_ID in "${SCENE_LIST[@]}"
do
    echo "=================================================="
    echo " 正在处理场景：$SCENE_ID"
    echo "=================================================="

    # 自动拼接路径
    DATA=/root/project/data/HQ200_select/${SCENE_ID}
    OUTPUT_DIR=${DATA}/${RESULT_TAG}
    CKPT_PATH=CKPT_DIR/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt

    # 执行训练脚本
    python examples/gsplat/simple_trainer_difix3d.py mcmc \
        --data_dir ${DATA} \
        --data_factor ${DATA_FACTOR} \
        --result_dir ${OUTPUT_DIR} \
        --no-normalize-world-space \
        --test_every 8

    # 如果需要加载ckpt，取消下面这行注释，并把上面命令里的# --ckpt去掉
    #     --ckpt ${CKPT_PATH}

    echo -e "\n✅ 场景 $SCENE_ID 处理完成！\n"
done

echo "🎉 所有场景处理完毕！"