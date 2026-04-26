import os
import json
import torch
import torch.backends.cudnn as cudnn
from cleanfid import fid

# ==========================================
# 🛡️ 终极防爆盾：拔掉 cuDNN，禁止一切底层骚操作
# ==========================================
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 绝对限制只使用单卡

# 强行关闭 cuDNN，防止 CUDNN_STATUS_NOT_INITIALIZED 报错
cudnn.enabled = False
cudnn.benchmark = False

# 1. 核心路径配置
BASE_DIR = "/root/project/data/HQ200_select"
PARENT_DIRS = ["ground", "under_ground"]
RESULT_TAG = "recon_difix3d_realcar_v4_fix_sized"
OUTPUT_JSON = os.path.join(BASE_DIR, "metrics_kid_fid.json")

def main():
    results = {
        "Averages": {},
        "Scenes": {}
    }
    
    metrics_tracker = {
        "FID_Pred": [],
        "FID_Fixed": [],
        "KID_Pred": [],
        "KID_Fixed": []
    }

    # 安全获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 强制极小 batch_size，防止显存瞬间爆炸
    EVAL_BATCH_SIZE = 4  

    print(f"🚀 开始全局扫描并计算 FID / KID 指标... (强制关闭cuDNN, Batch Size: {EVAL_BATCH_SIZE})\n")

    for p_dir in PARENT_DIRS:
        full_p_dir = os.path.join(BASE_DIR, p_dir)
        if not os.path.exists(full_p_dir):
            continue

        for scene_name in os.listdir(full_p_dir):
            scene_path = os.path.join(full_p_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue
            
            gt_dir = os.path.join(scene_path, "images")
            novel_dir = os.path.join(scene_path, RESULT_TAG, "renders", "novel")
            
            if not os.path.exists(gt_dir) or not os.path.exists(novel_dir):
                continue
            
            steps = []
            for step_str in os.listdir(novel_dir):
                if step_str.isdigit():
                    steps.append(int(step_str))
            
            if not steps:
                continue
            
            max_step = max(steps)
            pred_dir = os.path.join(novel_dir, str(max_step), "Pred")
            fixed_dir = os.path.join(novel_dir, str(max_step), "Fixed")
            
            if not os.path.exists(pred_dir) or not os.path.exists(fixed_dir):
                continue
            
            print(f"🎯 正在计算场景: {scene_name} (最高步数: {max_step})")
            
            try:
                torch.cuda.empty_cache()

                # ==========================================
                # 强行设置 use_dataparallel=False，防止内部多卡冲突
                # ==========================================
                fid_pred = fid.compute_fid(gt_dir, pred_dir, device=device, num_workers=0, batch_size=EVAL_BATCH_SIZE, use_dataparallel=False)
                fid_fixed = fid.compute_fid(gt_dir, fixed_dir, device=device, num_workers=0, batch_size=EVAL_BATCH_SIZE, use_dataparallel=False)
                
                kid_pred = fid.compute_kid(gt_dir, pred_dir, device=device, num_workers=0, batch_size=EVAL_BATCH_SIZE, use_dataparallel=False)
                kid_fixed = fid.compute_kid(gt_dir, fixed_dir, device=device, num_workers=0, batch_size=EVAL_BATCH_SIZE, use_dataparallel=False)
                
                print(f"   📊 Pred  -> FID: {fid_pred:.2f} | KID: {kid_pred:.4f}")
                print(f"   📊 Fixed -> FID: {fid_fixed:.2f} | KID: {kid_fixed:.4f}\n")
                
                results["Scenes"][scene_name] = {
                    "step": max_step,
                    "FID_Pred": round(fid_pred, 4),
                    "FID_Fixed": round(fid_fixed, 4),
                    "KID_Pred": round(kid_pred, 6),
                    "KID_Fixed": round(kid_fixed, 6)
                }
                
                metrics_tracker["FID_Pred"].append(fid_pred)
                metrics_tracker["FID_Fixed"].append(fid_fixed)
                metrics_tracker["KID_Pred"].append(kid_pred)
                metrics_tracker["KID_Fixed"].append(kid_fixed)
                
            except Exception as e:
                print(f"❌ 计算场景 {scene_name} 时发生错误: {e}")

    # ==========================================
    # 计算均值并保存
    # ==========================================
    if len(metrics_tracker["FID_Pred"]) > 0:
        n = len(metrics_tracker["FID_Pred"])
        results["Averages"] = {
            "Total_Scenes": n,
            "FID_Pred_avg": round(sum(metrics_tracker["FID_Pred"]) / n, 4),
            "FID_Fixed_avg": round(sum(metrics_tracker["FID_Fixed"]) / n, 4),
            "KID_Pred_avg": round(sum(metrics_tracker["KID_Pred"]) / n, 6),
            "KID_Fixed_avg": round(sum(metrics_tracker["KID_Fixed"]) / n, 6)
        }
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("==================================================")
    print(f"🎉 全部计算完成！共统计了 {len(results.get('Scenes', {}))} 个场景。")
    print(f"💾 结果已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()