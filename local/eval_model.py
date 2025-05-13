import os
import json
import argparse
import logging
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# 导入自定义模块
from WakewordModel.detector import WakeWordDetector
from AudioProcessor.logmel import LogMelFeatureExtractor
from DataLoader.dataset import WakeWordDataset # Uses the modified version

# 设置随机种子确保可重复性 (optional for eval, but good practice)
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置日志记录器
def setup_logger(workspace):
    log_dir = os.path.join(workspace, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "eval_log.log")
    
    logger = logging.getLogger("eval_logger") # Use a unique name
    logger.setLevel(logging.INFO)
    
    if logger.handlers: # Clear existing handlers for this logger instance
        logger.handlers.clear()
            
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 在测试集上评估函数
def evaluate_on_test(args, logger):
    set_seed(seed=42)
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"[INFO] Using device: {device}")

    # 创建test输出目录
    test_output_dir = os.path.join(args.workspace, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # 加载数据集信息
    dataset_info_path = os.path.join(args.workspace, "dataset", "dataset_info.json")
    if not os.path.exists(dataset_info_path):
        logger.error(f"[ERROR] dataset_info.json not found at {dataset_info_path}")
        return
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    negative_test_duration_seconds = dataset_info.get("negative", {}).get("test", {}).get("total_duration", 0)
    if negative_test_duration_seconds == 0:
        logger.warning("[WARNING] Negative test duration is 0 from dataset_info.json. FAPH will be inf or 0.")
        negative_test_duration_hours = 0
    else:
        negative_test_duration_hours = negative_test_duration_seconds / 3600.0
    logger.info(f"[INFO] Negative test set duration: {negative_test_duration_seconds:.2f} seconds ({negative_test_duration_hours:.2f} hours)")


    # 加载模型
    model_path = os.path.join(args.workspace, "train", "model", "model_best.pt")
    if not os.path.exists(model_path):
        logger.error(f"[ERROR] Best model not found at {model_path}")
        return

    logger.info(f"[INFO] Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    window_size = checkpoint['window_size']
    window_stride = checkpoint['window_stride']
    model_version = checkpoint['model_version']
    spec_group_num = checkpoint['spec_group_num']
    
    model = WakeWordDetector(model_version=model_version, spec_group_num=spec_group_num).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"[INFO] Model loaded. Trained for {checkpoint.get('epoch', 'N/A')} epochs.")
    logger.info(f"[INFO] Model window_size: {window_size}, window_stride: {window_stride}")
    logger.info(f"[INFO] Model version from checkpoint: {model_version}") 
    logger.info(f"[INFO] Model spec_group_num from checkpoint: {spec_group_num}")

    sample_rate = 16000 
    LogMelFeature = LogMelFeatureExtractor(
        device=device, sample_rate=sample_rate, win_length=480, 
        hop_length=160, n_fft=512, n_mels=40
    )

    logger.info("[INFO] Loading test dataset...")
    test_dataset = WakeWordDataset(args.workspace, "test", window_size, window_stride, sample_rate)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=18,
        pin_memory=args.use_gpu, # Modified num_workers from 18 to args.num_workers
        persistent_workers=False,
        prefetch_factor=4
    )
    logger.info(f"[INFO] Test dataset loaded with {len(test_dataset)} samples.")

    criterion = nn.BCEWithLogitsLoss().to(device)

    total_loss = 0
    all_scores = []
    all_labels = []
    all_filenames = []

    logger.info("[INFO] Starting evaluation on test set...")
    with torch.no_grad():
        for inputs, labels, filenames, _ in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_logmel = LogMelFeature(inputs)
            outputs = model(inputs_logmel).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            scores = torch.sigmoid(outputs).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 计算指标
    precision_values, recall_values, pr_thresholds = precision_recall_curve(all_labels, all_scores)
    aucpr = auc(recall_values, precision_values)
    
    # 为了FRR-FAPH曲线，我们需要FPR和TPR (TPR=Recall, FRR=1-TPR)
    # roc_curve的thresholds[0]是max(scores)+1，不用于计算，所以通常忽略第一个元素
    fpr_values, tpr_values, roc_thresholds = roc_curve(all_labels, all_scores)

    results_per_threshold = []
    frr_values = []
    faph_values = []
    
    # 使用PR曲线的阈值或者ROC曲线的阈值的一个子集进行计算，确保阈值覆盖合理范围
    # PR曲线的阈值通常更适合二分类不平衡问题
    # 我们将使用PR曲线的阈值，并补充一些标准阈值
    eval_thresholds = np.sort(np.unique(np.concatenate([pr_thresholds, np.arange(0.0, 1.05, 0.05)])))

    num_positive_samples = np.sum(all_labels == 1)
    num_negative_samples = np.sum(all_labels == 0)

    for thresh in eval_thresholds:
        if thresh < 0 or thresh > 1: continue # 确保阈值在[0,1]
        
        binary_preds = (all_scores >= thresh).astype(int)
        
        tp = np.sum((binary_preds == 1) & (all_labels == 1))
        fp = np.sum((binary_preds == 1) & (all_labels == 0))
        fn = np.sum((binary_preds == 0) & (all_labels == 1))
        # tn = np.sum((binary_preds == 0) & (all_labels == 0)) # Not directly used in P, R, F1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # Same as TPR
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # FRR: false rejection rate
        # FRR = FN / num_positive_samples
        frr = fn / num_positive_samples if num_positive_samples > 0 else 0 # False Rejection Rate
        # FRR表示唤醒词被系统错误拒绝的比例
        # 计算公式：误拒数(FN) / 正样本总数
        # 衡量了系统漏检率，即有唤醒词但未被检测到的情况
        
        # FAPH: False Alarms Per Hour
        # FAPH = FA / negative_test_duration_hours
        faph = fp / negative_test_duration_hours if negative_test_duration_hours > 0 else float('inf')
        # FAPH表示每小时系统产生的误报警次数
        # 计算公式：误报数(FP) / 负样本总时长(小时)
        # 衡量了系统在实际使用中每小时可能产生的误唤醒次数
        
        results_per_threshold.append({
            "threshold": float(thresh),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "frr": float(frr),
            "false_alarms": int(fp),
            "faph": float(faph)
        })
        frr_values.append(frr)
        faph_values.append(faph)

    # 找到最佳F1对应的阈值和结果
    best_f1_result = max(results_per_threshold, key=lambda x: x['f1_score'])
    best_threshold_float = best_f1_result['threshold']
    best_f1 = best_f1_result['f1_score']

    logger.info("======== Test Set Evaluation Results ========")
    logger.info(f"[RESULT] Average Loss: {avg_loss:.4f}")
    logger.info(f"[RESULT] AUCPR: {aucpr:.6f}")
    logger.info(f"[RESULT] Best F1 Score: {best_f1:.4f} @ Threshold={best_threshold_float:.4f}")
    
    # 保存详细结果到JSON
    output_results_json_path = os.path.join(test_output_dir, "result.json")
    final_results_data = {
        "summary": {
            "avg_loss": avg_loss,
            "aucpr": aucpr,
            "best_f1_score": best_f1,
            "best_f1_threshold": best_threshold_float,
            "negative_test_duration_hours": negative_test_duration_hours,
            "num_positive_samples_test": int(num_positive_samples),
            "num_negative_samples_test": int(num_negative_samples)
        },
        "metrics_per_threshold": results_per_threshold,
        "pr_curve": {
            "precision": precision_values.tolist(),
            "recall": recall_values.tolist(),
            # pr_thresholds includes an extra value at the end, often 1.
            # The PR curve is typically plotted for (recall_values[i], precision_values[i])
        },
        "frr_faph_curve": { # For plotting, we need to sort by FAPH or FRR
            "frr": frr_values, # These are already aligned with eval_thresholds
            "faph": faph_values
        }
    }
    with open(output_results_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results_data, f, indent=2, ensure_ascii=False)
    logger.info(f"[INFO] Detailed test results saved to {output_results_json_path}")

    # 记录错误样本
    errors = []
    all_preds_at_best_f1_thresh = (all_scores >= best_threshold_float).astype(int)
    for pred, label, filename, score_val in zip(all_preds_at_best_f1_thresh, all_labels, all_filenames, all_scores):
        if pred != label:
            errors.append({
                "filename": filename,
                "predict_label": int(pred),
                "true_label": int(label),
                "predict_score": float(score_val)
            })
    
    test_errors_path = os.path.join(test_output_dir, "test_errors.json")
    with open(test_errors_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    logger.info(f"[INFO] Test errors (at best F1 threshold) saved to {test_errors_path} ({len(errors)} errors)")
    
    # 生成并保存图像
    pdf_path = os.path.join(test_output_dir, "result.pdf")
    with PdfPages(pdf_path) as pdf:
        # 1. PR Curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall_values, precision_values, marker='.', label=f'AUCPR = {aucpr:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()
        logger.info("[INFO] PR curve generated.")

        # 2. FRR vs FAPH Curve
        # Sort by FAPH for a smoother plot, or plot as is if thresholds are monotonic
        # For plotting, it's often better to have FAPH on x-axis if it's the independent variable of interest
        # We will use the faph_values and frr_values calculated earlier, which are aligned by threshold
        
        # Filter out inf FAPH values for plotting if any, and corresponding FRR
        plot_faph = []
        plot_frr = []
        for faph_val, frr_val in zip(faph_values, frr_values):
            if faph_val != float('inf'):
                plot_faph.append(faph_val)
                plot_frr.append(frr_val)
        
        if plot_faph and plot_frr:
            # 排序以便绘图
            sorted_indices = np.argsort(plot_faph)
            sorted_faph = np.array(plot_faph)[sorted_indices]
            sorted_frr = np.array(plot_frr)[sorted_indices]

            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 完整图 - 使用对数尺度
            ax1.plot(sorted_faph, sorted_frr, marker='.')
            ax1.set_xlabel('False Alarm Per Hour (FAPH)')
            ax1.set_ylabel('False Rejection Rate (FRR)')
            ax1.set_title('FRR vs FAPH')
            ax1.grid(True)
            
            # 聚焦图 - 只显示低FAPH区域
            focus_limit = 200  # 聚焦区域的FAPH上限
            focus_indices = [i for i, v in enumerate(sorted_faph) if v <= focus_limit]
            if focus_indices:
                focus_faph = sorted_faph[focus_indices]
                focus_frr = sorted_frr[focus_indices]
                ax2.plot(focus_faph, focus_frr, marker='.')
                ax2.set_xlabel('False Alarm Per Hour (FAPH)')
                ax2.set_ylabel('False Rejection Rate (FRR)')
                ax2.set_title(f'FRR vs FAPH (FAPH ≤ {focus_limit})')
                ax2.grid(True)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            logger.info("[INFO] FRR vs FAPH curves generated (both full and focused views).")
        else:
            logger.warning("[WARNING] Not enough data points to plot FRR vs FAPH curve (possibly all FAPH are inf).")

    logger.info(f"[INFO] Evaluation plots saved to {pdf_path}")
    logger.info("============================================")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate wake word detection model on test set")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for evaluation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(args.workspace)
    
    logger.info("======== Evaluation Script Start ========")
    logger.info(f"Arguments: {vars(args)}")
    
    evaluate_on_test(args, logger)
    
    logger.info("======== Evaluation Script End ========")