#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model.py
基于BC-ResNet的唤醒词检测模型训练脚本

主要功能说明
1. 数据集处理：
正样本处理：每个正样本音频会被截断或填充至窗口大小，作为一个完整样本
负样本处理：使用滑动窗口技术切分长音频，每个窗口作为一个独立的负样本
2. 模型架构：
基于BC-ResNet架构，将原多分类模型修改为二分类
保留原模型的特征提取能力，只改变输出层
3. 训练过程：
实现预热+余弦退火的学习率调度
支持数据增强（噪声添加、时间偏移等）
每隔指定epoch在开发集上评估并保存模型
4. 评估指标：
计算准确率、精确率、召回率和F1分数
记录并保存预测错误的样本信息
5. 日志和模型保存：
将训练过程记录到日志文件
保存最佳模型和阶段性检查点
保存开发集错误信息供分析
"""

import os
import json
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from tqdm import tqdm
import librosa
import sys
from sklearn.metrics import precision_recall_curve, auc, f1_score

# 导入自定义模块
from WakewordModel.detector import WakeWordDetector
from AudioProcessor.logmel import LogMelFeatureExtractor
from DataLoader.dataset import WakeWordDataset
from AudioProcessor.specaug import SpecAugmentation


# 设置随机种子确保可重复性
def set_seed(seed=42):
    random.seed(seed) # Python内置的随机数生成器种子,影响random.choice(), random.shuffle()等操作
    np.random.seed(seed) # NumPy的随机数生成器种子,影响numpy.random模块的函数
    torch.manual_seed(seed) # PyTorch的CPU随机数生成器种子,影响torch.randn(), torch.rand()等操作
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # PyTorch的GPU随机数生成器种子,影响torch.randn(), torch.rand()等操作
        torch.cuda.manual_seed_all(seed) # PyTorch的所有GPU随机数生成器种子,影响torch.randn(), torch.rand()等操作，用于多GPU训练
    torch.backends.cudnn.deterministic = True # 使CUDA卷积操作使用确定性算法，牺牲一些性能以获得完全可重复的结果
    torch.backends.cudnn.benchmark = False # 禁用CUDNN的自动调优，以确保每次运行时使用相同的算法，防止cuDNN为提高性能而选择不同算法导致结果不一致

# 设置日志记录器
def setup_logger(workspace):
    log_dir = os.path.join(workspace, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # 日志格式：时间戳、日志级别和消息内容
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 评估函数
def evaluate_on_dev(model, dataloader, device, LogMelFeature, criterion, epoch, calculate_errors=False):
    '''
    调用evaluate_on_dev函数时，epoch=epoch(迭代用的idx)+1
    '''
    
    model.eval()
    total_loss = 0
    all_scores = [] # 原始预测分数（sigmoid后的概率值）
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for inputs, labels, filenames, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 提取音频logMel特征
            inputs_logmel = LogMelFeature(inputs) # 提取音频特征
            
            # 前向传播
            outputs = model(inputs_logmel).squeeze()
            loss = criterion(outputs, labels)
            
            # 统计
            total_loss += loss.item()
            scores = torch.sigmoid(outputs).cpu().numpy()  # 保存预测分数（0.0~1.0）
            
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 使用sklearn计算PR曲线和AUCPR
    precision_values, recall_values, thresholds_pr = precision_recall_curve(all_labels, all_scores)
    aucpr = auc(recall_values, precision_values)
    
    # 计算不同阈值下的F1值
    threshold_list = np.arange(0.0, 1.05, 0.05)
    f1_scores = {}
    
    for threshold in threshold_list:
        binary_preds = (all_scores >= threshold).astype(int)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        f1_scores[f"{threshold:.2f}"] = f1
        
    best_threshold = max(f1_scores.items(), key=lambda x: x[1])[0]
    best_f1 = f1_scores[best_threshold]
    
    best_threshold_float = float(best_threshold)
    all_preds = (all_scores >= best_threshold_float).astype(int)
    # 记录错误样本,这里错误样本的判断阈值基于前面计算得到的best_f1值情况下的best_threshold_float
    errors = []
    if calculate_errors:
        for pred, label, filename, score in zip(all_preds, all_labels, all_filenames, all_scores):
            if pred != label:
                errors.append({
                    "epoch": epoch,
                    "filename": filename,
                    "predict_label": int(pred),
                    "true_label": int(label),
                    "predict_score": float(score)  # 添加预测分数，方便分析
                })
    
    results = {
        'epoch': epoch,
        'loss': avg_loss,
        'aucpr': aucpr,
        'best_threshold': float(best_threshold),
        'best_f1': best_f1,
        'f1_scores': f1_scores,
    }
    
    return results, errors

# 主训练函数
def train(args):
    # 设置随机种子，确保实验结果可复现，seed相同时每次运行结果一致
    set_seed(seed=42)
    
    # 准备目录
    train_dir = os.path.join(args.workspace, "train")
    model_dir = os.path.join(train_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(args.workspace)
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"[INFO] Using device: {device}")
    
    # 加载数据集信息
    logger.info("======== Dataset Loading ========")
    dataset_info_path = os.path.join(args.workspace, "dataset", "dataset_info.json")
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    # 设置窗口大小和步长（单位为采样点数，音频在generate_dataset步骤中被重采样为16000hz，故转换时乘以16000）
    # 窗口大小 = 正集的训练集的percentile_90_duration，能将正集的90%样本包含在窗口内
    # 窗口步长 = 窗口大小 * 步长比例
    pos_avg_duration = dataset_info["positive"]["train"]["percentile_90_duration"]
    window_size = int(pos_avg_duration * 16000)  # 转为采样点数
    window_stride = int(window_size * args.window_stride_ratio) # 单位为采样点数
    
    logger.info(f"[INFO] Positive average duration: {pos_avg_duration:.4f}s")
    logger.info(f"[INFO] Window size: {window_size} samples ({window_size/16000:.4f}s)")
    logger.info(f"[INFO] Window stride: {window_stride} samples ({window_stride/16000:.4f}s)")

    # 加载训练用数据集
    sample_rate = 16000
    train_dataset = WakeWordDataset(args.workspace, "train", window_size, window_stride, sample_rate) # 加载train数据集
    dev_dataset = WakeWordDataset(args.workspace, "dev", window_size, window_stride, sample_rate) # 加载dev数据集
    
    train_loader = DataLoader(
        train_dataset, 
        # 要加载的数据集对象，必须满足：
        # 1. 继承torch.utils.data.Dataset类
        # 2. 实现两个必要方法:
        #     __len__(self): 返回数据集大小
        #     __getitem__(self, idx): 根据索引返回单个样本
        batch_size=args.batch_size, # 每个批次的样本数量
        shuffle=True, # 是否在每个epoch开始时打乱数据集（训练时通常设置为True，模型训练时正负样本交叉输入）
        num_workers=18, # 用于并行加载数据的子进程数量，提高数据加载效率
        pin_memory=args.use_gpu, # 是否将数据放入CUDA固定内存，可加速GPU训练时的数据传输
        persistent_workers=False, # 保持工作进程存活，工作进程在整个DataLoader生命周期内保持活跃，避免了每个epoch结束时销毁进程、新epoch开始时重新创建进程的开销
        prefetch_factor=4 # 预取因子，默认值: 2，每个工作进程预加载的数据批次倍数。每个工作进程预取的样本数 = prefetch_factor * batch_size。减少GPU等待时间: GPU处理完当前批次后，下一批次数据已准备就绪。增加内存占用: 每个工作进程都会缓存更多数据，总内存使用量增加
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=18,
        pin_memory=args.use_gpu,
        persistent_workers=False, # 保持工作进程存活，工作进程在整个DataLoader生命周期内保持活跃，避免了每个epoch结束时销毁进程、新epoch开始时重新创建进程的开销
        prefetch_factor=4 # 预取因子，默认值: 2，每个工作进程预加载的数据批次倍数。每个工作进程预取的样本数 = prefetch_factor * batch_size。减少GPU等待时间: GPU处理完当前批次后，下一批次数据已准备就绪。增加内存占用: 每个工作进程都会缓存更多数据，总内存使用量增加
    )
    
    # check dataset size
    logger.info(f"[INFO] Train dataset count(After delploy sliding window on neg dataset)")
    logger.info(f"[INFO] Train samples count: {len(train_dataset)}")
    logger.info(f"[INFO] Dev samples count: {len(dev_dataset)}")
    logger.info(f"========================================")
    
    # 初始化模型
    logger.info(f"======== Training Preparation ========")
    model = WakeWordDetector(model_version=args.model_version, spec_group_num = args.spec_group_num).to(device)
    logger.info(f"[INFO] Initialized bcresnet with version {args.model_version}")
    
    # 定义损失函数
    # 为正样本添加权重，值可以根据不平衡程度调整
    # 由于实际训练时负样本为正样本的10~50倍，可以设置为20或接近的值
    pos_weight = torch.tensor([10.0])  # 可以根据实际情况进行调整
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device) # 当你使用pos_weight参数时，这个损失函数会在内部保存这个权重张量。当模型和数据都在GPU（或其他特定设备）上时，损失函数内部的张量也需要在同一设备上，否则会导致计算错误。故需要to(device)将pos_weight转移到相同的设备上。
    # 定义优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0,  # 初始化为0，在训练中动态调整
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )
    
    # 学习率调度参数
    n_step_warmup = len(train_loader) * args.warmup_epoch # 使用len()函数获取train_loader的长度（即每个epoch的批次数），乘以args.warmup_epoch（预热epoch数）得到预热阶段的总步数（总批次数）
    total_iter = len(train_loader) * args.total_epochs # 训练的总迭代次数（总共需要遍历的批次数）
    
    # 定义音频logMel特征提取器
    LogMelFeature = LogMelFeatureExtractor(
        device=device,
        sample_rate=sample_rate, 
        win_length=480, 
        hop_length=160, 
        n_fft=512, 
        n_mels=40
    )
    
    # 初始化SpecAugmentation
    # 参数可以根据你的数据集和模型进行调整
    # freq_mask_param: 频率掩码的最大宽度 (F)。对于n_mels=40，可以尝试5-10。
    # time_mask_param: 时间掩码的最大宽度 (T)。取决于你的音频帧数，可以尝试10-30。
    # num_freq_masks: 应用的频率掩码数量 (m_F)。通常为1或2。
    # num_time_masks: 应用的时间掩码数量 (m_T)。通常为1或2。
    spec_augmenter = SpecAugmentation(
        freq_mask_param=8,    # 例如，最大8个mel bin被遮盖
        time_mask_param=20,   # 例如，最大20个时间帧被遮盖
        num_freq_masks=2,     # 应用1个频率遮盖
        num_time_masks=2      # 应用1个时间遮盖
    ).to(device) # 将SpecAugmentation模块也移动到相应的设备
    
    # 打印训练参数及训练配置
    logger.info(f"[INFO] Training parameters: ")
    logger.info(f"++++++++ workspace settings ++++++++")
    logger.info(f"  - Workspace: {args.workspace}")
    logger.info(f"++++++++ model settings +++++++++")
    logger.info(f"  - Model: BC-ResNet")
    logger.info(f"  - Model version: {args.model_version}")
    logger.info(f"++++++++ optimizer settings +++++++++")
    logger.info(f"  - optimizer: torch.optim.SGD") 
    logger.info(f"  - Learning rate: {args.init_lr}")
    logger.info(f"  - Learning rate lower limit: {args.lr_lower_limit}")
    logger.info(f"  - Weight decay: {args.weight_decay}")
    logger.info(f"  - Momentum: {args.momentum}")
    logger.info(f"++++++++ lossfunc settings +++++++++")
    logger.info(f"  - Loss function: BCEWithLogitsLoss")
    logger.info(f"++++++++ training settings +++++++++")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Warmup epochs: {args.warmup_epoch}")
    logger.info(f"  - Total epochs: {args.total_epochs}")
    logger.info(f"  - Warmup steps: {n_step_warmup}")
    logger.info(f"  - Total iterations: {total_iter}")
    logger.info(f"  - Window size: {window_size}")
    logger.info(f"  - Window stride: {window_stride}")
    logger.info(f"  - Device: {device}")
    logger.info(f"++++++++ augmentation settings +++++++++")
    logger.info(f"====================================")
    
    
    # 训练准备
    best_metrics = {
        'epoch': 0,
        'aucpr': 0,
        'best_f1': 0,
        'best_threshold': 0.5,
        'loss': float('inf'),
        'f1_scores': {}
    }
    all_dev_errors = []
    iterations = 0
    
    logger.info("======== Training Start ========")
    # 使用单个进度条显示整个训练过程
    with tqdm(total=total_iter, desc="Training") as pbar:
        for epoch in range(args.total_epochs):
            model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (inputs, labels, filenames, aug_types) in enumerate(train_loader):
                # 当前【批次】的迭代，每迭代一个datasetloader中的一个batch，iter++
                # 当使用 for inputs, labels, _ in train_loader: 迭代时，DataLoader会自动对批次中的样本进行合并：
                # inputs: 从[1, window_size] 变为 [batch_size, 1, window_size]
                
                # 更新学习率
                iterations += 1
                if iterations < n_step_warmup:
                    # 预热阶段，线性增加学习率
                    lr = args.init_lr * iterations / n_step_warmup
                else:
                    # 余弦退火阶段，使用余弦函数调整学习率
                    lr = args.lr_lower_limit + 0.5 * (args.init_lr - args.lr_lower_limit) * (
                        1 + np.cos(np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup))
                    )
                
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr # 更新当前学习率（给优化器看）
                
                # 处理数据
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 数据偏移
                # inputs = AudioOffset(inputs) # 随机时间偏移
                
                # 提取音频logMel特征
                inputs_logmel = LogMelFeature(inputs) # 提取音频特征
                
                # 应用SpecAugment
                # 创建一个布尔掩码，标记哪些样本需要SpecAugment
                needs_specaug_mask = torch.tensor([aug_type == 'specaugment' for aug_type in aug_types], device=device)
                
                # 如果批次中至少有一个样本需要SpecAugment
                if needs_specaug_mask.any():
                    # 选择需要增强的样本
                    samples_to_augment = inputs_logmel[needs_specaug_mask]
                    # 应用增强
                    augmented_samples = spec_augmenter(samples_to_augment)
                    # 将增强后的样本放回原位
                    # inputs_logmel是一个张量，可以直接通过布尔掩码进行索引和赋值
                    inputs_logmel[needs_specaug_mask] = augmented_samples
                
                # 前向传播
                outputs = model(inputs_logmel).squeeze()
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计在当前训练批次上的损失和准确率（使用固定阈值0.5）
                epoch_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.75).float()
                correct = (preds == labels).sum().item()
                epoch_correct += correct
                epoch_total += len(labels)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{args.total_epochs}",
                    "loss": f"{loss.item():.4f}",
                    "acc(threshold=0.75)": f"{correct/len(labels):.4f}",
                    "lr": f"{lr:.6f}"
                })
            
            # 计算训练指标
            train_loss = epoch_loss / len(train_loader) # 当前epoch的平均损失（每个epoch都会遍历所有训练数据）
            train_acc = epoch_correct / epoch_total # 当前epoch的平均准确率（每个epoch都会遍历所有训练数据）
            
            # 记录日志
            logger.info(f"[INFO] Epoch {epoch+1}/{args.total_epochs} - Train Loss: {train_loss:.4f}, Acc(threshold=0.75): {train_acc:.4f}")
            
            # 在dev集上评估（每eval_on_dev_epoch_stride步评估一次，最后一个epoch再评估一次）
            if (epoch + 1) % args.eval_on_dev_epoch_stride == 0 or epoch == args.total_epochs - 1:
                dev_metrics, dev_errors = evaluate_on_dev(
                    model, dev_loader, device, LogMelFeature, criterion, epoch+1, calculate_errors=True
                )
                
                # 记录错误
                all_dev_errors.extend(dev_errors)
                
                # 记录日志
                logger.info(f"[INFO] Epoch {dev_metrics['epoch']} : Evaluate on Dev - " + 
                           f"Loss: {dev_metrics['loss']:.4f}, AUCPR: {dev_metrics['aucpr']:.6f}, " + 
                           f"Best F1: {dev_metrics['best_f1']:.4f} @ threshold={dev_metrics['best_threshold']:.2f} " +
                           f"F1 Score: {dev_metrics['f1_scores']}"
                           )
                
                
                # 保存当前模型
                current_model_path = os.path.join(model_dir, f"model_{epoch+1}.pt")
                torch.save({
                    'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'window_size': window_size,
                    'window_stride': window_stride,
                    'model_version': args.model_version, 
                    'spec_group_num': args.spec_group_num, 
                    'metrics': dev_metrics
                }, current_model_path)
                
                # 更新最佳模型，更新依据：F1分数
                if dev_metrics['aucpr'] > best_metrics.get('aucpr', 0):
                    best_metrics = {**dev_metrics, 'epoch': epoch + 1}
                    best_model_path = os.path.join(model_dir, "model_best.pt")
                    torch.save({
                        'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'window_size': window_size,
                        'window_stride': window_stride,
                        'model_version': args.model_version, 
                        'spec_group_num': args.spec_group_num, 
                        'metrics': dev_metrics
                    }, best_model_path)
                    logger.info(f"[INFO] New best model saved! AUCPR: {best_metrics['aucpr']:.6f}")
    
    # 保存开发集错误记录
    dev_errors_path = os.path.join(train_dir, "dev_errors.json")
    with open(dev_errors_path, 'w') as f:
        json.dump(all_dev_errors, f, indent=2)
    
    # 训练完成日志
    logger.info("[INFO] Training completed!")
    logger.info(f"[INFO] Best model (epoch {best_metrics['epoch']}):")
    logger.info(f"  AUCPR: {best_metrics['aucpr']:.6f}")
    logger.info(f"  Best F1: {best_metrics['best_f1']:.4f} @ threshold={best_metrics['best_threshold']:.2f}")
    logger.info(f"  Loss: {best_metrics['loss']:.4f}")
    logger.info(f"  F1 Score: {best_metrics['f1_scores']}")
    logger.info(f"[INFO] Development errors saved to {dev_errors_path}")
    logger.info(f"====================================")

def parse_args():
    parser = argparse.ArgumentParser(description="Train wake word detection model")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    parser.add_argument("--model_version", type=float, default=1.0, choices=[1, 1.5, 2, 3, 6, 8], help="BC-ResNet model version (tau value), chooce betweeen 1, 1.5, 2, 3, 6, 8")
    parser.add_argument("--spec_group_num", type=int, default=5, help="Number of spec groups")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--window_stride_ratio", type=float, default=0.5, help="Window stride as ratio of window size")
    parser.add_argument("--total_epochs", type=int, default=200, help="Total number of training epochs")
    parser.add_argument("--warmup_epoch", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--eval_on_dev_epoch_stride", type=int, default=5, help="Evaluate on dev set every N epochs")
    parser.add_argument("--init_lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--lr_lower_limit", type=float, default=0, help="Lower limit for learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)