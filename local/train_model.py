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
from tqdm import tqdm
import librosa
import sys

# 导入自定义模块
from WakewordModel.bcresnet import BCResNets
from WakewordModel.detector import WakeWordDetector
from AudioProcessor.logmel import LogMelFeatureExtractor
from DataLoader.dataset import WakeWordDataset


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

def calculate_eer(scores, labels):
    # 设置多个阈值
    thresholds = np.linspace(0, 1, 100)
    fars = []
    frrs = []
    
    for threshold in thresholds:
        preds = (scores > threshold).astype(int)
        
        # 计算混淆矩阵元素
        tp = np.sum((preds == 1) & (labels == 1))
        fn = np.sum((preds == 0) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        
        # 计算FRR和FAR
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        fars.append(far)
        frrs.append(frr)
    
    # 找到FAR和FRR最接近的点
    abs_diffs = np.abs(np.array(fars) - np.array(frrs))
    min_index = np.argmin(abs_diffs)
    eer = (fars[min_index] + frrs[min_index]) / 2
    
    return eer, thresholds[min_index]

# 评估函数
def evaluate_on_dev(model, dataloader, device, LogMelFeature, criterion, epoch, calculate_errors=False):
    '''
    调用evaluate_on_dev函数时，epoch=epoch(迭代用的idx)+1
    '''
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for inputs, labels, filenames in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 提取音频logMel特征
            inputs_logmel = LogMelFeature(inputs) # 提取音频特征
            
            # 前向传播
            outputs = model(inputs_logmel).squeeze()
            loss = criterion(outputs, labels)
            
            # 统计
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.55).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    correct = sum(np.array(all_preds) == np.array(all_labels))
    accuracy = correct / len(all_labels)
    
    # 计算精确率和召回率
    # 真正例：模型预测为1（有唤醒词），实际标签也是1（有唤醒词）
    tp = sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
    # 假正例：模型预测为1（有唤醒词），实际标签是0（没有唤醒词）
    fp = sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
    # 真负例：模型预测为0（没有唤醒词），实际标签也是0（没有唤醒词）
    tn = sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
    # 假负例：模型预测为0（没有唤醒词），实际标签是1（有唤醒词）
    fn = sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))
    
    # 精确率：TP / (TP + FP)。在所有被预测为"有唤醒词"的样本中，真正包含唤醒词的比例
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    # 召回率：TP / (TP + FN)。在所有实际包含唤醒词的样本中，被正确预测为"有唤醒词"的比例
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    # FRR：False Rejection Rate，误拒绝率，FRR = FN / (TP + FN)，在所有包含唤醒词的正样本中，错误拒绝的比例
    frr = fn / (tp + fn) if tp + fn > 0 else 0
    # FAR：False Acceptance Rate，误接受率，FAR = FP / (TN + FP)，在所有不包含唤醒词的负样本中，错误接受的比例
    far = fp / (tn + fp) if tn + fp > 0 else 0
    # F1分数：2 * (精确率 * 召回率) / (精确率 + 召回率)。综合考虑精确率和召回率的指标
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # 记录错误样本
    errors = []
    if calculate_errors:
        for pred, label, filename in zip(all_preds, all_labels, all_filenames):
            if pred != label:
                errors.append({
                    "epoch": epoch,
                    "filename": filename,
                    "predict_label": int(pred),
                    "true_label": int(label)
                })
    
    return {
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }, errors

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
        num_workers=6, # 用于并行加载数据的子进程数量，提高数据加载效率
        pin_memory=args.use_gpu # 是否将数据放入CUDA固定内存，可加速GPU训练时的数据传输
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=args.use_gpu
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
    # 由于实际训练时负样本为正样本的50~100倍，可以设置为100或接近的值
    pos_weight = torch.tensor([50.0])  # 可以根据实际情况进行调整
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
    
    # 初始化预处理器
    # 寻找噪声目录
    noise_dir = "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/bcresnet/data/speech_commands_v0.02/_background_noise_", # 这里的输入的noise_dir结构应形如speech_commands_v0.02/_background_noise_一样，其中包含多个.wav噪声文件   
        
    # 针对不同的模型版本，设置不同的频谱增强参数
    frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}
    
    # 定义音频logMel特征提取器
    LogMelFeature = LogMelFeatureExtractor(
        device=device,
        sample_rate=sample_rate, 
        win_length=480, 
        hop_length=160, 
        n_fft=512, 
        n_mels=40
    )
    
    # 预处理器，需要修改，beta v1.0暂不添加预处理，只使用logMel特征提取器提取音频特征
    # preprocess = Preprocess(
    #     noise_loc = noise_dir,
    #     device = device,
    #     hop_length=160, # 提取logMel时的步长
    #     win_length=480, # 提取logMel时的窗口长度
    #     n_fft=512, # 提取logMel时的FFT大小
    #     n_mels=40, # 提取logMel时的梅尔频率数量
    #     specaug = (args.model_version >= 1.5),
    #     sample_rate = 16000, # 采样率
    #     frequency_masking_para = frequency_masking_para[args.model_version],
    #     time_masking_para=20, # 频域增强时，时间掩码的最大宽度
    #     frequency_mask_num=2, # 频域增强时，频率掩码的数量
    #     time_mask_num=2 # 频域增强时，时间掩码的数量
    # )
    # 训练过程中，数据增强使用样例：
    # inputs = self.preprocess_train(inputs, labels, augment=True) # 预处理训练数据，进行数据增强，包括1. 噪声增强 2. 时间偏移 3. 频谱增强（具体是否执行，依据选择的BC-Resnet版本）
    # outputs = self.model(inputs) # 得到模型输出，这里的输出是一个 batch 的预测结果，形状是[batch_size, num_classes]
    
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
    logger.info(f"  - Frequency masking parameter: {frequency_masking_para[args.model_version]}")
    logger.info(f"  - Noise directory: {noise_dir}")
    logger.info(f"  - SpecAugment: {args.model_version >= 1.5}")
    logger.info(f"====================================")
    
    
    # 训练准备
    best_metrics = {"f1": 0, "accuracy": 0, "epoch": 0}
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
            
            for inputs, labels, _ in train_loader:
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
                
                # 数据增强
                # inputs_aug = AudioAugmentation(inputs_logmel)
                
                # 前向传播
                outputs = model(inputs_logmel).squeeze()
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计在当前训练批次上的损失和准确率（使用固定阈值0.5）
                epoch_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.55).float()
                correct = (preds == labels).sum().item()
                epoch_correct += correct
                epoch_total += len(labels)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{args.total_epochs}",
                    "loss": f"{loss.item():.4f}",
                    "acc(threshold=0.55)": f"{correct/len(labels):.4f}",
                    "lr": f"{lr:.6f}"
                })
            
            # 计算训练指标
            train_loss = epoch_loss / len(train_loader) # 当前epoch的平均损失（每个epoch都会遍历所有训练数据）
            train_acc = epoch_correct / epoch_total # 当前epoch的平均准确率（每个epoch都会遍历所有训练数据）
            
            # 记录日志
            logger.info(f"[INFO] Epoch {epoch+1}/{args.total_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # 在dev集上评估（每eval_on_dev_epoch_stride步评估一次，最后一个epoch再评估一次）
            if (epoch + 1) % args.eval_on_dev_epoch_stride == 0 or epoch == args.total_epochs - 1:
                dev_metrics, dev_errors = evaluate_on_dev(
                    model, dev_loader, device, LogMelFeature, criterion, epoch+1, calculate_errors=True
                )
                
                # 记录错误
                all_dev_errors.extend(dev_errors)
                
                # 记录日志
                logger.info(f"[INFO] Epoch {dev_metrics['epoch']} : Evaluate on Dev - Loss: {dev_metrics['loss']:.4f}, Acc: {dev_metrics['accuracy']:.4f}, " + 
                           f"Prec: {dev_metrics['precision']:.4f}, Rec: {dev_metrics['recall']:.4f}, F1: {dev_metrics['f1']:.4f}")
                
                # 保存当前模型
                current_model_path = os.path.join(model_dir, f"model_{epoch+1}.pt")
                torch.save({
                    'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'window_size': window_size,
                    'window_stride': window_stride,
                    'metrics': dev_metrics
                }, current_model_path)
                
                # 更新最佳模型，更新依据：F1分数
                if dev_metrics['f1'] > best_metrics['f1']:
                    best_metrics = {**dev_metrics, 'epoch': epoch + 1}
                    best_model_path = os.path.join(model_dir, "model_best.pt")
                    torch.save({
                        'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'window_size': window_size,
                        'window_stride': window_stride,
                        'metrics': dev_metrics
                    }, best_model_path)
                    logger.info(f"[INFO] New best model saved! F1: {best_metrics['f1']:.4f}")
    
    # 保存开发集错误记录
    dev_errors_path = os.path.join(train_dir, "dev_errors.json")
    with open(dev_errors_path, 'w') as f:
        json.dump(all_dev_errors, f, indent=2)
    
    # 训练完成日志
    logger.info("[INFO] Training completed!")
    logger.info(f"[INFO] Best model (epoch {best_metrics['epoch']}):")
    logger.info(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_metrics['recall']:.4f}")
    logger.info(f"  F1: {best_metrics['f1']:.4f}")
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