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
from wakeword_model_class.bcresnet import BCResNets
from dataset_class.preprocessor import Preprocess

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

# 自定义数据集类
class WakeWordDataset(Dataset):
    """
    唤醒词数据集，处理正样本和负样本，应用滑动窗口
    workspace: 工作目录
    split: 数据集划分（"train"/"dev"/"test"）
    window_size: 窗口大小（单位：采样点数）
    window_stride: 窗口步长（单位：采样点数），= window_size * window_stride_ratio
    
    其中内部函数以"_"开头，表示私有函数
        1. 访问控制提示：告诉其他开发者该方法主要供类内部使用，不建议在类外部直接调用
        2. 非公开API：表明该方法不是类公开API的一部分，可能在未来版本中更改而不另行通知
        3. 使用from module import *时，以下划线开头的名称不会被导入
    """
    def __init__(self, workspace, split, window_size, window_stride):
        self.workspace = workspace
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.sr = 16000
        
        # 数据路径
        self.pos_audio_dir = os.path.join(workspace, "dataset", "positive", "audio")
        self.neg_audio_dir = os.path.join(workspace, "dataset", "negative", "audio")
        
        # 加载JSONL文件
        pos_jsonl_path = os.path.join(workspace, "dataset", "positive", f"pos_{split}.jsonl")
        neg_jsonl_path = os.path.join(workspace, "dataset", "negative", f"neg_{split}.jsonl")
        
        self.pos_samples = self._load_jsonl(pos_jsonl_path)
        self.neg_samples = self._load_jsonl(neg_jsonl_path)
        
        # 预处理正负样本
        self._preprocess_samples() 
        # 将所有的正负样本都加载到self.samples中
        # 正负样本区别在于label。且负样本多了start_idx，start_time，end_idx，end_time和seg_duration字段（对每个负样本使用滑动窗口切分成多个小负样本片段）
        
    def _load_jsonl(self, file_path):
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)
        return samples
    
    def _preprocess_samples(self):
        """
        处理正样本和负样本，为负样本生成滑动窗口索引
        
        对于所有正样本，一个正样本音频文件直接作为一个正样本（考虑到每个正样本音频文件只包含唤醒词本身）
        对于所有负样本，应用滑动窗口技术，将长音频切分为多个小片段作为多个负样本
        """
        self.samples = []
        
        # 处理正样本 - 每个音频文件作为一个样本
        for sample in self.pos_samples:
            self.samples.append({
                'type': 'positive',
                'filename': sample['filename'],
                'text': sample['text'],
                'label': 1, # 1 means contains wakeword, 0 means not
                'duration': sample['duration'] # 单位：秒
            })
        
        # 处理负样本 - 应用滑动窗口
        for sample in self.neg_samples:
            audio_path = os.path.join(self.neg_audio_dir, sample['filename'])
            audio, _ = librosa.load(audio_path, sr=self.sr) # librosa.load() 返回的 audio 是一个包含音频采样值的 NumPy 一维数组，及该音频文件的采样率（sr，本例中未被使用）
            audio_len = len(audio) # 这里的 audio_len 是采样点数量，而不是音频的持续时间（秒）。
            
            # 计算可以提取的窗口数量
            num_windows = max(1, (audio_len - self.window_size) // self.window_stride + 1)
            # 计算原理：
            # 1. 除以 self.window_stride（整除，舍弃小数部分）获得可能的位置数量
            # +1 是因为需要包含起始位置（第一个窗口）
            # max(1, ...) 确保即使音频小于窗口大小，也至少返回1个窗口
            
            for i in range(num_windows): # 这里的i从0开始，表示第i个窗口。遍历直至i<num_windows
                start_idx = i * self.window_stride
                end_idx = min(start_idx + self.window_size, audio_len)
                # 这里的start_idx及end_idx是采样点索引（第x个采样点），而不是时间索引（秒）
                
                self.samples.append({
                    'type': 'negative',
                    'filename': sample['filename'],
                    'text': sample['text'],
                    'label': 0, # 1 means contains wakeword, 0 means not
                    'start_idx': start_idx, # 单位：采样点
                    'start_time': start_idx / self.sr, # 单位：秒
                    'end_idx': end_idx, # 单位：采样点
                    'end_time': end_idx / self.sr, # 单位：秒
                    'seg_duration': (end_idx - start_idx) / self.sr, # 单位：秒
                    'duration': sample['duration']
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if sample['type'] == 'positive':
            # 加载正样本
            audio_path = os.path.join(self.pos_audio_dir, sample['filename'])
            audio, _ = librosa.load(audio_path, sr=self.sr)
            
            # 截断或零填充至窗口大小
            if len(audio) > self.window_size:
                audio = audio[:self.window_size]
            elif len(audio) < self.window_size:
                # 若正样本音频长度不足window_size，在音频尾部添加零填充，直至长度等于窗口大小
                padding = np.zeros(self.window_size - len(audio))
                audio = np.concatenate([audio, padding]) 
        else:
            # 加载负样本切片
            audio_path = os.path.join(self.neg_audio_dir, sample['filename'])
            full_audio, _ = librosa.load(audio_path, sr=self.sr) 
            # librosa.load默认mono=True，librosa.load 会将多声道音频自动转换为单声道（通过对所有通道取平均）
                # 当 mono=False 时：full_audio 将是一个二维数组，形状为 [n_channels, n_samples]
            # full_audio的形状为：一个NumPy一维数组，形状为 [n_samples]，包含音频采样值
            # _: 实际使用的采样率（这里使用下划线表示忽略该值）    
            
            start_idx = sample['start_idx']
            end_idx = sample['end_idx']
            
            # 提取切片并处理边界情况
            if end_idx <= len(full_audio):
                audio = full_audio[start_idx:end_idx]
            else:
                # 若当前遍历得到的seg的end_idx超出full_audio长度（说明该片段为对应负样本音频的最后一段，但可能小于指定的窗口大小），则将full_audio的剩余部分作为当前音频片段
                audio = np.zeros(self.window_size) # 创建一个长度为window_size的零audio，准备将其最后一段音频片段填入
                segment_len = len(full_audio) - start_idx
                audio[:segment_len] = full_audio[start_idx:]
            
            
            # 确保长度一致
            # 将每段负样本seg也通过尾部零填充至窗口大小
            if len(audio) < self.window_size:
                # 在Python中，if/else语句不会创建新的变量作用域。即使audio变量创建于上一段代码的if中，在if外也可以安全地访问audio变量
                # 在Python中，变量作用域由函数、类、模块等定义，而不是由if/else、for/while等控制流语句定义
                padding = np.zeros(self.window_size - len(audio))
                audio = np.concatenate([audio, padding])
        
        # 原始的audio是一维NumPy数组，形状为[window_size]
        # torch.FloatTensor(audio)将其转换为PyTorch张量，保持形状
        # .unsqueeze(0)表示在第0维添加一个新的维度，变为形状[1, window_size]，表示单声道（模仿sample, _ = torchaudio.load(audio_path)输出的sample形状）
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0) # 形状[1, window_size]
        label = torch.tensor(sample['label'], dtype=torch.float)
        
        return audio_tensor, label, sample['filename']
    
    # +++++ example +++++
    # def __getitem__(self, idx):
    #     audio_path = self.data_list[idx]
    #     sample, _ = torchaudio.load(audio_path)
    #     if self.transform:
    #         sample = self.transform(sample)
    #     label = self.labels[idx]
    #     return sample, label

# 修改BCResNet模型，将多分类改为二分类
class WakeWordDetector(nn.Module):
    def __init__(self, model_version=1):
        super(WakeWordDetector, self).__init__()
        tau = model_version
        base_c = int(tau * 8)
        
        # 使用原始BCResNet架构但输出为1个类别
        self.bcresnet = BCResNets(base_c=base_c, num_classes=1)
    
    def forward(self, x):
        '''
        由于num_classes=1(二分类任务)，每个样本只有一个得分值
        .squeeze(-1)移除最后一个维度，将输出转换为形状[batch_size]的一维张量
        返回一个批量样本的唤醒词检测分数
        每个元素代表对应批次中音频样本的原始检测分数(logits)，尚未经过sigmoid转换，不是概率值
        
        继承自nn.Module的forward方法是PyTorch模型的前向传播函数
        当执行model(inputs)时，PyTorch会自动调用model.forward(inputs)
        在训练循环中计算前向传播
        '''
        return self.bcresnet(x).squeeze(-1) # 返回一个批量样本的唤醒词检测分数

# 评估函数
def evaluate(model, dataloader, device, criterion, preprocess, epoch, calculate_errors=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for inputs, labels, filenames in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 预处理音频
            inputs = preprocess(inputs, labels, augment=False, is_train=False)
            
            # 前向传播
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # 统计
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    correct = sum(np.array(all_preds) == np.array(all_labels))
    accuracy = correct / len(all_labels)
    
    # 计算精确率和召回率
    tp = sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
    fp = sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
    fn = sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # 记录错误样本
    errors = []
    if calculate_errors:
        for pred, label, filename in zip(all_preds, all_labels, all_filenames):
            if pred != label:
                errors.append({
                    "epoch": epoch,
                    "filename": filename,
                    "predicted": int(pred),
                    "true": int(label)
                })
    
    return {
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
    logger.info(f"Using device: {device}")
    
    # 加载数据集信息
    dataset_info_path = os.path.join(args.workspace, "dataset", "dataset_info.json")
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    # 设置窗口大小和步长（单位为采样点数，音频在generate_dataset步骤中被重采样为16000hz，故转换时乘以16000）
    # 窗口大小 = 正集的训练集的avg_duration
    # 窗口步长 = 正集的训练集的avg_duration * 步长比例
    pos_avg_duration = dataset_info["positive"]["train"]["avg_duration"]
    window_size = int(pos_avg_duration * 16000)  # 转为采样点数
    window_stride = int(window_size * args.window_stride_ratio) # 单位为采样点数
    
    logger.info(f"Positive average duration: {pos_avg_duration:.4f}s")
    logger.info(f"Window size: {window_size} samples ({window_size/16000:.4f}s)")
    logger.info(f"Window stride: {window_stride} samples ({window_stride/16000:.4f}s)")

    # 加载训练用数据集
    train_dataset = WakeWordDataset(args.workspace, "train", window_size, window_stride) # 加载train数据集
    dev_dataset = WakeWordDataset(args.workspace, "dev", window_size, window_stride) # 加载dev数据集
    
    train_loader = DataLoader(
        train_dataset, 
        # 要加载的数据集对象，必须满足：
        # 1. 继承torch.utils.data.Dataset类
        # 2. 实现两个必要方法:
        #     __len__(self): 返回数据集大小
        #     __getitem__(self, idx): 根据索引返回单个样本
        batch_size=args.batch_size, # 每个批次的样本数量
        shuffle=True, # 是否在每个epoch开始时打乱数据集（训练时通常设置为True，模型训练时正负样本交叉输入）
        num_workers=4, # 用于并行加载数据的子进程数量，提高数据加载效率
        pin_memory=args.use_gpu # 是否将数据放入CUDA固定内存，可加速GPU训练时的数据传输
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=args.use_gpu
    )
    
    # check dataset size
    logger.info(f"======== Train dataset count(After delploy sliding window on neg dataset) ========")
    logger.info(f"Train samples count: {len(train_dataset)}")
    logger.info(f"Dev samples count: {len(dev_dataset)}")
    
    # 初始化模型
    model = WakeWordDetector(model_version=args.model_version).to(device)
    logger.info(f"Initialized bcresnet with version {args.model_version}")
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
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
    noise_dir = None
    noise_paths = [
        os.path.join(args.workspace, "dataset", "noise"),
        os.path.join(args.workspace, "dataset", "negative", "audio", "_background_noise_")
    ]
    for path in noise_paths:
        if os.path.exists(path):
            noise_dir = path
            logger.info(f"Found noise directory: {noise_dir}")
            break
        
    frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}
    
    preprocess = Preprocess(
        noise_dir = "/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/bcresnet/data/speech_commands_v0.02/_background_noise_", # 这里的输入的noise_dir结构应形如speech_commands_v0.02/_background_noise_一样，其中包含多个.wav噪声文件   
        device = device,
        specaug = (args.model_version >= 1.5),
        frequency_masking_para = frequency_masking_para[args.model_version],
    )
    # 数据增强使用样例：
    # inputs = self.preprocess_train(inputs, labels, augment=True) # 预处理训练数据，进行数据增强，包括1. 噪声增强 2. 时间偏移 3. 频谱增强（具体是否执行，依据选择的BC-Resnet版本）
    # outputs = self.model(inputs) # 得到模型输出，这里的输出是一个 batch 的预测结果，形状是[batch_size, num_classes]
    
    # 训练准备
    best_metrics = {"f1": 0, "accuracy": 0, "epoch": 0}
    all_dev_errors = []
    iterations = 0
    
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
                    lr = args.init_lr * iterations / n_step_warmup
                else:
                    lr = args.lr_lower_limit + 0.5 * (args.init_lr - args.lr_lower_limit) * (
                        1 + np.cos(np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup))
                    )
                
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr # 更新当前学习率（给优化器看）
                
                # 处理数据
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 预处理和数据增强
                inputs = preprocess(inputs, labels, augment=True)
                
                # 前向传播
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct = (preds == labels).sum().item()
                epoch_correct += correct
                epoch_total += len(labels)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{args.total_epochs}",
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct/len(labels):.4f}",
                    "lr": f"{lr:.6f}"
                })
            
            # 计算训练指标
            train_loss = epoch_loss / len(train_loader)
            train_acc = epoch_correct / epoch_total
            
            # 记录日志
            logger.info(f"Epoch {epoch+1}/{args.total_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # 在开发集上评估
            if (epoch + 1) % args.eval_on_dev_epoch_stride == 0 or epoch == args.total_epochs - 1:
                dev_metrics, dev_errors = evaluate(
                    model, dev_loader, device, criterion, preprocess, epoch+1, calculate_errors=True
                )
                
                # 记录错误
                all_dev_errors.extend(dev_errors)
                
                # 记录日志
                logger.info(f"Dev - Loss: {dev_metrics['loss']:.4f}, Acc: {dev_metrics['accuracy']:.4f}, " + 
                           f"Prec: {dev_metrics['precision']:.4f}, Rec: {dev_metrics['recall']:.4f}, F1: {dev_metrics['f1']:.4f}")
                
                # 保存当前模型
                current_model_path = os.path.join(model_dir, f"model_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'window_size': window_size,
                    'window_stride': window_stride,
                    'metrics': dev_metrics
                }, current_model_path)
                
                # 更新最佳模型
                if dev_metrics['f1'] > best_metrics['f1']:
                    best_metrics = {**dev_metrics, 'epoch': epoch + 1}
                    best_model_path = os.path.join(model_dir, "model_best.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'window_size': window_size,
                        'window_stride': window_stride,
                        'metrics': dev_metrics
                    }, best_model_path)
                    logger.info(f"New best model saved! F1: {best_metrics['f1']:.4f}")
    
    # 保存开发集错误记录
    dev_errors_path = os.path.join(train_dir, "dev_errors.json")
    with open(dev_errors_path, 'w') as f:
        json.dump(all_dev_errors, f, indent=2)
    
    # 训练完成日志
    logger.info("Training completed!")
    logger.info(f"Best model (epoch {best_metrics['epoch']}):")
    logger.info(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_metrics['recall']:.4f}")
    logger.info(f"  F1: {best_metrics['f1']:.4f}")
    logger.info(f"Development errors saved to {dev_errors_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train wake word detection model")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    parser.add_argument("--model_version", type=float, default=1.0, choices=[1, 1.5, 2, 3, 6, 8], help="BC-ResNet model version (tau value), chooce betweeen 1, 1.5, 2, 3, 6, 8")
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