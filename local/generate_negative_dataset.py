#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# generate_negative_dataset.py
# This script generates a negative dataset by sampling random audio files from the Mozilla Common Voice dataset.
# It creates /workspace/dataset/negative as the negative dataset directory.
# - workspace
#     - dataset
#         - negative
#             - audio/
#             - neg_train.jsonl
#             - neg_dev.jsonl
#             - neg_test.jsonl
# The generated negative dataset will not contain the specified wake word.

import argparse
import os
import pandas as pd
import random
import json
import librosa
import soundfile as sf
from tqdm import tqdm
import logging
import sys

def setup_logger(workspace):
    """配置日志记录器，同时输出到控制台和文件"""
    log_dir = os.path.join(workspace, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "generate_dataset_log.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器（如果有）
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Generate negative dataset from Mozilla Common Voice dataset")
    parser.add_argument("--wakeword", type=str, required=True, help="Wake word to exclude")
    parser.add_argument("--mcv_dir", type=str, required=True, help="Mozilla Common Voice dataset directory")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    parser.add_argument("--negative_train_duration", type=float, required=True, help="Negative training set duration in seconds")
    parser.add_argument("--negative_dev_duration", type=float, required=True, help="Negative development set duration in seconds")
    parser.add_argument("--negative_test_duration", type=float, required=True, help="Negative test set duration in seconds")
    return parser.parse_args()

def prepare_directories(workspace):
    """创建必要的目录结构"""
    negative_dir = os.path.join(workspace, "dataset", "negative")
    audio_dir = os.path.join(negative_dir, "audio")
    
    os.makedirs(negative_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    return negative_dir, audio_dir

def load_mcv_data(mcv_dir, split):
    """加载MCV数据集CSV文件"""
    csv_path = os.path.join(mcv_dir, f"{split}.csv")
    try:
        return pd.read_csv(csv_path, sep="\t")
    except FileNotFoundError:
        logging.error(f"[ERROR] 找不到文件: {csv_path}")
        raise
    except Exception as e:
        logging.error(f"[ERROR] 加载CSV文件时出错: {e}")
        raise

def filter_samples_without_wakeword(df, wakeword):
    """过滤出不包含唤醒词的样本"""
    # 不区分大小写地过滤
    filtered_df = df[~df["sentence"].str.lower().str.contains(wakeword.lower())]
    
    if len(filtered_df) == 0:
        logging.warning(f"[WARNING] 过滤后没有不包含唤醒词'{wakeword}'的样本")
    else:
        logging.info(f"[INFO] 过滤后剩余 {len(filtered_df)} 个不包含唤醒词的样本（原始样本数: {len(df)}）")
    
    return filtered_df

def sample_negative_audios(df, target_duration, clips_dir):
    """采样音频文件直到达到目标持续时间"""
    sampled_files = []
    total_duration = 0
    
    # 随机打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"[INFO] 开始采样，目标时长: {target_duration}秒")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"采样直到达到 {target_duration}s"):
        audio_path = os.path.join(clips_dir, row["path"])
        
        if not os.path.exists(audio_path):
            logging.warning(f"[WARNING] 文件不存在: {audio_path}")
            continue
        
        try:
            # 加载音频并获取持续时间
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            sampled_files.append({
                "path": row["path"],
                "sentence": row["sentence"],
                "duration": duration
            })
            
            total_duration += duration
            
            # 如果已达到目标持续时间，停止采样
            if total_duration >= target_duration:
                break
                
        except Exception as e:
            logging.warning(f"[WARNING] 处理 {audio_path} 时出错: {e}")
    
    # 检查是否达到目标时长
    if total_duration < target_duration:
        logging.warning(f"[WARNING] 采样完所有不含唤醒词的音频后仍未达到目标时长 {target_duration}秒，实际采样时长为 {total_duration:.2f}秒")
    
    logging.info(f"[INFO] 已采样 {len(sampled_files)} 个文件，总时长 {total_duration:.2f}秒")
    return sampled_files

def process_audio_files(sampled_files, mcv_dir, audio_dir, wakeword):
    """处理音频文件：将源音频转换为16kHz采样率的WAV格式"""
    clips_dir = os.path.join(mcv_dir, "clips")
    processed_files = []
    
    logging.info(f"[INFO] 开始处理 {len(sampled_files)} 个音频文件")
    
    for file_info in tqdm(sampled_files, desc="处理音频文件"):
        audio_path = os.path.join(clips_dir, file_info["path"])
        
        # 获取文件名（不含扩展名）并创建WAV文件路径
        filename = os.path.splitext(file_info["path"])[0]
        wav_filename = f"{filename}.wav"
        wav_path = os.path.join(audio_dir, wav_filename)
        
        try:
            # 加载MP3并以16kHz采样率重新采样
            y, sr = librosa.load(audio_path, sr=None)
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
            
            # 保存为WAV文件
            sf.write(wav_path, y_resampled, 16000)
            
            # 负样本中不包含唤醒词
            contains_wakeword = 0
            
            processed_files.append({
                "filename": wav_filename,
                "text": file_info["sentence"],
                "contains_wakeword": contains_wakeword,
                "duration": file_info["duration"]
            })
            
        except Exception as e:
            logging.error(f"[ERROR] 将 {audio_path} 转换为WAV时出错: {e}")
    
    logging.info(f"[INFO] 成功处理了 {len(processed_files)} 个音频文件")
    return processed_files

def save_jsonl(data, output_path):
    """保存数据到JSONL文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        logging.info(f"[INFO] 已保存数据到 {output_path}")
    except Exception as e:
        logging.error(f"[ERROR] 保存JSONL文件出错: {e}")
        raise

def main():
    args = parse_args()
    
    # 设置日志记录器
    logger = setup_logger(args.workspace)
    
    logging.info(f"[INFO] 开始生成负样本数据集")
    logging.info(f"[INFO] 唤醒词: {args.wakeword}")
    logging.info(f"[INFO] MCV数据集目录: {args.mcv_dir}")
    logging.info(f"[INFO] 工作区目录: {args.workspace}")
    
    try:
        # 准备目录
        negative_dir, audio_dir = prepare_directories(args.workspace)
        clips_dir = os.path.join(args.mcv_dir, "clips")
        
        # 检查clips目录是否存在
        if not os.path.exists(clips_dir):
            logging.error(f"[ERROR] clips目录不存在: {clips_dir}")
            return
        
        # 处理训练集
        logging.info(f"[INFO] 处理训练集...")
        train_df = load_mcv_data(args.mcv_dir, "train")
        filtered_train_df = filter_samples_without_wakeword(train_df, args.wakeword)
        train_samples = sample_negative_audios(filtered_train_df, args.negative_train_duration, clips_dir)
        train_processed = process_audio_files(train_samples, args.mcv_dir, audio_dir, args.wakeword)
        save_jsonl(train_processed, os.path.join(negative_dir, "neg_train.jsonl"))
        
        # 处理验证集
        logging.info(f"[INFO] 处理验证集...")
        dev_df = load_mcv_data(args.mcv_dir, "dev")
        filtered_dev_df = filter_samples_without_wakeword(dev_df, args.wakeword)
        dev_samples = sample_negative_audios(filtered_dev_df, args.negative_dev_duration, clips_dir)
        dev_processed = process_audio_files(dev_samples, args.mcv_dir, audio_dir, args.wakeword)
        save_jsonl(dev_processed, os.path.join(negative_dir, "neg_dev.jsonl"))
        
        # 处理测试集
        logging.info(f"[INFO] 处理测试集...")
        test_df = load_mcv_data(args.mcv_dir, "test")
        filtered_test_df = filter_samples_without_wakeword(test_df, args.wakeword)
        test_samples = sample_negative_audios(filtered_test_df, args.negative_test_duration, clips_dir)
        test_processed = process_audio_files(test_samples, args.mcv_dir, audio_dir, args.wakeword)
        save_jsonl(test_processed, os.path.join(negative_dir, "neg_test.jsonl"))
        
        logging.info(f"[INFO] 负样本数据集生成完成！负样本数据集保存于目录: {negative_dir}")
        
    except Exception as e:
        logging.error(f"[ERROR] 生成负样本数据集时发生错误: {e}")
        raise

if __name__ == "__main__":
    main()