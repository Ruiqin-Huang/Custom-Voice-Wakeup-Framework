#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# generate_dataset_info.py
# This script scans the dataset directory and generates a dataset_info.json file
# containing statistics about the positive and negative datasets.

import argparse
import os
import json
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
    
    # 创建文件处理器（以append模式）
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
    parser = argparse.ArgumentParser(description="Generate dataset info JSON file")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    return parser.parse_args()

def read_jsonl_file(file_path):
    """读取JSONL文件并返回解析后的数据列表"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    except FileNotFoundError:
        logging.error(f"[ERROR] 文件不存在: {file_path}")
        return []
    except Exception as e:
        logging.error(f"[ERROR] 读取文件 {file_path} 时出错: {e}")
        return []

def calculate_stats(data):
    """计算数据集的统计信息"""
    if not data:
        return {
            "total_duration": 0,
            "total_files": 0,
            "avg_duration": 0,
            "min_duration": 0,
            "max_duration": 0,
            "percentile_80": 0,
            "percentile_90": 0
        }
    
    durations = [item["duration"] for item in data]
    total_duration = sum(durations)
    total_files = len(data)
    avg_duration = total_duration / total_files if total_files > 0 else 0
    min_duration = min(durations) if total_files > 0 else 0
    max_duration = max(durations) if total_files > 0 else 0
    
    # 计算80%和90%百分位数
    sorted_durations = sorted(durations)
    index_80 = int(len(sorted_durations) * 0.8)
    index_90 = int(len(sorted_durations) * 0.9)
    percentile_80 = sorted_durations[index_80] if total_files > 0 else 0
    percentile_90 = sorted_durations[index_90] if total_files > 0 else 0
    
    return {
        "total_duration": round(total_duration, 2),
        "total_files": total_files,
        "avg_duration": round(avg_duration, 2),
        "min_duration": round(min_duration, 2),
        "max_duration": round(max_duration, 2),
        "percentile_80": round(percentile_80, 2),
        "percentile_90": round(percentile_90, 2)
    }

def main():
    args = parse_args()
    
    # 设置日志记录器
    setup_logger(args.workspace)
    
    logging.info(f"======== DATASET INFO GENERATION ========")
    logging.info(f"[INFO] 开始生成数据集统计信息...")
    
    dataset_dir = os.path.join(args.workspace, "dataset")
    positive_dir = os.path.join(dataset_dir, "positive")
    negative_dir = os.path.join(dataset_dir, "negative")
    
    if not os.path.exists(dataset_dir):
        logging.error(f"[ERROR] 数据集目录不存在: {dataset_dir}")
        return
    
    if not os.path.exists(positive_dir):
        logging.warning(f"[WARNING] 正样本目录不存在: {positive_dir}")
    
    if not os.path.exists(negative_dir):
        logging.warning(f"[WARNING] 负样本目录不存在: {negative_dir}")
    
    # 读取所有JSONL文件
    logging.info(f"[INFO] 读取jsonl文件...")
    pos_train_data = read_jsonl_file(os.path.join(positive_dir, "pos_train.jsonl"))
    pos_dev_data = read_jsonl_file(os.path.join(positive_dir, "pos_dev.jsonl"))
    pos_test_data = read_jsonl_file(os.path.join(positive_dir, "pos_test.jsonl"))
    neg_train_data = read_jsonl_file(os.path.join(negative_dir, "neg_train.jsonl"))
    neg_dev_data = read_jsonl_file(os.path.join(negative_dir, "neg_dev.jsonl"))
    neg_test_data = read_jsonl_file(os.path.join(negative_dir, "neg_test.jsonl"))
    
    # 计算正集统计信息
    logging.info(f"[INFO] 计算正样本统计信息...")
    pos_train_stats = calculate_stats(pos_train_data)
    pos_dev_stats = calculate_stats(pos_dev_data)
    pos_test_stats = calculate_stats(pos_test_data)
    
    # 计算负集统计信息
    logging.info(f"[INFO] 计算负样本统计信息...")
    neg_train_stats = calculate_stats(neg_train_data)
    neg_dev_stats = calculate_stats(neg_dev_data)
    neg_test_stats = calculate_stats(neg_test_data)
    
    # 计算正集总体统计信息
    pos_all_data = pos_train_data + pos_dev_data + pos_test_data
    pos_all_stats = calculate_stats(pos_all_data)
    
    # 计算负集总体统计信息
    neg_all_data = neg_train_data + neg_dev_data + neg_test_data
    neg_all_stats = calculate_stats(neg_all_data)
    
    # 创建dataset_info字典
    dataset_info = {
        "positive": {
            "all": {
                "total_duration": pos_all_stats["total_duration"],
                "total_files": pos_all_stats["total_files"]
            },
            "train": {
                "total_duration": pos_train_stats["total_duration"],
                "total_files": pos_train_stats["total_files"],
                "avg_duration": pos_train_stats["avg_duration"],
                "min_duration": pos_train_stats["min_duration"],
                "max_duration": pos_train_stats["max_duration"],
                "percentile_80_duration": pos_train_stats["percentile_80"],
                "percentile_90_duration": pos_train_stats["percentile_90"]
            },
            "dev": {
                "total_duration": pos_dev_stats["total_duration"],
                "total_files": pos_dev_stats["total_files"]
            },
            "test": {
                "total_duration": pos_test_stats["total_duration"],
                "total_files": pos_test_stats["total_files"]
            }
        },
        "negative": {
            "all": {
                "total_duration": neg_all_stats["total_duration"],
                "total_files": neg_all_stats["total_files"]
            },
            "train": {
                "total_duration": neg_train_stats["total_duration"],
                "total_files": neg_train_stats["total_files"]
            },
            "dev": {
                "total_duration": neg_dev_stats["total_duration"],
                "total_files": neg_dev_stats["total_files"]
            },
            "test": {
                "total_duration": neg_test_stats["total_duration"],
                "total_files": neg_test_stats["total_files"]
            }
        },
        "dataset_ratio": {
            "positive_negative_ratio": round(pos_all_stats["total_duration"] / neg_all_stats["total_duration"], 3) if neg_all_stats["total_duration"] > 0 else "N/A"
        }
    }
    
    # 保存dataset_info.json文件
    output_path = os.path.join(dataset_dir, "dataset_info.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        logging.info(f"[INFO] 数据集统计信息已保存至 {output_path}")
    except Exception as e:
        logging.error(f"[ERROR] 保存JSON文件时出错: {e}")
    
    logging.info(f"[INFO] 正样本总时长: {pos_all_stats['total_duration']}秒，共{pos_all_stats['total_files']}个文件")
    logging.info(f"[INFO] 负样本总时长: {neg_all_stats['total_duration']}秒，共{neg_all_stats['total_files']}个文件")
    logging.info(f"[INFO] 正样本训练集总时长: {pos_train_stats['total_duration']}秒，平均时长: {pos_train_stats['avg_duration']}秒，最小时长: {pos_train_stats['min_duration']}秒，最大时长: {pos_train_stats['max_duration']}秒，80%时长: {pos_train_stats['percentile_80']}秒，90%时长: {pos_train_stats['percentile_90']}秒")
    logging.info(f"[INFO] 正负样本比例（滑动窗口处理前）: {dataset_info['dataset_ratio']['positive_negative_ratio']}")
    logging.info(f"[INFO] 数据集统计信息生成完成！")
    logging.info(f"========================================")

if __name__ == "__main__":
    main()