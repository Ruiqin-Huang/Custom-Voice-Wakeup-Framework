#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# generate_noise_dataset.py
# This script processes noise audio files from a source directory, resamples them to 16kHz WAV format,
# and saves them in the /dataset/noise/audio directory.
# - workspace
#     - dataset
#         - noise
#             - audio/
# The script logs all processing steps and errors.

import os
import logging
import sys
import librosa
import soundfile as sf
from tqdm import tqdm

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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def prepare_directories(workspace):
    """创建必要的目录结构"""
    noise_dir = os.path.join(workspace, "dataset", "noise")
    audio_dir = os.path.join(noise_dir, "audio")
    
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    return audio_dir

def get_audio_files(source_dir):
    """获取源目录中的所有音频文件"""
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a']
    audio_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        logging.warning(f"[WARNING] 在 {source_dir} 中未找到任何音频文件")
    
    return audio_files

def process_audio_file(source_path, output_dir, verbose=False):
    """处理单个音频文件：重采样为16kHz并保存为WAV格式"""
    try:
        # 获取文件名并创建输出路径
        filename = os.path.splitext(os.path.basename(source_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.wav")
        
        # 如果文件已存在则跳过
        if os.path.exists(output_path):
            return True, 0  # 返回成功标志和持续时间0（表示跳过）
        
        # 加载并重采样音频
        y, sr = librosa.load(source_path, sr=None)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        
        # 保存为16kHz WAV文件
        sf.write(output_path, y_resampled, 16000)
        
        # 计算持续时间
        duration = librosa.get_duration(y=y_resampled, sr=16000)
        return True, duration
        
    except Exception as e:
        if verbose:
            logging.error(f"[ERROR] 处理 {source_path} 时出错: {str(e)}")
        return False, 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process noise audio files to 16kHz WAV format")
    parser.add_argument("--noise_source_dir", type=str, required=True, 
                       help="Directory containing source noise audio files")
    parser.add_argument("--workspace", type=str, required=True, 
                       help="Workspace directory where processed files will be saved")
    args = parser.parse_args()
    
    # 设置日志记录器
    logger = setup_logger(args.workspace)
    
    logging.info("======== 噪声数据集生成 ========")
    logging.info(f"[INFO] 源目录: {args.noise_source_dir}")
    logging.info(f"[INFO] 工作区目录: {args.workspace}")
    
    try:
        # 准备目录
        audio_dir = prepare_directories(args.workspace)
        
        # 获取源目录中的所有音频文件
        audio_files = get_audio_files(args.noise_source_dir)
        
        if not audio_files:
            logging.error("[ERROR] 未找到要处理的音频文件")
            return
        
        logging.info(f"[INFO] 找到 {len(audio_files)} 个要处理的音频文件")
        
        # 处理每个文件
        success_count = 0
        error_count = 0
        skipped_count = 0
        total_duration = 0
        
        # 使用tqdm显示进度条
        for audio_file in tqdm(audio_files, desc="处理噪声文件"):
            success, duration = process_audio_file(audio_file, audio_dir)
            if success:
                success_count += 1
                total_duration += duration
                if duration == 0:  # 文件已存在被跳过
                    skipped_count += 1
            else:
                error_count += 1
        
        # 只在结束时打印汇总信息
        logging.info(f"[INFO] 噪声处理统计:")
        logging.info(f"[INFO] - 总文件数: {len(audio_files)}")
        logging.info(f"[INFO] - 成功处理: {success_count - skipped_count} 个文件")
        logging.info(f"[INFO] - 跳过已存在: {skipped_count} 个文件")
        logging.info(f"[INFO] - 处理失败: {error_count} 个文件")
        logging.info(f"[INFO] - 总音频时长: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
        logging.info(f"[INFO] 噪声数据集生成完成")
        logging.info("====================================")
        
    except Exception as e:
        logging.error(f"[ERROR] 错误: {e}")
        raise

if __name__ == "__main__":
    main()