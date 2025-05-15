#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
deploy_model.py
模型部署和推理脚本 - 基于滑动窗口和非极大值抑制的唤醒词检测系统

主要功能说明：
1. 模型加载与导出：
   - 加载训练好的模型权重和配置
   - 支持导出为ONNX格式 (可选)
2. 实时音频处理：
   - 使用环形缓冲区存储最近音频
   - 基于滑动窗口进行检测
3. 唤醒词检测：
   - 非极大值抑制(NMS)过滤重复检测
   - 基于阈值判断唤醒词是否存在
4. 示例应用：
   - 支持从麦克风实时检测唤醒词
   - 支持从音频文件检测唤醒词
   
部署后的模型主要有三种使用方式：
1. 导出模型为ONNX格式（便于跨平台部署）
    python deploy_model.py export --workspace /path/to/workspace --output model.onnx

2. 处理音频文件（离线检测）
    python deploy_model.py process --model /path/to/model_best.pt --audio /path/to/audio.wav --visualize
    添加--visualize参数可以生成检测结果的可视化图表，直观展示检测分数曲线和识别位置。此功能适用于模型评估和调试。

3. 实时检测（在线应用）
    python deploy_model.py realtime --model /path/to/model_best.pt --threshold 0.75
    通过麦克风接收音频输入，并实时检测唤醒词。当检测到唤醒词时，会触发回调函数，可以在此处添加自定义行为，如启动语音助手或执行特定命令。
"""

import os
import json
import time
import argparse
import logging
import sys
import collections
import numpy as np
import torch
import torch.nn as nn
from threading import Thread, Event
import queue
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入自定义模块
from WakewordModel.detector import WakeWordDetector
from AudioProcessor.logmel import LogMelFeatureExtractor

def setup_logger(workspace):
    """配置日志记录器，同时输出到控制台和文件"""
    log_dir = os.path.join(workspace, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "deploy_log.log")
    
    logger = logging.getLogger("deploy_logger")
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
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 修改初始化日志记录器的调用
logger = setup_logger(workspace="路径到你的工作区")  # 替换为实际工作区路径

class WakeWordDetectorDeployer:
    def __init__(self, model_path, threshold=None, nms_window=None, nms_threshold=0.6):
        """
        初始化唤醒词检测器部署类
        
        参数:
            model_path: 模型路径
            threshold: 检测阈值，若为None则使用模型中保存的最佳阈值
            nms_window: 非极大值抑制窗口大小(单位:帧数)，若为None则使用默认值
            nms_threshold: 非极大值抑制阈值
        """
        self.device = "cpu" # 考虑兼容性及实时检测性能，默认使用CPU进行推理
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        logger.info(f"从 {model_path} 加载模型")
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # 提取模型参数
        self.model_version = self.checkpoint['model_version']
        self.spec_group_num = self.checkpoint['spec_group_num']
        self.window_size = self.checkpoint['window_size']
        self.window_stride = self.checkpoint['window_stride']
        
        # 如果训练时保存了最佳阈值，则使用该阈值，否则使用默认阈值0.5
        metrics = self.checkpoint.get('metrics', {})
        self.best_threshold = threshold if threshold is not None else metrics.get('best_threshold', 0.5)
        
        # 初始化模型并加载权重
        self.model = WakeWordDetector(model_version=self.model_version, 
                                      spec_group_num=self.spec_group_num).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # 初始化LogMel特征提取器
        self.sample_rate = 16000  # 固定采样率
        self.log_mel_extractor = LogMelFeatureExtractor(
            device=self.device,
            sample_rate=self.sample_rate, 
            win_length=480, 
            hop_length=160, 
            n_fft=512, 
            n_mels=40
        )
        
        # 非极大值抑制，用于过滤连续多帧中的冗余检测
        # 初始化NMS参数
        # 窗口帧数默认设置为window_size的1.5倍，NMS的时间窗口大小，单位是"帧数"（每帧对应一个步长的音频）
        # 决定了在多大范围内寻找局部最大值，太小会导致连续触发，太大可能漏过有效检测
        self.nms_window = nms_window if nms_window is not None else int(1.5 * self.window_size / self.window_stride)
        # NMS的阈值，只有超过此阈值的局部最大值才被保留，进一步过滤检测结果，确保只有高置信度的检测被保留
        self.nms_threshold = nms_threshold
        
        # 打印配置信息
        logger.info(f"模型版本: {self.model_version}")
        logger.info(f"频谱组数量: {self.spec_group_num}")
        logger.info(f"窗口大小: {self.window_size} 采样点 ({self.window_size/self.sample_rate:.4f}秒)")
        logger.info(f"窗口步长: {self.window_stride} 采样点 ({self.window_stride/self.sample_rate:.4f}秒)")
        logger.info(f"检测阈值: {self.best_threshold}")
        logger.info(f"NMS窗口大小: {self.nms_window} 帧")
        logger.info(f"NMS阈值: {self.nms_threshold}")
        
        # 初始化环形缓冲区，用于存储最新的音频数据
        self.audio_buffer = collections.deque(maxlen=self.window_size)
        self.scores_buffer = collections.deque(maxlen=self.nms_window) # 用于非极大值抑制
        
        # 初始化状态变量
        self.is_running = False
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # 检测冷却时间(秒)，避免连续多次触发
    
    def export_onnx(self, output_path):
        """导出模型为ONNX格式"""
        logger.info(f"导出ONNX模型到 {output_path}")
        
        # 创建示例输入，PyTorch使用它来"追踪"模型的计算图。当这个输入通过模型时，PyTorch会记录所有操作和中间结果，从而正确构建ONNX模型结构。
        # self.window_size // 160这个计算是因为特征提取器使用了hop_length=160，意味着每160个音频采样点生成一个时间帧。
        dummy_input = torch.randn(1, 1, 40, self.window_size // 160).to(self.device)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 导出模型
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info("ONNX模型导出成功")
            return True
        except Exception as e:
            logger.error(f"ONNX模型导出失败: {e}")
            return False
    
    def _non_max_suppression(self, score):
        """
        非极大值抑制，用于过滤连续多帧中的冗余检测
        
        参数:
            score: 当前帧的检测分数
            
        返回:
            是否保留当前检测结果
        """
        # 添加当前分数到缓冲区
        self.scores_buffer.append(score)
        
        # 如果缓冲区未满，不进行NMS
        if len(self.scores_buffer) < self.nms_window:
            # 仅当当前分数为最大值且大于阈值时保留
            return score == max(self.scores_buffer) and score > self.best_threshold
        
        # 获取中心位置索引
        center_idx = len(self.scores_buffer) // 2
        center_score = self.scores_buffer[center_idx]
        
        # 如果中心分数低于阈值，直接抑制
        if center_score < self.best_threshold:
            return False
        
        # 检查中心点是否是局部最大值
        for i in range(len(self.scores_buffer)):
            # 如果存在更高的分数，抑制当前检测
            if i != center_idx and self.scores_buffer[i] > center_score:
                return False
            
            # 如果存在相同分数，但位置在后面，抑制当前检测(保证只触发一次)
            if i > center_idx and abs(self.scores_buffer[i] - center_score) < 1e-5:
                return False
        
        # 最后检查是否超过NMS阈值
        return center_score > self.nms_threshold
    
    def process_audio_frame(self, audio_frame):
        """
        处理单帧音频数据
        
        参数:
            audio_frame: 音频数据，形状为(samples,)
            
        返回:
            检测结果(score, is_detected)
        """
        # 将新的音频数据添加到缓冲区
        self.audio_buffer.extend(audio_frame)
        
        # 如果缓冲区未满，返回None
        if len(self.audio_buffer) < self.window_size:
            return 0.0, False
        
        # 提取当前窗口的音频数据
        audio_window = np.array(list(self.audio_buffer))
        
        # 将音频数据转换为模型输入格式，[batch_size, channels, samples]
        audio_tensor = torch.tensor(audio_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 提取LogMel特征
        with torch.no_grad():
            logmel_features = self.log_mel_extractor(audio_tensor)
            
            # 模型推理
            outputs = self.model(logmel_features).squeeze()
            score = torch.sigmoid(outputs).item()
        
        # 非极大值抑制
        is_detected = self._non_max_suppression(score)
        
        # 检查检测冷却时间
        current_time = time.time()
        if is_detected and (current_time - self.last_detection_time) < self.detection_cooldown:
            is_detected = False  # 如果在冷却时间内，抑制检测
        
        # 如果检测到唤醒词，更新最后检测时间
        if is_detected:
            self.last_detection_time = current_time
        
        return score, is_detected

    def process_audio_file(self, audio_path, visualize=False):
        """
        处理音频文件
        
        参数:
            audio_path: 音频文件路径
            visualize: 是否可视化结果
            
        返回:
            检测结果列表[(timestamp, score, is_detected), ...]
        """
        # 重置缓冲区
        self.audio_buffer.clear()
        self.scores_buffer.clear()
        
        # 加载音频文件
        logger.info(f"加载音频文件: {audio_path}")
        try:
            # 使用librosa加载音频，自动归一化到[-1, 1]并处理重采样
            audio_data, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
            logger.info(f"音频长度: {len(audio_data)/self.sample_rate:.2f}秒")
        except Exception as e:
            logger.error(f"加载音频文件失败: {e}")
            return []
        
        # 处理音频
        results = []
        frame_start = 0
        
        logger.info("开始处理音频...")
        with tqdm(total=len(audio_data) // self.window_stride, desc="处理进度") as pbar:
            while frame_start + self.window_size <= len(audio_data):
                # 提取当前帧
                current_frame = audio_data[frame_start:frame_start + self.window_stride]
                
                # 处理当前帧
                score, is_detected = self.process_audio_frame(current_frame)
                
                if len(self.audio_buffer) == self.window_size:  # 确保缓冲区已满
                    timestamp = frame_start / self.sample_rate
                    results.append((timestamp, score, is_detected))
                    if is_detected:
                        logger.info(f"在 {timestamp:.2f}秒 检测到唤醒词 (分数: {score:.4f})")
                
                # 移动到下一帧
                frame_start += self.window_stride
                pbar.update(1)
        
        logger.info(f"音频处理完成，总共 {len(results)} 帧")
        
        # 可视化结果
        if visualize and results:
            timestamps = [r[0] for r in results]
            scores = [r[1] for r in results]
            detections = [r[2] for r in results]
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, scores, label='检测分数')
            plt.axhline(y=self.best_threshold, color='r', linestyle='--', label='检测阈值')
            
            # 标记检测点
            detection_times = [t for t, _, d in results if d]
            if detection_times:
                plt.scatter(detection_times, [1.05] * len(detection_times), 
                           color='green', marker='v', s=100, label='检测点')
            
            plt.xlabel('时间 (秒)')
            plt.ylabel('分数')
            plt.title('唤醒词检测结果')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # 保存和显示
            output_dir = os.path.dirname(audio_path)
            output_path = os.path.join(output_dir, f"detection_result_{os.path.basename(audio_path)}.png")
            plt.savefig(output_path)
            logger.info(f"可视化结果已保存到 {output_path}")
            plt.show()
        
        return results

    def start_realtime_detection(self, callback=None, block=False):
        """
        启动实时检测
        
        参数:
            callback: 检测到唤醒词时的回调函数，格式为callback(timestamp, score)
            block: 是否阻塞当前线程
        """
        if self.is_running:
            logger.warning("实时检测已经在运行")
            return False
        
        # 重置缓冲区
        self.audio_buffer.clear()
        self.scores_buffer.clear()
        
        # 创建停止事件
        self.stop_event = Event()
        self.is_running = True
        
        # 创建音频处理队列
        self.audio_queue = queue.Queue()
        
        # 音频回调函数
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"音频输入状态: {status}")
            
            # 将音频数据加入队列
            audio_data = indata[:, 0].copy()  # 只取第一个通道
            self.audio_queue.put(audio_data)
        
        # 创建音频输入流
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=self.window_stride
            )
            self.stream.start()
            logger.info("实时检测已启动")
        except Exception as e:
            logger.error(f"启动音频流失败: {e}")
            self.is_running = False
            return False
        
        # 处理线程函数
        def processing_thread():
            while not self.stop_event.is_set():
                try:
                    # 获取音频数据
                    audio_data = self.audio_queue.get(timeout=1)
                    
                    # 处理音频帧
                    score, is_detected = self.process_audio_frame(audio_data)
                    
                    # 如果检测到唤醒词且提供了回调函数
                    if is_detected and callback:
                        callback(time.time(), score)
                    
                except queue.Empty:
                    pass  # 超时，继续循环
                except Exception as e:
                    logger.error(f"处理音频时出错: {e}")
            
            logger.info("处理线程已停止")
        
        # 启动处理线程
        self.process_thread = Thread(target=processing_thread, daemon=True)
        self.process_thread.start()
        
        if block:
            try:
                # 阻塞主线程，直到用户中断
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("检测到用户中断")
                self.stop_realtime_detection()
        
        return True
    
    def stop_realtime_detection(self):
        """停止实时检测"""
        if not self.is_running:
            logger.warning("实时检测未在运行")
            return
        
        logger.info("正在停止实时检测...")
        self.stop_event.set()
        
        # 停止音频流
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # 等待处理线程结束
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
        
        self.is_running = False
        logger.info("实时检测已停止")


def parse_args():
    parser = argparse.ArgumentParser(description="唤醒词检测模型部署工具")
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 导出模型命令
    export_parser = subparsers.add_parser("export", help="导出模型")
    export_parser.add_argument("--workspace", type=str, required=True, help="工作目录")
    export_parser.add_argument("--onnx_output", type=str, help="输出ONNX文件路径")
    
    # 处理音频文件命令
    process_parser = subparsers.add_parser("process", help="处理音频文件")
    process_parser.add_argument("--model", type=str, required=True, help="模型路径")
    process_parser.add_argument("--audio", type=str, required=True, help="音频文件路径")
    process_parser.add_argument("--threshold", type=float, help="检测阈值")
    process_parser.add_argument("--visualize", action="store_true", help="可视化结果")
    
    # 实时检测命令
    realtime_parser = subparsers.add_parser("realtime", help="实时检测")
    realtime_parser.add_argument("--model", type=str, required=True, help="模型路径")
    realtime_parser.add_argument("--threshold", type=float, help="检测阈值")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 使用参数中的 workspace 初始化日志记录器
    logger = setup_logger(workspace=args.workspace)
    
    if args.command == "export":
        # 加载模型
        model_path = os.path.join(args.workspace, "train", "model", "model_best.pt")
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return 1
        
        # 设置输出路径
        output_path = args.onnx_output if args.onnx_output else os.path.join(args.workspace, "deploy", "model.onnx")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建部署器并导出模型
        deployer = WakeWordDetectorDeployer(model_path)
        deployer.export_onnx(output_path)
        
    elif args.command == "process":
        # 检查文件是否存在
        if not os.path.exists(args.model):
            logger.error(f"模型文件不存在: {args.model}")
            return 1
        if not os.path.exists(args.audio):
            logger.error(f"音频文件不存在: {args.audio}")
            return 1
        
        # 创建部署器并处理音频文件
        deployer = WakeWordDetectorDeployer(args.model, threshold=args.threshold)
        deployer.process_audio_file(args.audio, visualize=args.visualize)
        
    elif args.command == "realtime":
        # 检查文件是否存在
        if not os.path.exists(args.model):
            logger.error(f"模型文件不存在: {args.model}")
            return 1
        
        # 创建部署器并启动实时检测
        deployer = WakeWordDetectorDeployer(args.model, threshold=args.threshold)
        
        # 设置回调函数
        def detection_callback(timestamp, score):
            print(f"\n[检测到唤醒词] 时间: {time.strftime('%H:%M:%S')} 分数: {score:.4f}")
        
        # 启动实时检测
        deployer.start_realtime_detection(callback=detection_callback, block=True)
    
    else:
        logger.error("请指定有效的命令: export, process, realtime")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())