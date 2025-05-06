import os
import json
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

# 自定义唤醒词数据集类
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
    def __init__(self, workspace, split, window_size, window_stride, sample_rate):
        self.workspace = workspace
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.sr = sample_rate
        
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