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
        self.noise_audio_dir = os.path.join(workspace, "dataset", "noise", "audio")
        
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
        除了将该个正样本音频文件直接作为一个正样本，应用数据增强，包括：随机时间拉伸（单独应用1次，最多±20％，＋pitch修正）
        时间偏移（单独应用1次，最多±10％），噪声混合（随机取噪声目录中1段噪声音频.wav，选择正样本中的一个随机片段进行叠加混合（不是直接替换，是叠加混合），随机取两个噪音文件分别应用一次噪声增强）
        SpecAugment（1次），这样相当于原一条正样本被扩增为1+1+1+2+1=6条正样本
        
        对于所有负样本，应用滑动窗口技术，将长音频切分为多个小片段作为多个负样本
        """
        self.samples = []
        
        # 处理正样本 - 每个音频文件作为一个样本
        for sample in self.pos_samples:
            # 样本中的duration只记录原始音频的持续时间，不考虑增强后音频的持续时间
            # 1. 原始样本
            self.samples.append({
                'type': 'positive',
                'filename': sample['filename'],
                'text': sample['text'],
                'label': 1, # 1 means contains wakeword, 0 means not
                'duration': sample['duration'], # 单位：秒
                'augmentation': 'none'  # 标记为原始样本，不做增强
            })
            
            if self.split == 'train': # 仅对训练集正样本进行数据增强
                # 2. 随机时间拉伸增强 (±20%)
                self.samples.append({
                    'type': 'positive',
                    'filename': sample['filename'],
                    'text': sample['text'],
                    'label': 1,
                    'duration': sample['duration'],
                    'augmentation': 'time_stretch',
                })
                
                # 3. 时间偏移增强 (±10%)
                self.samples.append({
                    'type': 'positive',
                    'filename': sample['filename'],
                    'text': sample['text'],
                    'label': 1,
                    'duration': sample['duration'],
                    'augmentation': 'time_shift',
                })
                
                # 4-5. 噪声混合增强 (应用两次，每次随机选用不同噪声文件和SNR进行混合)
                self.samples.append({
                    'type': 'positive',
                    'filename': sample['filename'],
                    'text': sample['text'],
                    'label': 1,
                    'duration': sample['duration'],
                    'augmentation': 'noise_mix_0',
                })
                self.samples.append({
                    'type': 'positive',
                    'filename': sample['filename'],
                    'text': sample['text'],
                    'label': 1,
                    'duration': sample['duration'],
                    'augmentation': 'noise_mix_1',
                })
                
                # 6. SpecAugment增强
                self.samples.append({
                    'type': 'positive',
                    'filename': sample['filename'],
                    'text': sample['text'],
                    'label': 1,
                    'duration': sample['duration'],
                    'augmentation': 'specaugment'
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
    
            # 进行数据增强：随机拉伸+随机偏移+噪声混合（2次）+SpecAugment
            if sample['augmentation'] == 'time_stretch':
                # 随机时间拉伸 (±20%)
                stretch_factor = np.random.uniform(0.8, 1.2)  # ±20%范围内的随机因子
                audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
            elif sample['augmentation'] == 'time_shift':
                # 随机时间偏移 (±10%)
                shift_factor = np.random.uniform(-0.1, 0.1)  # ±10%范围内的随机偏移
                shift_samples = int(len(audio) * shift_factor)
                if shift_samples > 0:
                    # 音频内容向左移（丢弃音频开头）
                    audio = np.pad(audio, (0, shift_samples), 'constant')[shift_samples:]
                else:
                    # 音频内容向右移（丢弃音频结尾）
                    audio = np.pad(audio, (abs(shift_samples), 0), 'constant')[:-abs(shift_samples)]
            elif sample['augmentation'].startswith('noise_mix'):
                # 噪声混合增强
                if not hasattr(self, 'noise_files') or not self.noise_files: # 检查以下两个条件之一是否满足：当前对象(self)没有noise_files属性；当前对象有noise_files属性，但该属性值为空或等价于False的值
                    # 第一次调用时加载噪声文件列表
                    self.noise_files = [f for f in os.listdir(self.noise_audio_dir) if f.endswith('.wav')]
                
                if self.noise_files:
                    # 随机选择噪声文件
                    noise_file = np.random.choice(self.noise_files)
                    noise_path = os.path.join(self.noise_audio_dir, noise_file)
                    noise, _ = librosa.load(noise_path, sr=self.sr)
                    
                    # 如果噪声文件长度不足，则循环填充.(目的是确保整段音频都被噪声覆盖)(更接近真实场景，背景噪声通常持续存在)
                    if len(noise) < len(audio):
                        repeats = len(audio) // len(noise) + 1
                        noise = np.tile(noise, repeats)[:len(audio)]
                    
                    # 如果噪声文件太长，随机选择一段
                    elif len(noise) > len(audio):
                        start = np.random.randint(0, len(noise) - len(audio) + 1)
                        noise = noise[start:start + len(audio)]
                    
                    # 随机设置信噪比SNR (dB)
                    # 高SNR (如20dB)：信号明显强于噪声，听感清晰
                    # 低SNR (如5dB)：噪声影响较大，听感模糊但仍可辨别
                    # 0dB以下：噪声覆盖信号，几乎无法辨别原始内容
                        # 5dB：相当于嘈杂的公共场所，但语音仍可辨别
                        # 20dB：类似于安静环境中的低背景噪声
                    snr = np.random.uniform(5, 20)  # 5-20dB的信噪比范围
                    
                    # 计算信号和噪声功率
                    signal_power = np.mean(audio ** 2)
                    noise_power = np.mean(noise ** 2)
                    
                    # 根据SNR缩放噪声,确保了噪声被适当缩放，以达到期望的SNR水平
                    epsilon = 1e-10  # 一个很小的值，防止除零
                    signal_power = max(signal_power, epsilon)  # 确保信号功率不为零
                    noise_power = max(noise_power, epsilon)    # 确保噪声功率不为零
                    snr_factor = 10 ** (snr / 10)
                    scale = np.sqrt(signal_power / (noise_power * snr_factor))
                    if np.isnan(scale) or np.isinf(scale):
                        scale = 0.1  # 设置一个安全的默认值
                    scaled_noise = scale * noise
                    
                    # 叠加混合
                    audio = audio + scaled_noise
                    
                    # 归一化，防止溢出,防止混合后的音频幅值溢出，避免数字失真
                    max_abs_value = np.max(np.abs(audio))
                    if max_abs_value > 1.0:
                        audio = audio / max_abs_value

            # 在训练过程中提取logMel谱图后顺势增强，而不是在这里增强
            # elif sample['augmentation'] == 'specaugment':
                # SpecAugment增强
                # 移动至train_model.py过程中实现，仅对sample['augmentation'] == 'specaugment'的样本进行SpecAugment增强
    
            # 截断或零填充至窗口大小
            # 训练时随机化提高泛化能力，验证/测试时使用确定性方法保证一致性。
            if len(audio) > self.window_size:
                if self.split == 'train':  # 训练时做数据增强
                    # 随机选择起始点进行截断(但是会确保至少截断为window_size的长度)
                    start = np.random.randint(0, len(audio) - self.window_size + 1)
                    audio = audio[start:start + self.window_size]
                else:  # 验证/测试时居中截断
                    start = (len(audio) - self.window_size) // 2
                    audio = audio[start:start + self.window_size]
            elif len(audio) < self.window_size:
                if self.split == 'train':  # 训练时做数据增强
                    # 随机决定左右两侧的填充量
                    padding_left = np.random.randint(0, self.window_size - len(audio) + 1)
                    padding_right = self.window_size - len(audio) - padding_left
                    audio = np.pad(audio, (padding_left, padding_right), 'constant')
                else:  # 验证/测试时居中填充
                    padding_left = (self.window_size - len(audio)) // 2
                    padding_right = self.window_size - len(audio) - padding_left
                    audio = np.pad(audio, (padding_left, padding_right), 'constant') 
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
        
        return audio_tensor, label, sample['filename'], sample.get('augmentation', 'none')