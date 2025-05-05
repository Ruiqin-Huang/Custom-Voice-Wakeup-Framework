import torchaudio

class LogMelFeatureExtractor:
    '''
    LogMel类用于将音频波形转换为梅尔频谱图，并对其进行对数变换。
    该类的是对torchaudio.transforms.MelSpectrogram的封装，
    使其更易于使用和集成到数据预处理管道中。
    '''
    def __init__(
        # 设置logMel特征提取器默认参数
        self,
        device="cpu", 
        sample_rate=16000, 
        win_length=480, 
        hop_length=160, 
        n_fft=512, 
        n_mels=40
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        self.device = device

    def __call__(self, x):
        '''
        输入：
        输入的x是一个批次的音频，形状为[batch_size,1(单声道),采样点数]。
        该方法将其转换为梅尔频谱图，并对其进行对数变换。
        输出：
        返回值是一个对数梅尔频谱图，形状为[batch_size, 1, n_mels, time_frames]
        n_mels是梅尔频谱图的频率维度，对应于梅尔滤波器的数量。
        time_frames是时间维度，取决于输入音频的长度和hop_length参数。
        '''
        assert len(x.shape) == 3 # 确定输入x的形状为[batch_size,1(单声道),采样点数]
        self.mel = self.mel.to(self.device)
        output = (self.mel(x) + 1e-6).log()
        return output