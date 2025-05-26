import torch
import torchaudio.transforms as T

class SpecAugmentation(torch.nn.Module):
    """
    对输入的LogMel频谱图应用SpecAugment。
    SpecAugment包括频率掩码和时间掩码。
    """
    def __init__(self, freq_mask_param, time_mask_param, num_freq_masks=1, num_time_masks=1):
        """
        初始化SpecAugmentation。

        参数:
            freq_mask_param (int): 频率掩码的最大宽度 (F)。
            time_mask_param (int): 时间掩码的最大宽度 (T)。
            num_freq_masks (int): 应用的频率掩码数量 (m_F)。
            num_time_masks (int): 应用的时间掩码数量 (m_T)。
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

        # 创建频率掩码和时间掩码的变换列表
        transforms = []
        for _ in range(self.num_freq_masks):
            transforms.append(T.FrequencyMasking(freq_mask_param=self.freq_mask_param))
        for _ in range(self.num_time_masks):
            transforms.append(T.TimeMasking(time_mask_param=self.time_mask_param))
        
        self.transform = torch.nn.Sequential(*transforms)

    def forward(self, specgram):
        """
        对输入的频谱图应用SpecAugment。

        参数:
            specgram (Tensor): 输入的频谱图，形状应为 (..., num_freq_bins, num_time_frames)。
                               例如 (batch_size, num_channels, num_freq_bins, num_time_frames)
                               或 (num_channels, num_freq_bins, num_time_frames)
                               或 (num_freq_bins, num_time_frames)。

        返回:
            Tensor: 经过SpecAugment处理的频谱图。
        """
        return self.transform(specgram)

if __name__ == '__main__':
    # 示例用法
    # 假设我们有一个LogMel频谱图
    # batch_size=4, num_channels=1, n_mels=40, n_frames=101
    example_specgram = torch.randn(4, 1, 40, 101)
    
    # 初始化SpecAugmentation
    # F=8 (最大8个mel bin被遮盖), T=20 (最大20个时间帧被遮盖)
    # m_F=1 (1个频率遮盖), m_T=1 (1个时间遮盖)
    spec_augmenter = SpecAugmentation(freq_mask_param=8, time_mask_param=20, num_freq_masks=1, num_time_masks=1)
    
    # 应用增强
    augmented_specgram = spec_augmenter(example_specgram)
    
    print("Original specgram shape:", example_specgram.shape)
    print("Augmented specgram shape:", augmented_specgram.shape)

    # 检查是否真的应用了掩码 (可以通过打印或可视化来验证)
    # 例如，比较原始和增强后的频谱图的某个样本
    # print("Original sample 0, channel 0:\n", example_specgram[0, 0])
    # print("Augmented sample 0, channel 0:\n", augmented_specgram[0, 0])
    
    # 确保掩码值是0 (torchaudio.transforms.TimeMasking 和 FrequencyMasking 默认将掩码区域的值设为0)
    # 我们可以检查增强后的频谱图中是否存在0值，以及这些0值是否形成条带状
    
    # 验证多个掩码
    spec_augmenter_multi_mask = SpecAugmentation(freq_mask_param=5, time_mask_param=10, num_freq_masks=2, num_time_masks=2)
    augmented_specgram_multi = spec_augmenter_multi_mask(example_specgram)
    print("Augmented specgram (multi-mask) shape:", augmented_specgram_multi.shape)