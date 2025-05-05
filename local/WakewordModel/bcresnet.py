# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch.nn.functional as F
from torch import nn

# 由于BC-ResNet模型本身也是被from WakewordModel.bcresnet import BCResNets导入的
# 工作目录位于执行from WakewordModel.bcresnet import BCResNets的脚本所在目录，如local/
# 故这里需要使用相对路径导入
from .subspectralnorm import SubSpectralNorm


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane, # 输入通道数
        out_plane, # 输出通道数
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
        spec_groups=5, # 频率维度(n_mels)必须能被spec_groups整除
    ):
        super().__init__()

        def get_padding(kernel_size, use_dilation):
            rate = 1  # dilation rate
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx

        # padding and dilation rate
        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        # convbnrelu block
        layers = []
        layers.append(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)
        )
        if ssn:
            layers.append(SubSpectralNorm(out_plane, spec_groups)) # num_features=out_plane, spec_groups=5，输入的频率维度(n_mels)必须能被spec_groups整除
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride, spec_groups=5):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        # 2D part (f2)
        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1, spec_groups=spec_groups))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
                spec_groups=spec_groups,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        # 1D part (f1)
        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        # 2D part
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        # 1D part
        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride, spec_groups=5):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride, spec_groups=spec_groups))
    return stage


class BCResNets(nn.Module):
    '''
    基于 BC-ResNet 的 KWS 模型
    该模型是一个卷积神经网络（CNN），用于语音命令识别任务。它的设计灵感来自于 ResNet 架构，结合了深度可分离卷积和残差连接的优点。
    
    接口说明：
        - 输入形式：BCResNet 接受的输入形式是经过预处理的音频特征图：[batch_size, 1, n_mels, time_frames]
            - 其中 n_mels=40（频率维度），要求n_mels必须①能被spec_groups(子频带数量)整除（在SubSpectralNorm中使用）②且能被2³=8整除（在网络结构中有3次频率维度下采样）
            - time_frames≈100（时间维度，取决于hop_length=160的设置），可小幅度灵活调整
        
        - 输出形式：BCResNet 的输出是一个大小为 [batch_size, num_classes] 的张量，表示每个类别的预测概率。
            每个样本对应12个数值，表示属于各个类别的预测得分（logits）。
    
    网络结构：
        def forward(self, x):
            x = self.cnn_head(x)  # 初始卷积处理
            for i, num_modules in enumerate(self.n):
                for j in range(num_modules):
                    x = self.BCBlocks[i][j](x)  # 通过多个BC-ResBlock
            x = self.classifier(x)  # 分类器处理
            x = x.view(-1, x.shape[1])  # 调整形状为[batch_size, num_classes]
            return x
        
        bcresnet.py 中的 forward 方法实现了模型的前向传播过程，主要分为以下几个步骤：
            1. cnn_head：对输入音频特征图进行初始卷积处理，提取基本特征。
            2. BCBlocks：通过多个 BC-ResBlock 进行特征提取和变换。
            3. classifier：使用卷积和池化操作进行分类，输出每个类别的预测概率。
            4. 调整输出形状为 [batch_size, num_classes]，以便进行分类任务。
            
    输入形状放缩：
    1. 频率维度(n_mels)存在限制，不建议改变n_mels，原因如下：
        - 架构限制：BCResNet网络对n_mels有结构性要求。网络中有3次频率维度下采样（cnn_head一次，BCBlocks中两次），每次下采样都将频率维度减半，故要求n_mels必须能被2³=8整除
        - 训练-推理一致性：模型是在特定梅尔滤波器配置下学习的特征表示，改变n_mels（如从40到50）会导致频率特征分布变化，这会严重影响模型性能，因为特征不再匹配训练时学到的模式
    2. 时间维度(time_frames)具有一定灵活性，允许小范围修改，原因如下：
        - 自适应设计：网络使用nn.AdaptiveAvgPool2d((1, 1))自适应池化层，时间维度在前向传播过程中不会被强制调整尺寸
        - 实际应用限制：尽管技术上可以改变，但过大的变化可能导致模型表现下降，时间上下文信息会发生变化，影响模型对时序模式的理解
          
  
    '''
    def __init__(self, base_c, num_classes=12, spec_groups=5):
        super().__init__()
        self.num_classes = num_classes
        self.spec_groups = spec_groups  # 子频带数量
        self.n = [2, 2, 4, 4]  # identical modules repeated n times
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]  # num channels
        self.s = [1, 2]  # stage using stride
        self._build_network()

    def _build_network(self):
        # Head: (Conv-BN-ReLU)
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        # Body: BC-ResBlocks
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride, 
                                             spec_groups=self.spec_groups))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )

    def forward(self, x):
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x)
        x = x.view(-1, x.shape[1])
        return x
