import torch
import torch.nn as nn
from .bcresnet import BCResNets

class WakeWordDetector(nn.Module):
    def __init__(self, model_version=1, spec_group_num=5):
        super(WakeWordDetector, self).__init__()
        tau = model_version
        spec_groups = spec_group_num
        base_c = int(tau * 8)
        
        # 使用原始BCResNet架构但输出为1个类别
        self.bcresnet = BCResNets(base_c=base_c, spec_groups=spec_groups, num_classes=1)
    
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