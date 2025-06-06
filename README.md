# CUVOW: Custom Voice Wakeup Framework

**CUVOW** 是一个基于零样本语音克隆的自定义语音唤醒框架，提供了完整的端到端解决方案，用于训练、评估和部署个性化唤醒词检测模型。

## 1. 框架简介

CUVOW框架通过融合先进的KWS模型BC-ResNet和语音克隆模型CosyVoice 2，实现了一个完整、灵活、高效的唤醒词检测系统。框架特点包括：

- **零样本语音克隆**：无需大量目标唤醒词音频，仅用少量参考音频即可克隆出多样化的训练数据
- **端到端流程**：从数据准备到模型部署的完整解决方案
- **一键式操作**：通过简洁的脚本配置实现全流程自动化
- **多语言支持**：支持中文、英文、日语、韩语等多种语言的唤醒词
- **高性能推理**：轻量级模型设计，支持CPU实时推理

## 2. 安装指南

### 2.1 安装步骤

1. 克隆仓库并进入项目目录：

```bash
git clone https://github.com/username/CUVOW.git
cd CUVOW
```

2. 执行依赖安装脚本：

```bash
bash dependencies.sh
```

这将创建一个conda环境并安装所有必要的依赖项。

### 2.2 预训练模型获取

从从modelscope或github拉取CosyVoice 2：

1. https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B
2. https://github.com/Ruiqin-Huang/CosyVoice

下载CosyVoice 2预训练模型并放置在指定位置：

```bash
mkdir -p local/cosyvoice
# 下载CosyVoice2模型并解压到上述目录
```

## 3. 框架结构

### 3.1 目录结构

```
CUVOW/
├── data/                 # 数据存储目录
│   ├── Common_Voice_corpus_* # 负样本数据集
│   ├── MS-SNSD_noise_*   # 噪声数据集
│   └── clone_ref/        # 参考音频数据集
├── demo/                 # 示例工作目录
├── local/                # 核心功能模块
│   ├── AudioProcessor/   # 音频处理组件
│   ├── DataLoader/       # 数据加载组件
│   ├── WakewordModel/    # 唤醒词检测模型
│   ├── cosyvoice/        # 语音克隆组件
│   ├── clone_audio_with_cosyvoice.py    # 语音克隆脚本
│   ├── deploy_model.py                  # 模型部署脚本
│   ├── eval_model.py                    # 模型评估脚本
│   ├── generate_dataset_info.py         # 数据集信息生成脚本
│   ├── generate_negative_dataset.py     # 负样本生成脚本
│   ├── generate_noise_dataset.py        # 噪声数据集生成脚本
│   ├── generate_positive_dataset.py     # 正样本生成脚本
│   └── train_model.py                   # 模型训练脚本
├── dependencies.sh                       # 依赖安装脚本
├── train_custom_wakeword_detection_model.sh    # 自定义唤醒词训练脚本
└── train_and_eval_custom_wakeword_detection_model.sh  # 训练和评估自定义唤醒词检测模型脚本
```

### 3.2 核心模块

1. **语音克隆模块**：基于CosyVoice 2模型，实现零样本语音克隆
2. **数据加载模块**：处理正负样本数据集和噪声数据集
3. **音频预处理模块**：音频分帧和数据增强
4. **模型训练模块**：基于BC-ResNet架构的唤醒词检测模型训练
5. **模型评估模块**：多指标评估模型性能
6. **模型部署模块**：支持ONNX导出、离线检测和实时检测

## 4. 使用方法

### 4.1 自定义唤醒词训练

可以通过修改train_custom_wakeword_detection_model.sh脚本中的参数来自定义唤醒词检测模型：

```bash
# 修改唤醒词和相关参数
vim train_custom_wakeword_detection_model.sh

# 执行训练脚本
bash train_custom_wakeword_detection_model.sh
```

主要配置参数：
- `wakeword`: 自定义唤醒词文本（如"institute of technology"）
- `clone_ref_dir`: 参考音频目录
- `clone_num_samples`: 克隆音频数量
- `workspace`: 工作目录
- `run_stage`: 执行阶段（1:生成负样本, 2:生成正样本, 3:生成噪声数据, 4:生成数据集信息, 5:训练模型, 6:部署模型）

### 4.2 真实数据集训练

如果有准备了MCV格式的训练集和测试集，可以使用train_and_eval_custom_wakeword_detection_model.sh脚本训练和评估模型：

```bash
# 修改相关参数
vim train_and_eval_custom_wakeword_detection_model.sh

# 执行训练脚本
bash train_and_eval_custom_wakeword_detection_model.sh
```

### 4.3 模型部署

训练完成后，可以通过以下方式部署模型：

1. **导出ONNX模型**:
```bash
python local/deploy_model.py export --workspace ./demo/your_wakeword --onnx_output ./model.onnx
```

2. **离线文件检测**:
```bash
python local/deploy_model.py process --model ./demo/your_wakeword/train/model/model_best.pt --audio ./test.wav --visualize
```

3. **实时检测**:
```bash
python local/deploy_model.py realtime --model ./demo/your_wakeword/train/model/model_best.pt --threshold 0.75
```

## 5. 进阶功能

### 5.1 数据增强策略

框架实现了多种音频增强方法，提高模型鲁棒性：
- 时间拉伸 (±20%)
- 时间偏移 (±10%)
- 噪声混合 (SNR 5-20dB)
- SpecAugment频谱掩码

### 5.2 模型评估指标

评估模块提供全面的性能指标：
- 精确率-召回率曲线
- F1分数
- FRR (False Rejection Rate)
- FAPH (False Alarms Per Hour)
- FRR-FAPH曲线

### 5.3 自定义模型配置

可以调整BC-ResNet模型参数：
- `model_version`: 模型规模 (1, 1.5, 2, 3, 6, 8)
- `spec_group_num`: 频谱组数
- `batch_size`: 训练批次大小
- `total_epochs`: 训练总轮数

## 6. 实际应用案例

### 6.1 智能家居语音控制

```bash
# 训练"打开电视"唤醒词
bash train_custom_wakeword_detection_model.sh --wakeword "打开电视" --workspace ./demo/tv_control
```

### 6.2 多语言唤醒词

```bash
# 训练英文唤醒词
bash train_custom_wakeword_detection_model.sh --wakeword "institute of technology" --workspace ./demo/mit

# 训练中文唤醒词
bash train_custom_wakeword_detection_model.sh --wakeword "你好智能" --workspace ./demo/hello_ai
```

## 7. 常见问题

**Q: 如何为新的唤醒词准备参考音频？**
A: 您可以使用质量较好的其他唤醒词音频作为参考，框架会自动提取语音特征并克隆目标唤醒词。

**Q: 训练数据量的推荐比例是多少？**
A: 推荐正负样本时长比例为1:10~1:50，即负样本总时长是正样本的10-50倍。

**Q: 如何调整模型的灵敏度？**
A: 可以通过调整部署时的阈值参数，较高的阈值会减少误触发但可能增加漏检率。

## 8. 许可证

CUVOW框架基于Apache 2.0许可证开源。