# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from argparse import ArgumentParser
import shutil
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from bcresnet import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset


class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--ver", default=1, help="google speech command set version 1 or 2", type=int, choices=[1, 2]
        )
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        args = parser.parse_args()
        self.__dict__.update(vars(args)) 
        # 将命令行解析的参数直接转换为类的实例属性
        # 这行代码执行后，命令行参数自动成为了 Trainer 类的实例属性，例如：
            # --ver 参数值可通过 self.ver 访问
            # --tau 参数值可通过 self.tau 访问
            # --gpu 参数值可通过 self.gpu 访问
            # --download 参数值可通过 self.download 访问
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """
        # 超参数设置
        total_epoch = 200 # 训练的总轮数
        warmup_epoch = 5 # 预热轮数，前5个epoch使用线性增长的学习率（正常训练时，学习率会逐渐减小，余弦退火），warmup_epoch 主要用于在训练初期逐渐增加学习率，以避免模型在训练初期的震荡和不稳定。
        # 代码实现了一个包含预热阶段的余弦退火学习率调度：
        #     预热阶段：前5个epoch线性增加学习率至0.1
        #     退火阶段：余下195个epoch余弦递减学习率
        init_lr = 1e-1 # 初始学习率
        lr_lower_limit = 0 # 学习率下限，训练过程中学习率会逐渐减小到这个值

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9) # 使用SGD优化器，初始学习率设为0（后动态调整）。weight_decay=1e-3提供L2正则化，防止过拟合。momentum=0.9加速收敛并帮助跳出局部最小值
        n_step_warmup = len(self.train_loader) * warmup_epoch
        # n_step_warmup：预热阶段的总迭代次数
        #     len(self.train_loader)：表示一个epoch中的批次数量
        #     warmup_epoch：预热阶段的轮数（这里是5）
        total_iter = len(self.train_loader) * total_epoch
        # total_iter：整个训练过程的总迭代次数
        #     total_epoch：总训练轮数（这里是200）
        #     计算结果是整个训练过程中的总批次数
        iterations = 0 # iterations：当前迭代计数器，初始化为0

        # train
        for epoch in range(total_epoch):
            self.model.train()
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    # 预热阶段，学习率线性增加
                    lr = init_lr * iterations / n_step_warmup
                else:
                    # 余弦退火阶段，学习率逐渐减小
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr # 更新优化器的学习率

                # 这里的 sample 是一个 batch 的数据，包含了输入和标签
                # inputs 的第一维度的大小是 batch_size，第二维度是输入的特征维度
                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True) # 预处理训练数据，进行数据增强，包括1. 噪声增强 2. 时间偏移 3. 频谱增强（具体是否执行，依据选择的BC-Resnet版本）
                outputs = self.model(inputs) # 得到模型输出，这里的输出是一个 batch 的预测结果，形状是[batch_size, num_classes]
                loss = F.cross_entropy(outputs, labels) # 计算交叉熵损失函数
                loss.backward() # 反向传播计算梯度
                optimizer.step() # 更新模型参数
                self.model.zero_grad() # 清除梯度，准备下一次迭代

            # 每个epoch结束后，在验证集上进行测试
            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(self.valid_dataset, self.valid_loader, augment=True) # 在验证集上进行测试，使用数据增强，包括1. 噪声增强 2. 时间偏移，Test时不使用频谱增强
                print("valid acc: %.3f" % (valid_acc))
        # 训练结束后，在测试集上进行测试
        test_acc = self.Test(self.test_dataset, self.test_loader, augment=False)  # 测试集上不使用数据增强
        print("test acc: %.3f" % (test_acc))
        print("End.")

    def Test(self, dataset, loader, augment):
        """
        Tests the model on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
        """
        true_count = 0.0
        num_testdata = float(len(dataset))
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs) # 输出形状：[batch_size, num_classes]
            prediction = torch.argmax(outputs, dim=-1) # 以最大值的索引作为预测结果，输出形状：[batch_size]
            true_count += torch.sum(prediction == labels).detach().cpu().numpy() # 计算预测正确的样本数
        acc = true_count / num_testdata * 100.0  # percentage
        return acc

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """
        print("Check google speech commands dataset v1 or v2 ...")
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        base_dir = "./data/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
            url_test = url_test.replace("v0.01", "v0.02")
        test_dir = base_dir.replace("commands", "commands_test_set")
        if self.download:
            # 下载主数据集及测试集
            old_dirs = glob(base_dir.replace("commands_", "commands_*"))
            for old_dir in old_dirs:
                shutil.rmtree(old_dir)
            os.mkdir(test_dir)
            DownloadDataset(test_dir, url_test) # 下载并解压测试集
            os.mkdir(base_dir)
            DownloadDataset(base_dir, url) # 下载并解压主数据集
            SplitDataset(base_dir) # 划分数据集，只划分主数据集。根据数据集提供的validation_list.txt和testing_list.txt列表文件，调用split_data将音频文件分为训练/验证/测试三组
            print("Done...")

        # Define data loaders
        train_dir = "%s/train_12class" % base_dir
        valid_dir = "%s/valid_12class" % base_dir
        noise_dir = "%s/_background_noise_" % base_dir

        transform = transforms.Compose([Padding()])
        # DataLoader 的第一个参数必须是一个实现了 PyTorch Dataset 抽象类的对象。任何这样的对象都必须实现两个核心方法：
        #     __len__(self): 返回数据集中样本的总数
        #     __getitem__(self, idx): 根据索引返回一个样本（及其标签）
        # SpeechCommand 类正是这样一个自定义的 Dataset 子类
        self.train_dataset = SpeechCommand(train_dir, self.ver, transform=transform)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, drop_last=False 
            # num_workers=0，所有数据加载操作都在主进程中进行，不使用额外的子进程。【默认值为0】
            # drop_last=False，保留最后一个不完整的批次（如果数据集大小不能被batch_size整除）。【默认值为False，即保留最后一个批次】
        )
        # 使用 shuffle=True 确保每个训练周期数据顺序不同的原因：
        #     1. 防止模型记住训练顺序：如果数据总是以相同顺序出现，模型可能学习到这种顺序关系而非真实模式
        #     2. 避免批次偏差：如果数据是按类别排序的，不打乱会导致某些批次只包含单一类别
        #     3. 提高模型泛化能力：随机顺序迫使模型学习真正有用的特征模式
        # 即使设置 shuffle=True，DataLoader 也会确保在每个 epoch 中遍历完数据集中的所有样本。
        # 打乱操作只改变访问顺序，让模型在不同 epoch 会看到不同顺序的数据，不会跳过或重复任何样本。
        self.valid_dataset = SpeechCommand(valid_dir, self.ver, transform=transform)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=100, num_workers=0)
        self.test_dataset = SpeechCommand(test_dir, self.ver, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, num_workers=0)

        print(
            "check num of data train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )
        
        
        # 只有在使用BC-Resnet-1.5及以上版本时，才会使用SpecAugment应用频谱增强
        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device) # 测试时不使用SpecAugment数据增强(频域增强)

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        print("model: BC-ResNet-%.1f on data v0.0%d" % (self.tau, self.ver))
        self.model = BCResNets(base_c = int(self.tau * 8), num_classes = 12).to(self.device)
        # 依据参数构建BC-ResNet网络模型
            # 1. base_c = int(self.tau * 8)
            # base_c参数是BCResNet网络架构中的核心缩放因子，它直接决定了网络的宽度（通道数），从而影响模型的复杂度、参数量和计算量。
            
            # 2. num_classes = 12
            # num_classes参数指定了模型的输出类别数，这里设置为12，表示模型将用于12类语音命令的分类任务。
            # 需要依据语音分类任务中的具体类别数进行调整。

if __name__ == "__main__":
    _trainer = Trainer() # 加载数据，初始化模型
    _trainer() # 调用__call__方法，开始训练和测试
