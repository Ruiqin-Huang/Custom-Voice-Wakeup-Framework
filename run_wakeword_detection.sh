#!/bin/bash
set -e
# . ./path.sh || exit 1 # source path.sh

# Usage: 
# ./run_audio_diarization-cluster.sh --help to see the help message

# 定义默认值
# ======== 自定义唤醒词 ========
DEFAULT_WAKEWORD="hey fire fox" # 唤醒词
# ======== 数据集设置 ========
DEFAULT_NEG_SOURCE_DIR="/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Common_Voice/en/Common_Voice_corpus_4_en_sampled_22500-5000-5000" # 负样本数据集路径
DEFAULT_POS_SOURCE_DIR="/home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-ff-data-in-mcv-format" # 正样本数据集路径
DEFAULT_NEGATIVE_TRAIN_DURATION=20000 # 负样本训练集时长（秒）
DEFAULT_NEGATIVE_DEV_DURATION=4000 # 负样本验证集时长（秒）
DEFAULT_NEGATIVE_TEST_DURATION=4000 # 负样本测试集时长（秒）
DEFAULT_POSITIVE_TRAIN_DURATION=1000 # 正样本训练集时长（秒）
DEFAULT_POSITIVE_DEV_DURATION=200 # 正样本验证集时长（秒）
DEFAULT_POSITIVE_TEST_DURATION=200 # 正样本测试集时长（秒）
# ======== 模型设置 ========
DEFAULT_MODEL_VERSION=3 # 模型版本
DEFAULT_SPEC_GROUP_NUM=5 # 频谱组数
# ======== 训练设置 ========
DEFAULT_BATCH_SIZE=128 # 批大小
DEFAULT_WINDOW_STRIDE_RATIO=0.25 # 窗口步幅比率
DEFAULT_TOTAL_EPOCHS=200 # 总训练轮数
DEFAULT_WARMUP_EPOCH=10 # 预热轮数
DEFAULT_EVAL_ON_DEV_EPOCH_STRIDE=10 # 验证集评估轮数步幅
DEFAULT_INIT_LR=1e-1 # 初始学习率
DEFAULT_LR_LOWER_LIMIT=1e-6 # 学习率下限
DEFAULT_WEIGHT_DECAY=1e-3 # 权重衰减
DEFAULT_MOMENTUM=0.9 # 动量
# ======== 推理设置 ========
# TODO: 需要添加推理阶段的参数设置
# ======== 工作区设置 ========
DEFAULT_WORKSPACE="./workspace/hff-testrun-1-20250506" # 工作目录
# ======== 设备设置 ========
DEFAULT_USE_GPU="true" # 是否使用GPU
DEFAULT_GPUS="1" # GPU设备ID,目前仅支持单GPU训练
# ======== 实验设置 ========
DEFAULT_RUN_STAGE="4" # 指定要执行的阶段 (1-5)，用空格分隔

# 帮助函数
# show_usage() {
#     echo "Usage: ./run_audio_diarization-cluster.sh [OPTIONS]"
#     echo "Options:"
#     echo "  --audio_dir DIR           包含需要聚类的wav音频文件的目录 (默认: $DEFAULT_AUDIO_DIR)"
#     echo "  --workspace DIR           工作目录，用于存放输出结果 (默认: $DEFAULT_WORKSPACE)"
#     echo "  --num_speakers NUM        输入原始音频中的说话人数量 (默认: $DEFAULT_NUM_SPEAKERS)"
#     echo "  --gpus \"ID1 ID2...\"       要使用的GPU设备ID (默认: \"$DEFAULT_GPUS\")"
#     echo "  --use_gpu                 使用GPU进行计算 (需同时指定 --gpus)"
#     echo "  --proc_per_node NUM       每个节点的进程数量 (默认: $DEFAULT_PROC_PER_NODE)"
#     echo "  --run_stage \"STAGES...\"     指定要执行的阶段 (1-5)，用空格分隔 (默认: \"$DEFAULT_RUN_STAGE\")"
#     echo "  --help                    显示帮助信息"
#     exit 1
# }

# 初始化参数为默认值
# ======== 自定义唤醒词 ========
wakeword="$DEFAULT_WAKEWORD" # 唤醒词
# ======== 数据集设置 ========
neg_source_dir="$DEFAULT_NEG_SOURCE_DIR" # 负样本数据集路径
pos_source_dir="$DEFAULT_POS_SOURCE_DIR" # 正样本数据集路径
negative_train_duration="$DEFAULT_NEGATIVE_TRAIN_DURATION" # 负样本训练集时长（秒）
negative_dev_duration="$DEFAULT_NEGATIVE_DEV_DURATION" # 负样本验证集时长（秒）
negative_test_duration="$DEFAULT_NEGATIVE_TEST_DURATION" # 负样本测试集时长（秒）
positive_train_duration="$DEFAULT_POSITIVE_TRAIN_DURATION" # 正样本训练集时长（秒）
positive_dev_duration="$DEFAULT_POSITIVE_DEV_DURATION" # 正样本验证集时长（秒）
positive_test_duration="$DEFAULT_POSITIVE_TEST_DURATION" # 正样本测试集时长（秒）
# ======== 模型设置 ========
model_version="$DEFAULT_MODEL_VERSION" # 模型版本
spec_group_num="$DEFAULT_SPEC_GROUP_NUM" # 频谱组数
# ======== 训练设置 ========
batch_size="$DEFAULT_BATCH_SIZE" # 批大小
train_window_stride_ratio="$DEFAULT_WINDOW_STRIDE_RATIO" # 窗口步幅比率
total_epochs="$DEFAULT_TOTAL_EPOCHS" # 总训练轮数
warmup_epoch="$DEFAULT_WARMUP_EPOCH" # 预热轮数
eval_on_dev_epoch_stride="$DEFAULT_EVAL_ON_DEV_EPOCH_STRIDE" # 验证集评估轮数步幅
init_lr="$DEFAULT_INIT_LR" # 初始学习率
lr_lower_limit="$DEFAULT_LR_LOWER_LIMIT" # 学习率下限
weight_decay="$DEFAULT_WEIGHT_DECAY" # 权重衰减
momentum="$DEFAULT_MOMENTUM" # 动量
# ======== 推理设置 ========
# TODO: 需要添加推理阶段的参数设置
# ======== 工作区设置 ========
workspace="$DEFAULT_WORKSPACE" # 工作目录
# ======== 设备设置 ========
use_gpu="$DEFAULT_USE_GPU" # 是否使用GPU
gpu="$DEFAULT_GPUS" # GPU设备ID,目前仅支持单GPU训练
# ======== 实验设置 ========
run_stage="$DEFAULT_RUN_STAGE" # 指定要执行的阶段 (1-5)，用空格分隔


# 解析命令行参数
# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --
#         --help)
#             show_usage
#             ;;
#         *)
#             echo "未知选项: $1"
#             show_usage
#             ;;
#     esac
# done

# 验证 gpu 和 use_gpu 参数的组合是否合理
# if [[ -n "$gpus" && "$use_gpu" == "false" ]]; then # 检查$gpus是否非空
#     echo "错误: 指定了 --gpus 参数但未指定 --use_gpu 参数"
#     show_usage
# fi

# if [[ "$use_gpu" == "true" && -z "$gpus" ]]; then # 检查$gpus是否为空
#     echo "错误: 指定了 --use_gpu 参数但未指定 --gpus 参数"
#     show_usage
# fi

# 显示已配置的参数
echo "实验配置参数:"
echo "========自定义唤醒词========"
echo "wakeword: $wakeword"
echo "========数据集设置========"
echo "neg_source_dir: $neg_source_dir"
echo "pos_source_dir: $pos_source_dir"
echo "negative_train_duration: $negative_train_duration"
echo "negative_dev_duration: $negative_dev_duration"
echo "negative_test_duration: $negative_test_duration"
echo "positive_train_duration: $positive_train_duration"
echo "positive_dev_duration: $positive_dev_duration"
echo "positive_test_duration: $positive_test_duration"
echo "========模型设置========"
echo "  model_version: $model_version"
echo "  spec_group_num: $spec_group_num"
echo "========训练设置========"
echo "  batch_size: $batch_size"
echo "  window_stride_ratio: $train_window_stride_ratio"
echo "  total_epochs: $total_epochs"
echo "  warmup_epoch: $warmup_epoch"
echo "  eval_on_dev_epoch_stride: $eval_on_dev_epoch_stride"
echo "  init_lr: $init_lr"
echo "  lr_lower_limit: $lr_lower_limit"
echo "  weight_decay: $weight_decay"
echo "  momentum: $momentum"
echo "========推理设置========"
# TODO: 需要添加推理阶段的参数设置
echo "========工作区设置========"
echo "  workspace: $workspace"
echo "========设备设置========"
echo "  use_gpu: $use_gpu"
echo "  gpu: $gpu"
# TODO: 多GPU分布式训练
# echo "  proc_per_node: $proc_per_node"
echo "========实验设置========"
echo "  run_stage: $run_stage"


# 在切换目录【前】保存原始脚本目录的绝对路径
SCRIPT_DIR=$(dirname "$(realpath "$0")") # 读取脚本及指令文件需要需要使用来源目录的绝对路径（本.sh脚本所在路径），但是在workspace工作目录下执行
# conf_file=${SCRIPT_DIR}/conf/diar.yaml # 模型配置文件路径，指定extract_diar_embeddings及cluster_and_postprocess过程中模型设置

# Create workspace directory
mkdir -p ${workspace}

# 本实验所有音频采样率为16kHz(16000Hz)

# Stage 1: Generate negative dataset
# 生成负样本数据集（从通用语料库数据集）
if [[ $run_stage =~ (^|[[:space:]])1($|[[:space:]]) ]]; then
    echo "++++++++ Stage 1: Generate negative dataset ++++++++"
    # 从Mozilla Common Voice数据集中生成负样本数据集，这里采样的依据不是根据音频数量，而是根据音频时长。（考虑到正负集中每条音频时长不相同，本实验中提到的【正集:负集】的比例指的是音频时长的比例，而非音频数量的比例）
    python ${SCRIPT_DIR}/local/generate_negative_dataset.py --wakeword "$wakeword" --mcv_dir "$neg_source_dir" --workspace "$workspace" --negative_train_duration "$negative_train_duration" --negative_dev_duration "$negative_dev_duration" --negative_test_duration "$negative_test_duration"
else
    echo "++++++++ Skipping Stage 1: Generate negative dataset ++++++++"
fi

# Stage 2: Generate positive dataset
# 生成正样本数据集，有三种生成形式，分别是：[1]加载录制得到的真实唤醒词数据集 [2]加载从mcv中对齐+拼接得到的唤醒词数据集 [3]加载TTS合成得到的唤醒词数据集 [4]加载克隆得到的唤醒词数据集
if [[ $run_stage =~ (^|[[:space:]])2($|[[:space:]]) ]]; then
    echo "++++++++ Stage 2: Generate positive dataset ++++++++"
    # 从指定的正样本数据集中生成正样本数据集，这里采样的依据不是根据音频数量，而是根据音频时长。（考虑到正负集中每条音频时长不相同，本实验中提到的【正集:负集】的比例指的是音频时长的比例，而非音频数量的比例）
    python ${SCRIPT_DIR}/local/generate_positive_dataset.py --wakeword "$wakeword" --pos_source_dir "$pos_source_dir" --workspace "$workspace" --positive_train_duration "$positive_train_duration" --positive_dev_duration "$positive_dev_duration" --positive_test_duration "$positive_test_duration"
else
    echo "++++++++ Skipping Stage 2: Generate positive dataset ++++++++"
fi

# Stage 3: Generate dataset_info.json for the dataset
# 生成数据集统计信息dataset_info.json，依据workspace/dataset/positive和workspace/dataset/negative目录下的jsonl文件生成
if [[ $run_stage =~ (^|[[:space:]])3($|[[:space:]]) ]]; then
    echo "++++++++ Stage 3: Generate dataset_info.json for the dataset ++++++++"
    python ${SCRIPT_DIR}/local/generate_dataset_info.py --workspace "$workspace"
else
    echo "++++++++ Skipping Stage 3: Generate dataset_info.json for the dataset ++++++++"
fi

# Stage 4: Train the model
# 训练模型，依据指定的模型配置文件进行训练，其中训练和推理过程中的window_size及window_stride均取决于正集训练集的平均音频时长，window_stride = window_stride_ratio * window_size
if [[ $run_stage =~ (^|[[:space:]])4($|[[:space:]]) ]]; then
    echo "++++++++ Stage 4: Train the model ++++++++"
    if [[ "$use_gpu" == "true" ]]; then
    # eg.: $gpu = "0" # 使用的GPU设备ID
        python ${SCRIPT_DIR}/local/train_model.py --workspace "$workspace" --model_version "$model_version" --spec_group_num "$spec_group_num" --batch_size "$batch_size" --window_stride_ratio "$train_window_stride_ratio" --total_epochs "$total_epochs" --warmup_epoch "$warmup_epoch" --eval_on_dev_epoch_stride "$eval_on_dev_epoch_stride" --init_lr "$init_lr" --lr_lower_limit "$lr_lower_limit" --weight_decay "$weight_decay" --momentum "$momentum" --gpu "$gpu" --use_gpu
    else
        python ${SCRIPT_DIR}/local/train_model.py --workspace "$workspace" --model_version "$model_version" --spec_group_num "$spec_group_num" --batch_size "$batch_size" --window_stride_ratio "$train_window_stride_ratio" --total_epochs "$total_epochs" --warmup_epoch "$warmup_epoch" --eval_on_dev_epoch_stride "$eval_on_dev_epoch_stride" --init_lr "$init_lr" --lr_lower_limit "$lr_lower_limit" --weight_decay "$weight_decay" --momentum "$momentum"
    fi
else
    echo "++++++++ Skipping Stage 4: Train the model ++++++++"
fi

# Stage 5: Eval on the test set
# 在测试集上评估模型性能并生成结果报告
if [[ $run_stage =~ (^|[[:space:]])5($|[[:space:]]) ]]; then
    echo "++++++++ Stage 5: Eval on the test set ++++++++"
    if [[ "$use_gpu" == "true" ]]; then
        python ${SCRIPT_DIR}/local/eval_model.py --workspace "$workspace" --model_version "$model_version" --spec_group_num "$spec_group_num" --batch_size "$batch_size" --window_stride_ratio "$train_window_stride_ratio" --gpu "$gpu" --use_gpu
    else
        python ${SCRIPT_DIR}/local/eval_model.py --workspace "$workspace" --model_version "$model_version" --spec_group_num "$spec_group_num" --batch_size "$batch_size" --window_stride_ratio "$train_window_stride_ratio"
    fi
else
    echo "++++++++ Skipping Stage 5: Eval on the test set ++++++++"
fi
