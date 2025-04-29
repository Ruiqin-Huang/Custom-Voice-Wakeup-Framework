#!/bin/bash
set -e
. ./path.sh || exit 1 # source path.sh

# Usage: 
# ./run_audio_diarization-cluster.sh --help to see the help message

# 定义默认值
# DEFAULT_AUDIO_DIR="./example/audio_7_speakers"
# DEFAULT_WORKSPACE="./workspaces/demo_aishell-1_7_speakers"
# DEFAULT_NUM_SPEAKERS=2
# DEFAULT_GPUS=""
# DEFAULT_PROC_PER_NODE=8
# DEFAULT_RUN_STAGE="1 2 3 4"
# DEFAULT_USE_GPU=false

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
# audio_dir=$DEFAULT_AUDIO_DIR
# workspace=$DEFAULT_WORKSPACE
# num_speakers=$DEFAULT_NUM_SPEAKERS
# gpus=$DEFAULT_GPUS
# proc_per_node=$DEFAULT_PROC_PER_NODE
# run_stage=$DEFAULT_RUN_STAGE
# use_gpu=$DEFAULT_USE_GPU


# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --
        --help)
            show_usage
            ;;
        *)
            echo "未知选项: $1"
            show_usage
            ;;
    esac
done

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
# echo "  workspace: $workspace"
# echo "  num_speakers: $num_speakers"
# echo "  use_gpu: $use_gpu"
# echo "  gpus: $gpus"
# echo "  proc_per_node: $proc_per_node"
# echo "  run_stage: $run_stage"

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
        python ${SCRIPT_DIR}/local/train_model.py --workspace "$workspace" --model_version "$model_version" --batch_size "$train_batch_size" --window_stride_ratio "$train_window_stride_ratio" --total_epochs "$total_epochs" --warmup_epoch "$warmup_epoch" --init_lr "$init_lr" --lr_lower_limit "$lr_lower_limit" --weight_decay "$weight_decay" --momentum "$momentum" --gpu "$gpu" --use_gpu
    else
        python ${SCRIPT_DIR}/local/train_model.py --workspace "$workspace" --model_version "$model_version" --batch_size "$train_batch_size" --window_stride_ratio "$train_window_stride_ratio" --total_epochs "$total_epochs" --warmup_epoch "$warmup_epoch" --eval_on_dev_epoch_stride "$eval_on_dev_epoch_stride" --init_lr "$init_lr" --lr_lower_limit "$lr_lower_limit" --weight_decay "$weight_decay" --momentum "$momentum"
    fi
else
    echo "++++++++ Skipping Stage 4: Train the model ++++++++"
fi

# # Stage 5: Evaluate the model on test dataset
# # 在测试集上评估模型
# if [[ $run_stage =~ (^|[[:space:]])5($|[[:space:]]) ]]; then
#     echo "++++++++ Stage 5: Evaluate the model on test dataset ++++++++"
#     if [[ "$use_gpu" == "true" ]]; then
#         python ${SCRIPT_DIR}/local/evaluate_model.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers" --gpu "$gpus" --use_gpu
#     else
#         python ${SCRIPT_DIR}/local/evaluate_model.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers"
#     fi
# else
#     echo "++++++++ Skipping Stage 5: Evaluate the model on test dataset ++++++++"
# fi

# # Stage 6: Generate results
# # 生成结果
# if [[ $run_stage =~ (^|[[:space:]])6($|[[:space:]]) ]]; then
#     echo "++++++++ Stage 6: Generate results ++++++++"
#     if [[ "$use_gpu" == "true" ]]; then
#         python ${SCRIPT_DIR}/local/generate_results.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers" --gpu "$gpus" --use_gpu
#     else
#         python ${SCRIPT_DIR}/local/generate_results.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers"
#     fi
# else
#     echo "++++++++ Skipping Stage 6: Generate results ++++++++"
# fi

# # Stage 7: Continue training the model(optional)
# # 使用额外的数据集继续训练模型（可选），数据集输入格式与原始数据集保持一致
# if [[ $run_stage =~ (^|[[:space:]])7($|[[:space:]]) ]]; then
#     echo "++++++++ Stage 7: Continue training the model ++++++++"
#     if [[ "$use_gpu" == "true" ]]; then
#         python ${SCRIPT_DIR}/local/continue_training.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers" --gpu "$gpus" --use_gpu
#     else
#         python ${SCRIPT_DIR}/local/continue_training.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers"
#     fi
# else
#     echo "++++++++ Skipping Stage 7: Continue training the model ++++++++"
# fi

# echo "++++++++ All stages completed ++++++++"

# # Stage 1: Generate dataset
# if [[ $run_stage =~ (^|[[:space:]])1($|[[:space:]]) ]]; then
#     echo "++++++++ Stage 1: Generate metadata for the dataset ++++++++"
#     if [[ "$use_gpu" == "true" ]]; then
#         python ${SCRIPT_DIR}/local/audio_diarization.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers" --gpu "$gpus" --use_gpu
#     else
#         python ${SCRIPT_DIR}/local/audio_diarization.py --input_dir "$audio_dir" --workspace "$workspace" --num_speakers "$num_speakers"
#     fi
    
#     # 执行语音识别
#     # 修改metadata.csv文件，填充transcription列（注意transcription列可以放到audio_diarization.py中创建）
#     # python ${SCRIPT_DIR}/local/语音识别.py --input_dir "$audio_dir" --workspace "$workspace"

#     # 将在workspace目录下构建dataset文件夹
#     # 构建得到的workspace/dataset文件夹格式应如
#     # - workspace/dataset
#     #    - dataset  
#     #      - audio
#     #         - 1.wav
#     #         - 2.wav
#     #         - ...
#     #      - audio_source
#     #      - metadata.csv
#     # metadata.csv包含了音频文件的路径和对应的说话人标签
# else
#     echo "++++++++ Skipping Stage 1: Generate metadata for the dataset ++++++++"
# fi

# # Stage 2: Perform VAD on each audio file
# if [[ $run_stage =~ (^|[[:space:]])2($|[[:space:]]) ]]; then
#     echo "++++++++ Stage 2: Perform VAD on each audio file ++++++++"
#     if [[ "$use_gpu" == "true" ]]; then
#         torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/voice_activity_detection.py --workspace ${workspace} --gpu $gpus --use_gpu
#     else
#         torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/voice_activity_detection.py --workspace ${workspace}
#     fi
# else
#     echo "++++++++ Skipping Stage 2: Perform VAD on each audio file ++++++++"
# fi

# # Stage 3: Extract speaker embeddings
# if [[ $run_stage =~ (^|[[:space:]])3($|[[:space:]]) ]]; then
#     echo "++++++++ Stage 3: Extract speaker embeddings ++++++++"
#     torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/prepare_subseg_json.py --workspace ${workspace} --dur 1.0 --shift 0.5 --min_seg_len 0.5 --max_seg_num 150
#     speaker_model_id="iic/speech_campplus_sv_zh_en_16k-common_advanced" # 预训练声纹提取模型
#     if [[ "$use_gpu" == "true" ]]; then
#         torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/extract_diar_embeddings.py --workspace ${workspace} --model_id $speaker_model_id --conf $conf_file --batchsize 64 --gpu $gpus --use_gpu
#     else
#         torchrun --nproc_per_node=$proc_per_node ${SCRIPT_DIR}/local/extract_diar_embeddings.py --workspace ${workspace} --model_id $speaker_model_id --conf $conf_file --batchsize 64
#     fi
# else
#     echo "++++++++ Skipping Stage 3: Extract speaker embeddings ++++++++"
# fi

# # Stage 4: Cluster embeddings
# if [[ $run_stage =~ (^|[[:space:]])4($|[[:space:]]) ]]; then
#     echo "++++++++ Stage 4: Cluster embeddings ++++++++"
#     python ${SCRIPT_DIR}/local/cluster_and_postprocess.py --workspace ${workspace} --conf $conf_file 
# else
#     echo "++++++++ Skipping Stage 4: Cluster embeddings ++++++++"
# fi

# # TODO: Need to add the evaluation stage
# # Stage 5: Evaluate and generate results
# # if [[ $run_stage =~ (^|[[:space:]])5($|[[:space:]]) ]]; then
# #     echo "++++++++ Stage 5: Evaluate and generate results ++++++++"
# #     python ${SCRIPT_DIR}/local/evaluate_cluster_result.py --workspace ${workspace}
# # else
# #     echo "++++++++ Skipping Stage 5: Evaluate and generate results ++++++++"
# # fi

# echo "++++++++ All stages completed ++++++++"
# echo "Speaker clustering completed."
# echo "results can be found in ${workspace}/results"