import argparse
import os
import random
import sys
import torchaudio
import logging
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'cosyvoice/third_party/Matcha-TTS'))

'''
# 调用样例
python clone_audio_with_cosyvoice.py --reference_audio_dir /hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-fire-fox-real/clips --num_samples 442 --wakeword "hey fire fox" --workspace /hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/bcresnet/data/clone_dataset --model_path /hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/bcresnet/local/cosyvoice/pretrained_models/CosyVoice2-0.5B --output_audio_dir "$pos_source_dir"
'''

def setup_logger(workspace):
    """配置日志记录器，同时输出到控制台和文件"""
    log_dir = os.path.join(workspace, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "generate_dataset_log.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器（如果有）
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 确保可以找到CosyVoice模块
# 您可能需要根据您的项目结构调整此路径
sys.path.append('./cosyvoice/third_party/Matcha-TTS') # 假设CosyVoice在您的项目中的这个路径下
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    logging.error("[ERROR] 无法导入CosyVoice模块。请确保：")
    logging.error("[ERROR] 1. 'cosyvoice/third_party/Matcha-TTS' 路径正确，并且包含CosyVoice库。")
    logging.error("[ERROR] 2. CosyVoice及其依赖已正确安装。")
    sys.exit(1)

def get_transcription_from_lab(audio_path):
    """
    从与音频文件同名的.lab文件中读取转录文本
    """
    base_path = os.path.splitext(audio_path)[0]
    lab_path = base_path + '.lab'
    
    if os.path.exists(lab_path):
        try:
            with open(lab_path, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
                return transcription
        except Exception as e:
            logging.error(f"[ERROR] 读取转录文件 '{lab_path}' 时出错: {e}")
    
    # 如果没有找到.lab文件或读取失败，返回None
    return None

def clone_audio(args):
    """
    使用CosyVoice从指定文件夹中随机选取n条音频作为参考音色，
    克隆得到n条音频，并生成train.tsv文件。
    """
    # 设置日志记录器
    logger = setup_logger(args.workspace)
    
    logging.info(f"======== 使用COSYVOICE 2 克隆音频 ========")
    
    # 检查参考音频目录是否存在
    if not os.path.isdir(args.reference_audio_dir):
        logging.error(f"[ERROR] 参考音频目录 '{args.reference_audio_dir}' 不存在或不是一个目录。")
        return

    # 获取所有支持的音频文件
    supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    all_audio_files = []
    for root, _, files in os.walk(args.reference_audio_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                all_audio_files.append(os.path.join(root, file))

    if not all_audio_files:
        logging.error(f"[ERROR] 在目录 '{args.reference_audio_dir}' 中未找到支持的音频文件。")
        return

    # 随机选择n条音频作为参考
    if args.num_samples > len(all_audio_files):
        logging.warning(f"[WARNING] 请求的样本数 ({args.num_samples}) 大于可用音频文件数 ({len(all_audio_files)})。")
        logging.warning(f"[WARNING] 将使用所有可用的 {len(all_audio_files)} 个音频文件。")
        selected_audio_files = all_audio_files
    else:
        selected_audio_files = random.sample(all_audio_files, args.num_samples)

    logging.info(f"[INFO] 已选择 {len(selected_audio_files)} 个音频文件作为参考音色。")

    # 使用output_audio_dir参数（如果提供）或默认路径
    if args.output_audio_dir:
        output_dir = args.output_audio_dir
        output_clips_dir = os.path.join(output_dir, "clips")
    else:
        output_dir = os.path.join(args.workspace, "dataset", "clone_output")
        output_clips_dir = os.path.join(output_dir, "clips")
    
    # 创建输出目录
    os.makedirs(output_clips_dir, exist_ok=True)
    logging.info(f"[INFO] 克隆的音频将保存到: {output_clips_dir}")

    # 初始化CosyVoice
    logging.info("[INFO] 正在初始化CosyVoice模型...")
    try:
        cosyvoice = CosyVoice2(args.model_path, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)
    except Exception as e:
        logging.error(f"[ERROR] 初始化CosyVoice时出错: {e}")
        logging.error("[ERROR] 请确保模型路径正确，并且模型文件已下载且完整。")
        return
    logging.info("[INFO] CosyVoice模型初始化完成。")

    tsv_entries = []

    logging.info(f"[INFO] 开始克隆音频，目标唤醒词: '{args.wakeword}'")
    for i, ref_audio_path in enumerate(tqdm(selected_audio_files, desc="克隆音频中")):
        try:
            # 加载参考音频
            prompt_speech_16k = load_wav(ref_audio_path, 16000) # CosyVoice期望16kHz采样率
            
            # 从.lab文件读取转录文本作为风格提示
            prompt_text = get_transcription_from_lab(ref_audio_path)
            
            # 如果没有找到转录文本，使用默认的"hey snips"作为备选
            if not prompt_text:
                logging.warning(f"[WARNING] 未找到音频 '{ref_audio_path}' 对应的转录文件，使用默认文本作为风格提示。")
                prompt_text = "hey snips"
                
            # 执行零样本语音克隆
            cloned_speech_stream = cosyvoice.inference_zero_shot(
                args.wakeword,           # 要合成的文本
                prompt_text,             # 从.lab文件读取的转录文本作为风格提示
                prompt_speech_16k,       # 参考音色
                stream=False             # 非流式输出
            )

            # 保存克隆的音频
            for idx, result in enumerate(cloned_speech_stream):
                if 'tts_speech' in result and result['tts_speech'] is not None:
                    # 生成唯一的文件名
                    base_filename = os.path.splitext(os.path.basename(ref_audio_path))[0]
                    output_filename = f"cloned_{base_filename}_{i}_{idx}.wav"
                    output_path = os.path.join(output_clips_dir, output_filename)
                    torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
                    tsv_entries.append(f"{output_filename}\t{args.wakeword}")
                else:
                    logging.warning(f"[WARNING] 克隆参考音频 '{ref_audio_path}' 的第 {idx} 个结果未包含 'tts_speech' 或为 None。")

        except Exception as e:
            logging.error(f"[ERROR] 处理参考音频 '{ref_audio_path}' 时出错: {e}")
            logging.warning("[WARNING] 跳过此参考音频。")

    # 生成train.tsv文件
    tsv_output_path = os.path.join(output_dir, "train.tsv")
    try:
        with open(tsv_output_path, 'w', encoding='utf-8') as f:
            f.write("path\tsentence\n") # 写入表头
            for entry in tsv_entries:
                f.write(entry + "\n")
        logging.info(f"[INFO] train.tsv 文件已生成: {tsv_output_path}")
    except Exception as e:
        logging.error(f"[ERROR] 写入 train.tsv 文件时出错: {e}")

    logging.info("[INFO] 音频克隆和TSV文件生成完成。")
    logging.info("====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用CosyVoice克隆音频并生成训练TSV文件。")
    parser.add_argument("--reference_audio_dir", type=str, required=True,
                        help="包含参考音频文件的目录。")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="要从参考目录中随机选择的音频样本数量。")
    parser.add_argument("--wakeword", type=str, required=True,
                        help="要克隆并输出的唤醒词文本内容。")
    parser.add_argument("--workspace", type=str, required=True,
                        help="工作目录，用于存放日志文件。")
    parser.add_argument("--output_audio_dir", type=str, default=None,
                        help="输出目录，克隆的音频和TSV文件将保存在此目录下。如不指定，则使用workspace目录。")
    parser.add_argument("--model_path", type=str, default="pretrained_models/CosyVoice-300M",
                        help="CosyVoice预训练模型的路径。例如：'pretrained_models/CosyVoice2-0.5B'")

    args = parser.parse_args()
    clone_audio(args)