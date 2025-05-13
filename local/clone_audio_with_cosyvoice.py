import argparse
import os
import random
import sys
import torchaudio
from tqdm import tqdm

'''
# 调用样例
python clone_audio_with_cosyvoice.py --reference_audio_dir /home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/datasets/Hey-Fire-Fox/hey-fire-fox-real/clips --num_samples 442 --text_to_clone "hey fire fox" --workspace /home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/bcresnet/data/clone_dataset --model_path /home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/bcresnet/local/cosyvoice/pretrained_models/CosyVoice2-0.5B
'''

# 确保可以找到CosyVoice模块
# 您可能需要根据您的项目结构调整此路径
sys.path.append('cosyvoice/third_party/Matcha-TTS') # 假设CosyVoice在您的项目中的这个路径下
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    print("错误：无法导入CosyVoice模块。请确保：")
    print("1. 'third_party/Matcha-TTS' 路径正确，并且包含CosyVoice库。")
    print("2. CosyVoice及其依赖已正确安装。")
    print("您可以尝试将 'third_party/Matcha-TTS' 的绝对路径添加到PYTHONPATH环境变量中。")
    sys.exit(1)

def clone_audio(args):
    """
    使用CosyVoice从指定文件夹中随机选取n条音频作为参考音色，
    克隆得到n条音频，并生成train.tsv文件。
    """
    # 检查参考音频目录是否存在
    if not os.path.isdir(args.reference_audio_dir):
        print(f"错误：参考音频目录 '{args.reference_audio_dir}' 不存在或不是一个目录。")
        return

    # 获取所有支持的音频文件
    supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    all_audio_files = []
    for root, _, files in os.walk(args.reference_audio_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                all_audio_files.append(os.path.join(root, file))

    if not all_audio_files:
        print(f"错误：在目录 '{args.reference_audio_dir}' 中未找到支持的音频文件。")
        return

    # 随机选择n条音频作为参考
    if args.num_samples > len(all_audio_files):
        print(f"警告：请求的样本数 ({args.num_samples}) 大于可用音频文件数 ({len(all_audio_files)})。")
        print(f"将使用所有可用的 {len(all_audio_files)} 个音频文件。")
        selected_audio_files = all_audio_files
    else:
        selected_audio_files = random.sample(all_audio_files, args.num_samples)

    print(f"已选择 {len(selected_audio_files)} 个音频文件作为参考音色。")

    # 创建输出目录
    output_clips_dir = os.path.join(args.workspace, "clips")
    os.makedirs(output_clips_dir, exist_ok=True)
    print(f"克隆的音频将保存到: {output_clips_dir}")

    # 初始化CosyVoice
    print("正在初始化CosyVoice模型...")
    try:
        # 注意：'pretrained_models/CosyVoice2-0.5B' 是示例路径，您需要替换为实际的预训练模型路径
        cosyvoice = CosyVoice2(args.model_path, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)
    except Exception as e:
        print(f"初始化CosyVoice时出错: {e}")
        print("请确保模型路径正确，并且模型文件已下载且完整。")
        return
    print("CosyVoice模型初始化完成。")

    tsv_entries = []

    print(f"开始克隆音频，目标文本: '{args.text_to_clone}'")
    for i, ref_audio_path in enumerate(tqdm(selected_audio_files, desc="克隆音频中")):
        try:
            # 加载参考音频
            prompt_speech_16k = load_wav(ref_audio_path, 16000) # CosyVoice期望16kHz采样率

            # 执行零样本语音克隆
            # 注意：根据CosyVoice的API，inference_zero_shot可能需要两个文本提示
            # 这里我们使用相同的文本作为两个提示，或者您可以根据需要调整
            # '希望你以后能够做的比我还好呦。' 是示例中的第二个提示，这里我们用args.text_to_clone代替
            # 或者，如果您的CosyVoice版本或用法允许单个文本，请相应调整。
            # 查阅您使用的CosyVoice版本的文档以获取准确的API用法。
            # 假设 inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, ...)
            # 如果您的模型或用法不同，请调整这里的参数。
            # 为了简单起见，这里假设第一个文本是主要文本，第二个是风格提示文本（如果需要）
            # 如果您的模型不需要第二个文本提示，可以将其设置为空字符串或根据API调整。
            cloned_speech_stream = cosyvoice.inference_zero_shot(
                args.text_to_clone,       # 要合成的文本
                'hey snips',       # 作为风格提示的文本 (如果模型需要)
                prompt_speech_16k,        # 参考音色
                stream=False              # 非流式输出
            )

            # 保存克隆的音频
            # inference_zero_shot 返回一个生成器，即使stream=False
            for idx, result in enumerate(cloned_speech_stream):
                if 'tts_speech' in result and result['tts_speech'] is not None:
                    # 生成唯一的文件名
                    base_filename = os.path.splitext(os.path.basename(ref_audio_path))[0]
                    output_filename = f"cloned_{base_filename}_{i}_{idx}.wav"
                    output_path = os.path.join(output_clips_dir, output_filename)
                    torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
                    tsv_entries.append(f"{output_filename}\t{args.text_to_clone}")
                else:
                    print(f"警告：克隆参考音频 '{ref_audio_path}' 的第 {idx} 个结果未包含 'tts_speech' 或为 None。")

        except Exception as e:
            print(f"处理参考音频 '{ref_audio_path}' 时出错: {e}")
            print("跳过此参考音频。")

    # 生成train.tsv文件
    tsv_output_path = os.path.join(args.workspace, "train.tsv")
    try:
        with open(tsv_output_path, 'w', encoding='utf-8') as f:
            f.write("path\tsentence\n") # 写入表头
            for entry in tsv_entries:
                f.write(entry + "\n")
        print(f"train.tsv 文件已生成: {tsv_output_path}")
    except Exception as e:
        print(f"写入 train.tsv 文件时出错: {e}")

    print("音频克隆和TSV文件生成完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用CosyVoice克隆音频并生成训练TSV文件。")
    parser.add_argument("--reference_audio_dir", type=str, required=True,
                        help="包含参考音频文件的目录。")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="要从参考目录中随机选择的音频样本数量。")
    parser.add_argument("--text_to_clone", type=str, required=True,
                        help="要克隆并输出的文本内容。")
    parser.add_argument("--workspace", type=str, required=True,
                        help="工作目录，克隆的音频和TSV文件将保存在此目录的 'dataset/clone' 子目录下。")
    parser.add_argument("--model_path", type=str, default="pretrained_models/CosyVoice-300M", # 请修改为您的模型路径
                        help="CosyVoice预训练模型的路径。例如：'pretrained_models/CosyVoice2-0.5B'")

    args = parser.parse_args()
    clone_audio(args)