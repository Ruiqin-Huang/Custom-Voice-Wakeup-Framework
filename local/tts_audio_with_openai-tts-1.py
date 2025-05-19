import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import soundfile as sf
from dotenv import load_dotenv
import time

"""
python tts_audio_with_tts-1.py \
  --num_samples 442 \
  --text_to_generate "hey fire fox" \
  --workspace /path/to/your/workspace \
  --api_key "sk-XEQpaCeoDgTSHBozB800Ec195e0a48B395E482F9922713Fb" \
  --base_url "https://api.shubiaobiao.cn/v1/"
  
python tts_audio_with_tts-1.py --num_samples 442 --text_to_generate "hey fire fox" --workspace /home/hrq/DHG-Workspace/Research_on_Low-Cost_Custom_Voice_Wake-Up_Based_on_Voice_Cloning/baselines/KWS/bcresnet/data/tts_dataset --api_key "sk-XEQpaCeoDgTSHBozB800Ec195e0a48B395E482F9922713Fb" --base_url "https://api.shubiaobiao.cn/v1/"
"""

def generate_tts_samples(args):
    """
    使用OpenAI的TTS-1模型生成指定数量的音频样本，
    并在每个音频前后添加随机空白，然后生成train.tsv文件。
    """
    # 创建输出目录
    output_clips_dir = os.path.join(args.workspace, "clips")
    os.makedirs(output_clips_dir, exist_ok=True)
    print(f"合成的音频将保存到: {output_clips_dir}")

    # 配置OpenAI客户端
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    
    # OpenAI TTS-1 可用的说话人列表
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    # 计算每个说话人生成的样本数
    samples_per_voice = args.num_samples // len(voices)
    remainder = args.num_samples % len(voices)
    
    # 确保总样本数为442
    samples_distribution = [samples_per_voice + (1 if i < remainder else 0) for i in range(len(voices))]
    
    print(f"将生成总共 {args.num_samples} 个TTS样本，使用 {len(voices)} 个说话人声音。")
    for i, voice in enumerate(voices):
        print(f"  - {voice}: {samples_distribution[i]} 个样本")
    
    tsv_entries = []
    sample_count = 0
    
    # 为每个说话人生成样本
    for voice_idx, voice in enumerate(voices):
        for i in tqdm(range(samples_distribution[voice_idx]), 
                     desc=f"生成 {voice} 的TTS样本"):
            try:
                # 生成TTS音频
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=args.text_to_generate
                )
                
                # 临时保存原始TTS音频
                temp_audio_path = os.path.join(output_clips_dir, f"temp_{voice}_{i}.mp3")
                response.stream_to_file(temp_audio_path)
                
                # 读取生成的音频
                audio_data, sample_rate = sf.read(temp_audio_path)
                
                # 随机生成前后空白的长度（总共0.5~1.0秒）
                total_silence_seconds = random.uniform(0.5, 1.0)
                front_silence_ratio = random.random()  # 0~1之间的随机比例
                
                front_silence_seconds = total_silence_seconds * front_silence_ratio
                back_silence_seconds = total_silence_seconds * (1 - front_silence_ratio)
                
                # 计算需要添加的空白样本数
                front_silence_samples = int(front_silence_seconds * sample_rate)
                back_silence_samples = int(back_silence_seconds * sample_rate)
                
                # 创建前后的静音
                if len(audio_data.shape) == 1:  # 单声道
                    front_silence = np.zeros(front_silence_samples, dtype=audio_data.dtype)
                    back_silence = np.zeros(back_silence_samples, dtype=audio_data.dtype)
                    # 添加前后静音
                    padded_audio = np.concatenate([front_silence, audio_data, back_silence])
                else:  # 多声道
                    front_silence = np.zeros((front_silence_samples, audio_data.shape[1]), dtype=audio_data.dtype)
                    back_silence = np.zeros((back_silence_samples, audio_data.shape[1]), dtype=audio_data.dtype)
                    # 添加前后静音
                    padded_audio = np.concatenate([front_silence, audio_data, back_silence])
                
                # 生成最终输出文件名
                output_filename = f"tts_{voice}_{sample_count:03d}.wav"
                output_path = os.path.join(output_clips_dir, output_filename)
                
                # 保存添加了静音的音频
                sf.write(output_path, padded_audio, sample_rate)
                
                # 添加到TSV条目
                tsv_entries.append(f"{output_filename}\t{args.text_to_generate}")
                
                # 删除临时文件
                os.remove(temp_audio_path)
                
                sample_count += 1
                
                # 添加小延迟以避免API限制
                time.sleep(0.5)
                
            except Exception as e:
                print(f"生成样本时出错: {e}")
                print(f"跳过此样本。")
    
    # 生成train.tsv文件
    tsv_output_path = os.path.join(args.workspace, "train.tsv")
    try:
        with open(tsv_output_path, 'w', encoding='utf-8') as f:
            f.write("path\tsentence\n")  # 写入表头
            for entry in tsv_entries:
                f.write(entry + "\n")
        print(f"train.tsv 文件已生成: {tsv_output_path}")
    except Exception as e:
        print(f"写入 train.tsv 文件时出错: {e}")

    print("TTS音频生成和TSV文件创建完成。")
    print(f"总共生成了 {sample_count} 个音频样本.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用OpenAI的TTS-1模型生成音频样本并创建训练TSV文件。")
    parser.add_argument("--num_samples", type=int, default=442,
                        help="要生成的TTS样本总数量。默认为442。")
    parser.add_argument("--text_to_generate", type=str, default="hey fire fox",
                        help="要合成的文本内容。")
    parser.add_argument("--workspace", type=str, required=True,
                        help="工作目录，合成的音频将保存在此目录的'clips'子目录下，TSV文件保存在此目录下。")
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenAI API密钥。")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1",
                        help="OpenAI API基础URL，可以设置为代理或替代服务。")

    args = parser.parse_args()
    generate_tts_samples(args)