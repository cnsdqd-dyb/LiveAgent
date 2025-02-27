
import numpy as np
import librosa

# 设置参数
SR = 22050  # 采样率
CHUNK_DURATION = 0.1  # 每个块的时长（秒）

# 读取 WAV 文件并进行 chunk 处理
def analyze_chunks(filename):
    # 读取音频文件
    audio, sr = librosa.load(filename, sr=SR)
    
    # 计算每个 chunk 的样本数
    chunk_samples = int(SR * CHUNK_DURATION)
    num_chunks = len(audio) // chunk_samples
    
    # 存储结果
    results = []

    for i in range(num_chunks):
        chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]
        f1, f2 = analyze_vowel(chunk, sr)
        
        # 根据 F1 和 F2 判断元音
        if f1 > 2800:  # E 的示例条件
            results.append("检测到元音 E")
        elif f1 < 2800 and f1 > 2400:  # A 的示例条件
            results.append("检测到元音 A")
        elif f1 > 2000 and f1 < 2600:  # O 的示例条件
            results.append("检测到元音 O")
        elif f1 > 600 and f1 < 2000 and f2 < 300:  # U 的示例条件
            results.append("检测到元音 U")
        elif f1 > 600 and f1 < 1700:  # I 的示例条件
            results.append("检测到元音 I")
        else:
            results.append("未检测到明确的元音")

    return results

# 音高分析
def analyze_vowel(audio, sr):
    # 提取音高
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, S=librosa.stft(audio))
    
    # 选择最大音高的索引
    f1 = np.max(pitches, axis=0).mean()
    f2 = np.max(pitches, axis=1).mean()
    print(f"提取的 F1: {f1:.2f} Hz, F2: {f2:.2f} Hz")
    return f1, f2

# 主程序
if __name__ == "__main__":
    # 分析 WAV 文件中的元音
    results = analyze_chunks('test_audio.wav')
    for result in results:
        print(result)
