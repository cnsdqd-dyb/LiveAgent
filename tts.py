from fish_audio_sdk import Session, TTSRequest, Prosody
import io
import base64
from pydub import AudioSegment
from pydub.playback import play
import os
import threading
import queue
import time
import numpy as np
import librosa
from live_model import Live2DController
import json
# 设置参数
SR = 22050  # 采样率
CHUNK_DURATION = 0.1  # 每个块的时长（秒）

class AudioTTS:
    def __init__(self, api_key, live_control:Live2DController=None):
        # 设置代理
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        
        # 初始化会话
        self.session = Session(api_key)
        
        # 音频队列
        self.audio_queue = queue.Queue()
        
        # 控制标志
        self.is_running = True
        
        # 启动消费者线程
        self.consumer_thread = threading.Thread(target=self.audio_player, daemon=True)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
        self.live_control = live_control
        
        # 声音ID映射
        self.voice_ids = {
            "高冷御姐": "7f92f8afb8ec43bf81429cc1c9199cb1",
            "夹子音": "7af4d620be1c4c6686132f21940d51c5",
            "元气声": "1aacaeb1b840436391b835fd5513f4c4",
            "温柔声": "faccba1a8ac54016bcfc02761285e67f",
            "夹子可莉": "efc1ce3726a64bbc947d53a1465204aa",
            "可爱的女生": "0dce3af322a342db8b7644212f481918",
            "咄咄逼人": "0d70f9c5bd1947039b9cc10260bba689",
            "董宇辉": "c7cbda1c101c4ce8906c046f01eca1a2",
            "温柔男生": "d99547e2dad64ce0aa085319a3c9cc56",
            "蔡徐坤": "e4642e5edccd4d9ab61a69e82d4f8a14",
            "周杰伦": "1512d05841734931bf905d0520c272b1",
            "夯大力": "84ed22e0ec8746969adf08bef0407494",
            "黑手": "f7561ff309bd4040a59f1e600f4f4338"
        }

    def text_to_speech(self, text, voice_name, speed=1.0, play_audio=True, expression="[normal]", motion="<idle_stand>"):
        """
        生产者：将文本转换为音频数据并加入队列
        """
        try:
            # print(f"添加文本到队列: {text}")
            if voice_name not in self.voice_ids:
                raise ValueError(f"未知的声音名称: {voice_name}")
            
            # 设置语音参数
            prosody = Prosody()
            prosody.speed = speed
                
            # 收集音频数据
            audio_data = io.BytesIO()
            for chunk in self.session.tts(TTSRequest(
                text=text,
                reference_id=self.voice_ids[voice_name],
                opus_bitrate=64,
                prosody=prosody
            )):
                audio_data.write(chunk)
            data_pair = (audio_data, expression, motion)

            self.audio_queue.put(data_pair)
        except Exception as e:
            print(f"文本转语音错误: {str(e)}")
            return None

    def load_interupt(self):
        try:
            with open("interrupt.json", "r") as f:
                self.interrupt = json.load(f)["interrupt"]
        except:
            self.interrupt = False

    def audio_player(self):
        """
        消费者：从队列中获取并播放音频
        """
        while self.is_running:
            try:
                self.load_interupt()
            
                data_pair = self.audio_queue.get(timeout=0.1)
                if data_pair is None or self.interrupt:
                    time.sleep(.1)
                    continue
                audio_data, expression, motion = data_pair
                if expression and self.live_control and expression != "[normal]":
                    print(f"设置表情: {expression}")
                    self.live_control.set_expression(expression)

                if motion and self.live_control and motion != "<idle_stand>":
                    print(f"设置动作: {motion}")
                    self.live_control.set_motion(motion)
                elif self.live_control:
                    self.live_control.set_motion("<idle_stand>")
                # 播放音频
                audio_data.seek(0)
                audio = AudioSegment.from_mp3(audio_data)
                
                # 启动分析线程
                analysis_thread = threading.Thread(target=self.analyze_and_print_chunks, args=(audio_data,))
                analysis_thread.start()
                
                play(audio)
                
                # 等待分析线程完成
                analysis_thread.join()
                
                # 标记任务完成
                self.audio_queue.task_done()
                
            except queue.Empty:
                time.sleep(.1)
                continue
            except Exception as e:
                print(f"音频播放错误: {str(e)}")
                time.sleep(1)

    def analyze_and_print_chunks(self, audio_data):
        """
        分析音频块并打印结果
        """
        audio_data.seek(0)
        audio, sr = librosa.load(audio_data, sr=SR)
        results, volumes = self.analyze_chunks(audio, sr)
        results.append("")
        volumes.append(0)
        for result, volume in zip(results, volumes):
            if self.live_control:
                self.live_control.set_mouth(result, rate=volume)
            time.sleep(CHUNK_DURATION - 0.01)

    def stop(self):
        """
        停止播放并清理资源
        """
        self.is_running = False
        # 发送停止信号
        self.audio_queue.put(None)
        # 等待消费者线程结束
        self.consumer_thread.join(timeout=1)
        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    # 读取 WAV 文件并进行 chunk 处理
    def analyze_chunks(self, audio, sr):
        # 计算每个 chunk 的样本数
        chunk_samples = int(SR * CHUNK_DURATION)
        num_chunks = len(audio) // chunk_samples
        
        # 存储结果
        results = []
        volumes = []
        for i in range(num_chunks):
            chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]
            f1, f2, volume = self.analyze_vowel(chunk, sr)
            volumes.append(min(volume / 10, 1))
            # 根据 F1 和 F2 判断元音
            if f1 > 2800:  # E 的示例条件
                results.append("E")
            elif f1 < 2800 and f1 > 2400:  # A 的示例条件
                results.append("A")
            elif f1 > 2000 and f1 < 2600:  # O 的示例条件
                results.append("O")
            elif f1 > 600 and f1 < 2000 and f2 < 300:  # U 的示例条件
                results.append("U")
            elif f1 > 600 and f1 < 1700:  # I 的示例条件
                results.append("I")
            else:
                results.append("")
        results.append("")
        return results, volumes

    # 音高分析
    def analyze_vowel(self, audio, sr):
        # 提取音高
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, S=librosa.stft(audio))
        
        # 选择最大音高的索引
        f1 = np.max(pitches, axis=0).mean()
        f2 = np.max(pitches, axis=1).mean()
        # print(f"提取的 F1: {f1:.2f} Hz, F2: {f2:.2f} Hz")
        volume = np.linalg.norm(audio)
        return f1, f2, volume

# 使用示例
if __name__ == "__main__":
    tts = AudioTTS("40293f990bec4378957a155b477c59b1", live_control=Live2DController())
    
    # 测试多个连续播放
    texts = [
        "第一句话，你好！",
        "第二句话，今天天气真好。",
        "第三句话，再见！"
    ]
    
    # 添加多个文本进行测试
    for text in texts:
        tts.text_to_speech(
            text=text,
            voice_name="温柔声",
            speed=1.0,
            play_audio=True
        )
    
    # 等待所有音频播放完成
    tts.audio_queue.join()
    
    # 停止服务
    tts.stop()
