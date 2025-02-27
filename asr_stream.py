import pyaudio
import wave
import numpy as np
from funasr import AutoModel
import threading
import queue
import time
import collections

class AudioProcessor:
    def __init__(self):
        # ASR 模型初始化
        self.model = AutoModel(model="asr_model/paraformer_zh_streaming")
        
        # 音频参数
        self.CHUNK = 960  # 每个缓冲区的帧数
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.chunk_size = [0, 16, 8]  # 600ms
        self.encoder_chunk_look_back = 8
        self.decoder_chunk_look_back = 2
        
        # 语音检测参数
        self.ENERGY_THRESHOLD = 0.1  # 能量阈值，根据实际情况调整
        self.SILENCE_LIMIT = 1  # 静音判断秒数
        
        # 状态控制
        self.is_recording = False
        self.is_listening = False
        self.trigger_word = "你好助手"
        self.end_word = "再见"
        
        # 音频数据队列
        self.audio_queue = queue.Queue()
        self.text_buffer = []
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        
    def calculate_energy(self, audio_chunk):
        """计算音频片段的能量"""
        return np.mean(np.abs(audio_chunk))
        
    def start_listening(self):
        """开始监听音频输入"""
        self.is_listening = True
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()
        self.cache = {}
        print("开始监听... 说出触发词「{}」开始对话".format(self.trigger_word))
        
    def _process_audio(self):
        """音频处理主循环"""
        ring_buffer = collections.deque(maxlen=int(self.RATE * 0.5))  # 0.5秒的缓冲区
        
        while self.is_listening:
            try:
                audio_chunk = np.frombuffer(
                    self.stream.read(self.CHUNK), 
                    dtype=np.float32
                )
                
                # 计算能量
                energy = self.calculate_energy(audio_chunk)
                
                # 能量超过阈值，认为是语音
                if energy > self.ENERGY_THRESHOLD:
                    ring_buffer.extend(audio_chunk)
                    
                    if len(ring_buffer) == ring_buffer.maxlen:
                        self._process_speech_chunk(np.array(ring_buffer))
                        ring_buffer.clear()
                
                        
            except Exception as e:
                print(f"Error in audio processing: {e}")
                input()
                
    def _process_speech_chunk(self, audio_chunk):
        """处理检测到的语音片段"""
        try:
            # 这里的cache应该复用之前的 而不是每次都初始化
            res = self.model.generate(
                input=audio_chunk,
                cache=self.cache,
                is_final=True,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back
            )
            
            text = res[0]["text"]
            print(f"实时识别结果: {text}")
            if text.strip():
                if not self.is_recording:
                    if self.trigger_word in text:
                        print("\n检测到触发词！开始记录对话...")
                        self.is_recording = True
                        self.text_buffer = []
                else:
                    if self.end_word in text:
                        print("\n检测到结束词！停止记录对话...")
                        self.is_recording = False
                        full_text = "".join(self.text_buffer)
                        self._send_to_llm(full_text)
                        self.text_buffer = []
                    else:
                        print(f"识别结果: {text}")
                        self.text_buffer.append(text)
                        
        except Exception as e:
            print(f"Error in speech processing: {e}")
            
    def _send_to_llm(self, text):
        """发送文本到LLM API"""
        try:
            # 这里替换成你的LLM API调用
            print(f"\n发送到LLM的完整文本: {text}")
            # 示例API调用
            # response = requests.post(
            #     "YOUR_LLM_API_ENDPOINT",
            #     json={"text": text},
            #     headers={"Authorization": "YOUR_API_KEY"}
            # )
            # print(f"LLM响应: {response.json()}")
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        print("停止监听")

def main():
    processor = AudioProcessor()
    try:
        processor.start_listening()
        # 保持主程序运行
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n正在停止程序...")
        processor.stop_listening()

if __name__ == "__main__":
    main()
