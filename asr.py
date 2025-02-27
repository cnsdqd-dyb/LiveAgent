import sounddevice as sd
import numpy as np
import time
from scipy.io.wavfile import write
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import threading
import os
class VoiceListener:
    def __init__(self, model_dir="asr_model/SenseVoiceSmall", device="cuda:0"):
        # 初始化参数
        self.volume_threshold = 0.01
        self.recording = False
        self.audio_buffer = []
        self.silence_frames = 0
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
        
        # 初始化模型
        self.model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            hub="hf",
            disable_update=True,
        )
        
        # 计算帧率相关参数
        self.frames_per_second = 16000 / 960  # 采样率/每帧样本数
        self.silence_threshold_frames = int(self.frames_per_second)  # 1秒对应的帧数
        self.text_buffer = []
        self.off = False
        self.threading_start_listening()

    def reset(self):
        print("重置...")
        self.off = False
        self.recording = False
        self.audio_buffer = []
        self.silence_frames = 0

    def audio_callback(self, indata, frames, time_, status):
        try:
            # print("音频...", self.off, self.recording)
            volume = np.linalg.norm(indata)
            
            # write json
            with open("audio.json", "w") as f:
                if self.off:
                    f.write('{"listening": false}')
                else:
                    f.write('{"listening": true}')

            if self.off:
                time.sleep(1)
                return None
            
            if volume > self.volume_threshold and not self.recording:
                print("开始录音...")
                self.recording = True
                self.audio_buffer = []
                self.silence_frames = 0
            
            if self.recording:
                if volume < self.volume_threshold:
                    self.silence_frames += 1
                    if self.silence_frames >= self.silence_threshold_frames:
                        # print("停止录音...")
                        self.recording = False
                        audio_data = np.concatenate(self.audio_buffer, axis=0)
                        return self.process_audio(audio_data)
                else:
                    self.silence_frames = 0
                
                self.audio_buffer.append(indata.copy())

        except Exception as e:
            print(f"音频处理错误: {str(e)}")
            return None

    def process_audio(self, audio_data):
        wav_file_path = "recorded_audio.wav"
        write(wav_file_path, 16000, audio_data)

        try:
            res = self.model.generate(
                input=wav_file_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=1,
                merge_vad=True,
                merge_length_s=15,
            )
            text = rich_transcription_postprocess(res[0]["text"]).replace("😡", "")
            if text.strip() != "" and len(text) > 5:
                self.text_buffer.append(text)
                self.off = True
                print(text)
            return text
        except Exception as e:
            self.text_buffer.append(f"语音识别失败：{str(e)}")
            return f"语音识别失败：{str(e)}"

    def start_listening(self):
        """开始监听音频"""
        with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=16000):
            print("开始监测音频...")
            while True:
                try:
                    time.sleep(.1)
                except KeyboardInterrupt:
                    break

    def stop_listening(self):
        """停止监听音频"""
        sd.stop()

    def threading_start_listening(self):
        """多线程开始监听音频"""
        self.asr_thread = threading.Thread(target=self.start_listening)
        self.asr_thread.daemon = True
        self.asr_thread.start()

    def threading_stop_listening(self):
        """多线程停止监听音频"""
        threading.Thread(target=self.stop_listening).start()

if __name__ == "__main__":
    # 创建实例
    voice_listener = VoiceListener(model_dir="asr_model/SenseVoiceSmall", device="cuda")
    while True:
        input()
        voice_listener.off = False

