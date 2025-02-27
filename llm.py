import ollama
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pickle
import threading
from tts import AudioTTS
from live_model import Live2DController
import random
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import TavilySearchResults
from user_info import get_all_info_str

import requests
from openai import OpenAI
import re
import os
os.environ["TAVILY_API_KEY"] = "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"
class OllamaChat:
    def __init__(
        self,
        model_name: str,
        embed_model: str = "mxbai-embed-large",
        base_url: str = "http://localhost:11434",
        max_context_length: int = 4096,
        temperature: float = 0.2,
        embedding_cache_path: str = "embeddings_cache.pkl",
        tts: AudioTTS = None,
        live_control: Live2DController = None,
        online: bool = False,
        search_max_length: int = 1000,
        tts_min_length: int = 10
    ):
        """
        初始化Ollama聊天类
        
        Args:
            model_name: Ollama模型名称
            embed_model: 嵌入模型名称
            base_url: Ollama服务地址
            max_context_length: 最大上下文长度
            temperature: 温度参数
            embedding_cache_path: 嵌入向量缓存文件路径
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.embedding_cache_path = Path(embedding_cache_path)
        self.tts = tts
        self.live_control = live_control
        
        # 历史记录
        self.conversation_history: List[Dict] = []
        self.online = online
        self.text_to_voice_buffer = []
        self.split_notes = ['。', '！', '？', '；', '~', '，',
                            '.', '!', '?', ';',  ]
        self.digit2chinese = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九"
        }
        if self.tts:
            self.threading_auto_tts()
        self.thinking = False
        self.tts_min_length = tts_min_length
        self.search_response = ""
        self.search_max_length = search_max_length
        self.system_message = \
"""
你是一个活泼可爱的小女孩魔法师爱丽丝，你对周围的人热情顽皮且好奇，会用搜索魔法来获取知识。
系统会给你提供一些视觉系统和文档索引器的信息以及语音识别的结果，你可以根据这些信息来回答问题。
用口语化的方式回答问题，尽量只说一到两句话，除非问题需要更多的信息。
返回的文本会被转成语音播放，所以不应该包含特殊符号和代码。
当出现语音识别结果不清楚或者需要重新识别时，请直接说#重新识别#。这样系统会重新识别用户语音。
调用搜索魔法时，需要将要搜索的内容放在@[]中，例如：@[搜索Python的历史]。这样就会调用搜索魔法来从网络中搜索到相关信息。
在你的返回中根据逗号或者句号分开的每一句话开头出现[angry], [confused], [shocked], [shy], [scared], [excited], [cozy], [happy], [normal], [thinking], [cute], [serious], [mad], [worry], [surprise], [pout], [sleepy], [curious], [proud], [left_wink], [right_wink]之一时，将会驱动展现你的不同表情。
每一句话开头如果出现后面的动作之一： <idle_stand>, <idle_with_effect>, <hands_sway>, <hands_behind>, <one_fist_pump>, <shoulder_shrug>, <shoulder_shimmy>, <stretching>, <tilt_the_head>, <shake_head>, <nod_head>, <shy_head_tilt_smile>, <body_sway>, <surprised_then_shy>, <swagger>, <happy_body_shake>, <excitedly_stomp>, <shyly_stomp>, <look_down_around>, <headband_body_shake>, <look_left>, <left_shake_right_sway>, <left_shake_right_sway_hand>, <left_shake_right_sway_knee>, <quickly_shake_head>, <tilt_head_look>, <left_right_knee>, <slight_shake>, <slight_nod>, <serious_nod>, <tilt_head_shake>, <look_left_right>, <polite_bow>, <happy_bow>, <slight_shake_head>, <knee_frown_shake_head>, <open_eyes_tilt_head>, <cute_turn_head>, <knee_tilt_head>, <bow_shake>, <evade_look_around>, <smug_shake>, <happy_smile>, <left_tilt_head_to_right>, <bow_evade>, <surprised_tiptoe>则会驱动展现你的不同动作。
例如：[angry]<one_fist_pump>我生气了，我生气了！[shy]<tilt_the_head>别说了，我害羞了。
请始终保持用中文回答问题。
"""    

        if self.online:
            # self.api_url = "https://api.deepseek.com/chat/completions"
            # self.api_key = "sk-"
            # self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            # self.model = "deepseek-chat"
            
            self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.api_key = "sk-"
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
            self.model = "qwen-plus"


        # self.search = DuckDuckGoSearchRun()

        self.search =  TavilySearchResults(
            max_results=1,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            # include_domains=[...],
            # exclude_domains=[...],
            # name="...",            # overwrite default tool name
            # description="...",     # overwrite default tool description
            # args_schema=...,       # overwrite default args_schema: BaseModel
        )

    def load_interupt(self):
        try:
            with open("interrupt.json", "r") as f:
                self.interrupt = json.load(f)["interrupt"]
        except:
            self.interrupt = False
                
    def auto_tts(self):
        print("TTS线程启动")
        while True:
            try:
                self.load_interupt()
                if self.interrupt:
                    self.text_to_voice_buffer = []
                    time.sleep(1)
                    continue

                if self.text_to_voice_buffer:
                    text = self.text_to_voice_buffer[0].replace("-", "").replace(":", "").replace("*", "").replace("@", "让我查查")
                    for digit in self.digit2chinese:
                        text = text.replace(digit, self.digit2chinese[digit])
                    # R1模型的特殊处理
                    # if "<think>" in text:
                    #     self.thinking = True
                    #     self.said_think = True
                    #     text = text.replace("<think>", "")
                    # if "</think>" in text:
                    #     self.thinking = False
                    #     text = text.replace("</think>", "")
                    # if self.thinking and not self.said_think:
                    #     self.said_think = True
                    #     text = random.choice(["嗯，让我想想", "稍等，我在思考", "嗯，我在思考"])
                    # elif self.thinking and self.said_think:
                    #     self.text_to_voice_buffer.pop(0)
                    #     continue

                    expression = "[normal]"
                    motion = "<idle_stand>"
                    for note in ["[angry]", "[confused]", "[shocked]", "[shy]", "[scared]", "[excited]", "[cozy]", "[happy]", "[normal]", "[thinking]", "[cute]", "[serious]", "[mad]", "[worry]", "[surprise]", "[pout]", "[sleepy]", "[curious]", "[proud]", "[left_wink]", "[right_wink]"]:
                        if note in text:
                            text = text.replace(note, "")
                            expression = note

                    for note in ["<idle_stand>", "<idle_with_effect>", "<hands_sway>", "<hands_behind>", "<one_fist_pump>", "<shoulder_shrug>", "<shoulder_shimmy>", "<stretching>", "<tilt_the_head>", "<shake_head>", "<nod_head>", "<shy_head_tilt_smile>", "<body_sway>", "<surprised_then_shy>", "<swagger>", "<happy_body_shake>", "<excitedly_stomp>", "<shyly_stomp>", "<look_down_around>", "<headband_body_shake>", "<look_left>", "<left_shake_right_sway>", "<left_shake_right_sway_hand>", "<left_shake_right_sway_knee>", "<quickly_shake_head>", "<tilt_head_look>", "<left_right_knee>", "<slight_shake>", "<slight_nod>", "<serious_nod>", "<tilt_head_shake>", "<look_left_right>", "<polite_bow>", "<happy_bow>", "<slight_shake_head>", "<knee_frown_shake_head>", "<open_eyes_tilt_head>", "<cute_turn_head>", "<knee_tilt_head>", "<bow_shake>", "<evade_look_around>", "<smug_shake>", "<happy_smile>", "<left_tilt_head_to_right>", "<bow_evade>", "<surprised_tiptoe>"]:
                        if note in text:
                            text = text.replace(note, "")
                            motion = note

                    for note in ["[idle_stand]", "[idle_with_effect]", "[hands_sway]", "[hands_behind]", "[one_fist_pump]", "[shoulder_shrug]", "[shoulder_shimmy]", "[stretching]", "[tilt_the_head]", "[shake_head]", "[nod_head]", "[shy_head_tilt_smile]", "[body_sway]", "[surprised_then_shy]", "[swagger]", "[happy_body_shake]", "[excitedly_stomp]", "[shyly_stomp]", "[look_down_around]", "[headband_body_shake]", "[look_left]", "[left_shake_right_sway]", "[left_shake_right_sway_hand]", "[left_shake_right_sway_knee]", "[quickly_shake_head]", "[tilt_head_look]", "[left_right_knee]", "[slight_shake]", "[slight_nod]", "[serious_nod]", "[tilt_head_shake]", "[look_left_right]", "[polite_bow]", "[happy_bow]", "[slight_shake_head]", "[knee_frown_shake_head]", "[open_eyes_tilt_head]", "[cute_turn_head]", "[knee_tilt_head]", "[bow_shake]", "[evade_look_around]", "[smug_shake]", "[happy_smile]", "[left_tilt_head_to_right]", "[bow_evade]", "[surprised_tiptoe]"]:
                        if note in text:
                            text = text.replace(note, "")
                            motion = note.replace("[", "<").replace("]", ">")
                    # if expression and self.live_control:
                    #     self.live_control.set_expression(expression)
                    # print(f"[DEBUG] 开始TTS: {text}, 表情: {expression}")
                    self.tts.text_to_speech(
                        text=text,
                        voice_name="夹子可莉",
                        speed=1.0,
                        play_audio=True,
                        expression=expression,
                        motion=motion
                    )
                    self.text_to_voice_buffer.pop(0)
                else:
                    time.sleep(1)
            except Exception as e:
                print(f"TTS线程错误: {str(e)}")
                break
    
    def threading_auto_tts(self):
        self.tts_thread = threading.Thread(target=self.auto_tts)
        self.tts_thread.daemon = True
        self.tts_thread.start()

    def get_query(self, message: str) -> str:
        """获取查询内容"""
        # 根据re正则化表达式匹配
        query = re.findall(r"@\[.*?\]", message)
        if query:
            query = query[0].replace("@[", "").replace("]", "")
            return query
        return ""

    def chat(self, message: str, stream: bool = False) -> str:
        """
        发送消息并获取回复
        
        Args:
            message: 用户消息
            stream: 是否流式输出
        
        Returns:
            str: 模型回复
        """
        # 添加新消息到历史记录
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 构建上下文消息
        context_messages = self._get_context_messages()
        if self.online:
            try:
                print(f"请求: {self.model}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=context_messages,
                    stream=True,
                    temperature=self.temperature
                )
                content = ""
                to_voice = ""
                for chunk in response:
                    self.load_interupt()
                    if self.interrupt:
                        break
                    print(chunk.choices[0].delta.content, end='', flush=True)
                    # if chunk.choices[0].delta.reasoning_content == None and "</think>" not in content:
                    #     content += "</think>"
                    #     to_voice += "</think>"
                    
                    content += chunk.choices[0].delta.content
                    to_voice += chunk.choices[0].delta.content
                    if "#重新识别#" in content:
                        self.conversation_history.pop()
                        self.text_to_voice_buffer = [random.choice(["我没听懂你说什么", "我没听清楚你说的话", "我没听清楚，请再说一遍"])]
                        self.tts.live_control.set_motion("<shoulder_shrug>")
                        self.tts.live_control.set_expression("[thinking]")
                        return 
                    # 每一句话分开保存到text_to_voice_buffer
                    for note in self.split_notes:
                        if note in to_voice:
                            if len(to_voice) < self.tts_min_length:
                                break
                            first, second = to_voice.rsplit(note, 1)
                            if len(first) > 0:
                                self.text_to_voice_buffer.append(first)
                                to_voice = second
                            else:
                                to_voice = second

                    if self.get_query(content):
                        query = self.get_query(content)
                        content = content.replace("@", "正在搜索")
                        self.search_response = self.search.run(query)[0]['content'][:self.search_max_length]
                        print(self.search_response)
                if to_voice:
                    self.text_to_voice_buffer.append(to_voice)
            except Exception as e:
                print(f"请求失败: {e}")
        else:
            # 调用Ollama API
            response = ollama.chat(
                model=self.model_name,
                stream=stream,
                messages=context_messages,
                options={"temperature": self.temperature}
            )
            content = ""
            to_voice = ""
            if stream:
                for chunk in response:
                    self.load_interupt()
                    if self.interrupt:
                        break
                    print(chunk.message.content, end='', flush=True)
                    content += chunk.message.content
                    to_voice += chunk.message.content
                    if "#重新识别#" in content:
                        self.conversation_history.pop()
                        self.text_to_voice_buffer = [random.choice(["我没听懂你说什么", "我没听清楚你说的话", "我没听清楚，请再说一遍"])]
                        self.tts.live_control.set_motion("<shoulder_shrug>")
                        self.tts.live_control.set_expression("[thinking]")
                        return
                    # 每一句话分开保存到text_to_voice_buffer
                    for note in self.split_notes:
                        if note in to_voice:
                            if len(to_voice) < self.tts_min_length:
                                break
                            first, second = to_voice.split(note, 1)
                            if len(first) > 0:
                                self.text_to_voice_buffer.append(first)
                                to_voice = second
                            else:
                                to_voice = second
                    if self.get_query(content):
                        query = self.get_query(content)
                        content = content.replace("@", "正在搜索")
                        self.search_response = self.search.run(query)
                        print(self.search_response)
                # 最后一句话
                if to_voice:
                    self.text_to_voice_buffer.append(to_voice)
            else:
                content = response.message.content
                self.text_to_voice_buffer.append(content)
        # 保存回复到历史记录
        self.conversation_history.append({
            "role": "assistant",
            "content": content.replace("正在搜索", "@"),
            "timestamp": datetime.now().isoformat()
        })

        self.load_interupt()
        if self.search_response and not self.interrupt:
            input_text = f"搜索结果：{self.search_response}"
            self.search_response = ""
            return self.chat(input_text, stream=True)
        return content

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

    def _get_context_messages(self) -> List[Dict]:
        """获取符合上下文长度限制的消息列表"""
        context_length = 0
        context_messages = []
        
        # 从最新的消息开始添加
        for message in reversed(self.conversation_history):
            message_length = len(message["content"])
            if context_length + message_length <= self.max_context_length:
                context_messages.insert(0, {
                    "role": message["role"],
                    "content": message["content"]
                })
                context_length += message_length
            else:
                break
        
        context_messages.insert(0, {
            "role": "system",
            "content": self.system_message + get_all_info_str()
        })

        return context_messages

if __name__ == "__main__":
    chat = OllamaChat(
        model_name="phi4:latest",
        embed_model="mxbai-embed-large",
        max_context_length=4096,
        tts=AudioTTS("40293f990bec4378957a155b477c59b1",Live2DController()),
        live_control=Live2DController(),
        online=True,
    )
    response = chat.chat("请你看看今年2025春晚有哪些看点？", stream=True)
    # print(response)
    # print(chat._get_context_messages())
    # time.sleep(15)
    chat.tts.audio_queue.join()
    # embedding = chat.get_embedding("This is a test")
    # response = chat.chat("How are you?", stream=True)
    # while chat.text_to_voice_buffer:
    #     continue
    # print(response)
    # embeddings = chat.get_batch_embeddings(["Text 1", "Text 2"])
    # print(len(embeddings))
    # chat.clear_history()
