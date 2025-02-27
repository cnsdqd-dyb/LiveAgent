import os
import pygame
from pygame.locals import *
import live2d.v3 as live2d
import json
import time
from typing import Tuple
import random
from live2d.v3 import StandardParams
import threading
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from multiprocessing import shared_memory


class Live2DController:
    def __init__(self, control_file: str = "live2d_control.json", control_mouth_file: str = "live2d_mouth.json"):
        """
        初始化Live2D控制器
        Args:
            control_file: 控制文件路径
        """
        self.control_file = control_file
        self.control_mouth_file = control_mouth_file
        self._ensure_control_file()

    def _ensure_control_file(self):
        """确保控制文件存在，不存在则创建"""
        if not os.path.exists(self.control_file):
            default_control = {
                "expression": "exp_08",
                "model_path": "v3/Mao/Mao.model3.json",
                "display_size": {
                    "width": 200,
                    "height": 200
                },
                "position": {
                    "x": 0.0,
                    "y": 0.0
                },
                "scale": 1.0,
                "motion": 0,
                "stop": False,
                "time": time.time(),
                "mouth_word": "",
                "mouth_rate": 1
            }
            self.update_control(default_control)

    def _read_control(self) -> dict:
        """读取当前控制文件内容"""
        try:
            with open(self.control_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取控制文件错误: {e}")
            return {}

    def update_mouth(self, data: dict):
        """更新嘴巴动作文件"""
        try:
            with open(self.control_mouth_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"更新嘴巴动作文件错误: {e}")

    def update_control(self, data: dict):
        """更新控制文件"""
        try:
            data["time"] = time.time()
            with open(self.control_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"更新控制文件错误: {e}")

    def set_mouth(self, word: str, rate: int = 1):
        """
        设置嘴巴动作
        Args:
            word: 嘴巴动作文字
            rate: 动作速率
        """
        control_data = self._read_control()
        control_data['mouth_word'] = word
        control_data['mouth_rate'] = rate
        self.update_mouth(control_data)

    def set_expression(self, expression: str):
        """
        设置表情
        Args:
            expression: "[angry]", "[sad]", "[surprise]", "[shy]", "[scared]", "[excited]", "[cozy]", "[happy]", "[normal]"
        """
        # "exp_01" - "exp_20" [angry], [shocked], [shy], [scared], [excited], [cozy], [happy], [normal], [thinking], [cute], [serious], [mad], [worry], [surprise], [pout], [sleepy], [curious], [proud], [left_wink], [right_wink]
        
        expression_map = {
            "[angry]": "exp_08",
            "[shocked]": "exp_07",
            "[shy]": "exp_06",
            "[scared]": "exp_05",
            "[excited]": "exp_04",
            "[cozy]": "exp_03",
            "[happy]": "exp_02",
            "[normal]": "exp_01",
            "[thinking]": "exp_09",
            "[cute]": "exp_10",
            "[serious]": "exp_11",
            "[mad]": "exp_12",
            "[worry]": "exp_13",
            "[surprise]": "exp_14",
            "[pout]": "exp_15",
            "[sleepy]": "exp_16",
            "[curious]": "exp_17",
            "[proud]": "exp_18",
            "[left_wink]": "exp_19",
            "[right_wink]": "exp_20",
        }   
        expression_id = expression_map.get(expression, "exp_01")
        control_data = self._read_control()
        control_data['expression'] = expression_id
        self.update_control(control_data)

    def set_motion(self, motion: str):
        """
        设置动作
        Args:
            motion: 动作类型 <idle_stand>, <idle_with_effect>, <hands_sway>, <hands_behind>, <one_fist_pump>, <shoulder_shrug>, <shoulder_shimmy>, <stretching>, <tilt_the_head>, <shake_head>, <nod_head>, <shy_head_tilt_smile>, <body_sway>, <surprised_then_shy>, <swagger>, <happy_body_shake>, <excitedly_stomp>, <shyly_stomp>, <look_down_around>, <headband_body_shake>, <look_left>, <left_shake_right_sway>, <left_shake_right_sway_hand>, <left_shake_right_sway_knee>, <quickly_shake_head>, <tilt_head_look>, <left_right_knee>, <slight_shake>, <slight_nod>, <serious_nod>, <tilt_head_shake>, <look_left_right>, <polite_bow>, <happy_bow>, <slight_shake_head>, <knee_frown_shake_head>, <open_eyes_tilt_head>, <cute_turn_head>, <knee_tilt_head>, <bow_shake>, <evade_look_around>, <smug_shake>, <happy_smile>, <left_tilt_head_to_right>, <bow_evade>, <surprised_tiptoe>
        """
        motion_map = {
            "<idle_stand>": 0,
            "<idle_with_effect>": 1, # 背后两个魔法小球
            "<hands_sway>": 2, # 两个手在两侧摆动
            "<hands_behind>": 3, # 双手放在背后
            "<one_fist_pump>": 4, # 一只手握拳,
            "<shoulder_shrug>": 5, # 耸肩
            "<shoulder_shimmy>": 6, # 肩部摇摆
            "<stretching>": 7, # 伸展
            "<tilt_the_head>": 8, # 点头
            "<shake_head>": 9, # 摇头
            "<nod_head>": 10, # 点头
            "<shy_head_tilt_smile>": 11, # 左右看
            "<body_sway>": 12, # 身体摇摆
            "<surprised_then_shy>": 13, # 惊讶后害羞
            "<swagger>": 14, # 大摇大摆
            "<happy_body_shake>": 15, # 开心摇摆
            "<excitedly_stomp>": 16, # 兴奋跺脚
            "<shyly_stomp>": 17, # 害羞跺脚
            "<look_down_around>": 18, # 低头看看
            "<headband_body_shake>": 19, # 头带摇摆
            "<look_left>": 20, # 左看
            "<left_shake_right_sway>": 21,  # 左摇右摆
            "<left_shake_right_sway_hand>": 22,  # 左摇右摆摆手
            "<left_shake_right_sway_knee>": 23,  # 左摇右摆屈膝
            "<quickly_shake_head>": 24,  # 快速摇头
            "<tilt_head_look>": 25,  # 歪头看
            "<left_right_knee>": 26,  # 左右屈膝
            "<slight_shake>": 27,  # 轻微晃动
            "<slight_nod>": 28,  # 轻轻点头
            "<serious_nod>": 29,  # 认真点头
            "<tilt_head_shake>": 30,  # 歪头摇摆
            "<look_left_right>": 31,  # 左看看右看看
            "<polite_bow>": 32,  # 礼貌的低头
            "<happy_bow>": 33,  # 开心的低头
            "<slight_shake_head>": 34,  # 轻轻摇头
            "<knee_frown_shake_head>": 35,  # 屈膝皱眉摇头
            "<open_eyes_tilt_head>": 36,  # 睁大眼睛歪头
            "<cute_turn_head>": 37,  # 可爱的来回扭头
            "<knee_tilt_head>": 38,  # 屈膝歪头
            "<bow_shake>": 39,  # 低头扭动
            "<evade_look_around>": 40,  # 逃避的四处看
            "<smug_shake>": 41,  # 得意的晃动
            "<happy_smile>": 42,  # 开心微笑
            "<left_tilt_head_to_right>": 43,  # 左歪头到右歪头
            "<bow_evade>": 44,  # 低头逃避
            "<surprised_tiptoe>": 45  # 惊讶踮起脚
        }
        motion_id = motion_map.get(motion, 0)
        control_data = self._read_control()
        control_data['motion'] = motion_id
        self.update_control(control_data)


    def set_position(self, x: float, y: float):
        """
        设置位置
        Args:
            x: x坐标 (-1.0 到 1.0)
            y: y坐标 (-1.0 到 1.0)
        """
        control_data = self._read_control()
        control_data['position'] = {
            'x': max(-1.0, min(1.0, x)),
            'y': max(-1.0, min(1.0, y))
        }
        self.update_control(control_data)

    def set_scale(self, scale: float):
        """
        设置缩放
        Args:
            scale: 缩放比例 (建议0.5到2.0)
        """
        control_data = self._read_control()
        control_data['scale'] = max(0.5, min(2.0, scale))
        self.update_control(control_data)


    def set_display_size(self, width: int, height: int):
        """
        设置显示窗口大小
        Args:
            width: 宽度
            height: 高度
        """
        control_data = self._read_control()
        control_data['display_size'] = {
            'width': width,
            'height': height
        }
        self.update_control(control_data)

    def set_model(self, model_path: str):
        """
        设置模型路径
        Args:
            model_path: 模型文件路径
        """
        control_data = self._read_control()
        control_data['model_path'] = model_path
        self.update_control(control_data)

    def stop_model(self):
        """停止模型显示"""
        control_data = self._read_control()
        control_data['stop'] = True
        self.update_control(control_data)

    def start_model(self):
        """启动模型显示"""
        control_data = self._read_control()
        control_data['stop'] = False
        self.update_control(control_data)

class Live2DModel:
    def __init__(self, display_size: Tuple[int, int] = (300, 400), 
                 model_path = "v3/Mao/Mao.model3.json",
                 control_file = "live2d_control.json",
                 face_file = "image_description.json",
                 mouth_file = "live2d_mouth.json"):

        self.display_size = display_size
        self.face_file = face_file
        self.running = False
        self.model = None
        self.current_expression = "exp_01"
        self.current_motion = 0
        self.dx = 0.0
        self.dy = 0.0
        self.scale = 1.0
        self.last_control_time = 0
        
        self.model_json = os.path.join("live2d-py/Resources/", model_path)
        self.control_file = control_file
        self.mouth_file = mouth_file
        self.running = True
        self.mouth_word = ""
        self.mouth_rate = 1

        self.expression_interval = 4
        self.fullscreen = False
        self.frame_path = "frame.png"

        self.no = 0

        # 创建默认控制文件
        if not os.path.exists(control_file):
            self._create_default_control_file()
        control_data = self._read_control_file()
        if control_data:
            if 'display_size' in control_data:
                self.display_size = (
                    control_data['display_size']['width'],
                    control_data['display_size']['height']
                )
            if 'model_path' in control_data:
                self.model_json = os.path.join("live2d-py/Resources/", control_data['model_path'])
        self._init_pygame()

    def _create_default_control_file(self):
        """创建默认的控制文件"""
        default_control = {
            "expression": "exp_01",
            "model_path": "v3/Mao/Mao.model3.json",
            "display_size": {
                "width": 400,
                "height": 400
            },
            "position": {
                "x": 0.0,
                "y": 0.0
            },
            "scale": 1.0,
            "motion": 0,
            "stop": False,
            "time": time.time(),
            "mouth_word": "",
            "mouth_rate": 1
        }
        with open(self.control_file, 'w', encoding='utf-8') as f:
            json.dump(default_control, f, indent=4)

    def _read_mouth_file(self):
        """读取嘴巴动作文件"""
        try:
            if os.path.exists(self.mouth_file):
                with open(self.mouth_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"读取嘴巴动作文件错误: {e}")
        return None
    
    def _read_face_file(self):
        """读取人脸识别文件"""
        try:
            if os.path.exists(self.face_file):
                with open(self.face_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"读取人脸识别文件错误: {e}")

    def _read_control_file(self):
        """读取控制文件"""
        try:
            if os.path.exists(self.control_file):
                with open(self.control_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"读取控制文件错误: {e}")
        return None

    def _init_pygame(self):
        pygame.init()
        pygame.mixer.init()
        live2d.init()
        # live2d.setLogEnable(True)
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0,0), FULLSCREEN | DOUBLEBUF | OPENGL | NOFRAME, vsync=1)
        else:
            self.screen = pygame.display.set_mode(self.display_size,  DOUBLEBUF | OPENGL | NOFRAME, vsync=1)
        pygame.display.set_caption("")

        if live2d.LIVE2D_VERSION == 3:
            live2d.glewInit()

        self.model = live2d.LAppModel()
        self.model.LoadModelJson(self.model_json)
        if not self.fullscreen:
            self.model.Resize(*self.display_size)
        # self.model.StartRandomMotion("Idle", 300, None, None)

        self.action_interval = 10
        self.last_action_time = time.time()
        self.last_motion_time = time.time()

        self.shm = None
        self.shared_frame = None

    def save_frame_(self):
        """保存当前帧"""
        try:
            screen = pygame.display.set_mode((self.display_size[0], self.display_size[1]), pygame.HIDDEN | pygame.DOUBLEBUF | pygame.OPENGL)
            size = screen.get_size()
            buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
            screen_surf = pygame.image.fromstring(buffer, size, "RGBA")
            # 上下翻转
            screen_surf = pygame.transform.flip(screen_surf, False, True)
            pygame.image.save(screen_surf, self.frame_path)

        except Exception as e:
            print(f"保存帧错误: {e}")

    def save_frame(self):
        """保存当前帧"""
        try:
            self.screen = pygame.display.set_mode((self.display_size[0], self.display_size[1]), pygame.HIDDEN | pygame.DOUBLEBUF | pygame.OPENGL)
            size = self.screen.get_size()
            buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
            screen_surf = pygame.image.fromstring(buffer, size, "BGRA")
            screen_surf = pygame.transform.flip(screen_surf, False, True)

            # pygame.image.save(screen_surf, self.frame_path)

            # 将 pygame surface 转换为 numpy 数组
            frame_array = pygame.surfarray.array3d(screen_surf)
            frame_array = frame_array.swapaxes(0, 1)

            if self.shm is None: 
                frame_shape = frame_array.shape
                # 计算正确的内存大小
                frame_size = frame_array.nbytes  # 使用 nbytes 获取准确的字节数
                # print(f"发送端 - Frame shape: {frame_shape}, Frame size: {frame_size} bytes")
                
                self.shm = shared_memory.SharedMemory(create=True, size=frame_size, name='frame_share')
                self.shared_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=self.shm.buf)
                self.last_time = time.time()

                # 将形状信息写入单独的共享内存
                shape_str = f"{frame_shape[0]},{frame_shape[1]},{frame_shape[2]}"
                self.shape_shm = shared_memory.SharedMemory(create=True, size=len(shape_str.encode()), name='frame_shape')
                self.shape_shm.buf[:len(shape_str.encode())] = shape_str.encode()

            # 写入新帧
            self.shared_frame[:] = frame_array[:]

        except Exception as e:
            print(f"保存帧错误: {e}")
            import traceback
            traceback.print_exc()

    def _update_from_control(self):
        """从控制文件更新模型状态"""
        current_time = time.time()

        # if int(current_time) % 10 == 0:
        #     print(f"NO: {self.no}")
        #     input("Press Enter to continue...")
        #     self.model.StartMotion(group = "Idle", no = self.no, priority = 3)
        #     self.no += 1
        # if int(current_time) % 3 == 0:
        #     self.model.SetRandomExpression()


        # 每100ms检查一次控制文件
        if current_time - self.last_control_time >= 0.1:
            control_data = self._read_control_file()
            mouth_data = self._read_mouth_file()
            face_data = self._read_face_file()
            
            if control_data:
                # 更新表情
                if 'expression' in control_data:
                    new_expression = control_data['expression']
                    if current_time - float(control_data['time']) >= self.expression_interval:
                        self.model.SetExpression("exp_01")
                        self.current_expression = "exp_01"
                    elif new_expression != self.current_expression:
                        self.model.SetExpression(new_expression)
                        self.current_expression = new_expression

                if 'motion' in control_data:
                    if control_data['motion'] != self.current_motion:
                        self.model.StartMotion(group = "Idle", no = control_data['motion'], priority = 1)
                        self.current_motion = control_data['motion']
                        self.last_motion_time = current_time

                # 更新位置
                if face_data:
                    if len(face_data['centers']) > 0:
                        center = random.choice(face_data['centers'])
                        # print(f"center: {center}")
                        self.dx = (center[0] - 0.5) * 0.01 
                        self.dy = (center[1] - 0.5) * 0.01
                        screen_x = -self.dx * 5000 + self.display_size[0] / 2
                        screen_y = self.dy * 5000 + self.display_size[1] / 2
                        # print(f"screen_x: {screen_x}, screen_y: {screen_y}")
                        self.model.Drag(screen_x, screen_y)

                # 更新缩放
                if 'scale' in control_data:
                    self.scale = control_data['scale']
                    # self.model.SetScale(self.scale)

                # 更新动作
                # if 'motion' in control_data:
                #     if current_time - self.last_action_time >= self.action_interval:
                #         self.last_action_time = current_time + random.randint(0, 3)
                        # self.model.StartRandomMotion("Idle", 3)
                        
                        # self.model.StartMotion(group = "Idle", no = 30, priority = 1)

                if 'stop' in control_data:
                    self.running = not control_data['stop']
                
                if mouth_data:
                    if 'mouth_word' in mouth_data:
                        self.mouth_word = mouth_data['mouth_word']
                    
                    if 'mouth_rate' in mouth_data:
                        self.mouth_rate = mouth_data['mouth_rate']

            self.last_control_time = current_time



    def loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

            if not self.running:
                break
            
            
            live2d.clearBuffer(14/255., 17/255., 23/255., 0)
            self.model.SetScale(self.scale)
            self.model.Update()
            if int(time.time() * 300) % 10 == 0:
                self.save_frame()
            self._update_from_control()
            if self.mouth_word:
                self.model.SetParameterValue(f"Param{self.mouth_word}", self.mouth_rate, 1)
            self.model.Draw()
            
            pygame.display.flip()
            pygame.time.wait(10)

            
        
        live2d.dispose()
        pygame.quit()
        quit()


if __name__ == "__main__":
    Live2DModel(display_size=(600, 800)).loop()
