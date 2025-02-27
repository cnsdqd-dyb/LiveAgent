from image_caption import ImageCaptioner
from face_detect import FaceDetector
import time
from PIL import Image
import cv2
import threading
import json
import os

class VideoAnalyzer:
    def __init__(self, model_path="blip_model/small"):
        """
        初始化视频分析器
        Args:
            model_path (str): 图像描述模型的路径
        """
        self.captioner = None
        self.face = None
        self.init_thread = threading.Thread(target=self.init_face_captioner, daemon=True)
        self.init_thread.start()
        self.is_running = False
        self.thread = None
        self.video_capture = None

        self.start()

    def init_face_captioner(self):
        self.face = FaceDetector()
        self.captioner = ImageCaptioner()
        
    def start(self, camera_index=0):
        """
        启动视频分析守护线程
        Args:
            camera_index (int): 摄像头索引，默认为0
        """
        if self.is_running:
            print("视频分析器已经在运行")
            return
            
        self.is_running = True
        self.video_capture = cv2.VideoCapture(camera_index)
        
        # 创建并启动守护线程
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("视频分析器已启动")
        
    def stop(self):
        """
        停止视频分析
        """
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.video_capture:
            self.video_capture.release()
        print("视频分析器已停止")
        
    def _run(self):
        """
        视频分析的主循环
        """
        video_count = 0

        while self.captioner is None or self.face is None:
            time.sleep(1)
        
        while self.is_running:
            ret, frame = self.video_capture.read()
            if not ret:
                print("无法获取视频帧")
                break
                
            video_count += 1
            
            if video_count % self.captioner.interval == 0:
                if not self.captioner.similar_check(frame):
                    try:
                        # 转换帧为PIL图像
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        
                        # 生成图像描述
                        caption = self.captioner.generate_caption(
                            image, 
                            "a meeting photography of"
                        )
                        caption = caption.replace("a meeting photography of", "")
                        
                        # 检测人脸
                        faces, centers = self.face.detect_faces(frame)
                        
                        # 保存结果到JSON文件
                        self._save_results(caption, faces, centers)
                        
                    except Exception as e:
                        print(f"处理帧时发生错误: {str(e)}")
                        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def _save_results(self, caption, faces, centers):
        """
        保存分析结果到JSON文件
        """
        try:
            
            if os.path.exists("image_description.json"):
                with open("image_description.json", "r", encoding='utf-8') as f:
                    raw_results = json.load(f)

            if len(faces) == 0:
                faces = raw_results.get("faces", [])
                centers = raw_results.get("centers", [])
            
            results = {
                "caption": caption,
                "faces": faces,
                "centers": centers
            }
            with open("image_description.json", "w", encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存结果时发生错误: {str(e)}")
            
    def is_active(self):
        """
        检查分析器是否正在运行
        """
        return self.is_running and self.thread and self.thread.is_alive()

if __name__ == "__main__":
    analyzer = VideoAnalyzer()
    analyzer.start()
    
    try:
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        analyzer.stop()
        print("视频分析器已停止")