import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

class ImageCaptioner:
    def __init__(self, model_path="blip_model/small", device="cuda"):
        """
        初始化图像描述生成器
        
        Args:
            model_path (str): 模型路径
            device (str): 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16
        ).to(device)
        self.interval = 10
        self.image_old = None
        self.current_caption = ""

    def similar_check(self, image_new, threshold=0.1):

        if self.image_old is None:
            self.image_old = image_new
            return False
        else:
            diff = np.array(image_new) - np.array(self.image_old)
            # 归一化
            diff = diff / 255.0
            diff = np.mean(diff)
            if diff < threshold:
                return True
            else:
                self.image_old = image_new
                return False

    def generate_caption(self, image, conditional_text=None):
        """
        生成图像描述
        
        Args:
            image: PIL.Image 对象或图片URL字符串
            conditional_text (str, optional): 条件文本提示
        
        Returns:
            str: 生成的图像描述
        """
        # 本地图片路径
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                raise Exception(f"无法加载图片: {e}")
        # 如果输入是URL，下载并转换为PIL Image
        if isinstance(image, str):
            try:
                image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
            except Exception as e:
                raise Exception(f"无法从URL加载图片: {e}")
        
        # 确保图片是PIL Image格式
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL Image对象或图片URL")

        # 处理输入
        if conditional_text:
            inputs = self.processor(image, conditional_text, return_tensors="pt").to(self.device, torch.float16)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        # 生成描述
        out = self.model.generate(**inputs)
        
        # 解码输出
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        self.current_caption = caption
        return caption

if __name__ == "__main__":
    # 初始化
    captioner = ImageCaptioner(model_path="blip_model/small")

    # # 使用URL
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    # caption1 = captioner.generate_caption(img_url)
    # print("无条件生成:", caption1)

    # # 使用条件文本
    # caption2 = captioner.generate_caption(img_url, conditional_text="a photography of")
    # print("条件生成:", caption2)

    # local_image = Image.open("D:/我的文件/v_llm_live2d_tts/image.jpg")
    # caption3 = captioner.generate_caption(local_image, conditional_text="The detailed description of the picture in house:")
    # print("本地图片:", caption3)

