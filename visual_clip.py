import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("clip_model/clip_vit_base_patch16")
processor = CLIPProcessor.from_pretrained("clip_model/clip_vit_base_patch16")

# 加载图像
image_path = "D:/ascxk-eidxo_wps图片/butterfly.png"
image = Image.open(image_path)

# 输入文本
text_input = ["蝴蝶", "雪地"]

# 处理输入
inputs = processor(text=text_input, images=image, return_tensors="pt", padding=True)

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取 logits
logits_per_image = outputs.logits_per_image  # 图像与文本的相似度
probs = logits_per_image.softmax(dim=1)  # 计算概率

# 输出结果
print("Grounding Probabilities:")
for i, prob in enumerate(probs[0]):
    print(f"{text_input[i]}: {prob.item():.4f}")
