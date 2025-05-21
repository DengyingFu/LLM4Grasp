
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from groundingdino.util.inference import load_model, load_image, predict, my_annotate
import base64
import io
from typing import Tuple
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
import cv2


app = Flask(__name__)

def image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    将 OpenCV 处理的 NumPy 图像转换为 Base64 编码字符串
    :param image: NumPy 格式的 BGR 图像
    :param format: 图像格式（默认 "JPEG"，也可为 "PNG"）
    :return: Base64 字符串
    """
    # 1. OpenCV (BGR) 转换为 PIL (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # 2. 将 PIL Image 转换为字节流
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)  # 以 format 格式（JPEG/PNG）保存

    # 3. Base64 编码
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_str
def load_image_from_base64(image_base64: str) -> Tuple[np.array, torch.Tensor]:
    """
    从 Base64 字符串加载图像，并进行预处理
    :param image_base64: Base64 编码的图像字符串
    :return: (原始 NumPy 图像, 预处理后的 PyTorch Tensor)
    """
    # 1. 解码 Base64 并转换为 PIL.Image
    image_data = base64.b64decode(image_base64)  # 解码 Base64
    image_source = Image.open(io.BytesIO(image_data)).convert("RGB")  # 转换为 RGB 图像

    # 2. 转换为 NumPy 数组（未归一化）
    image = np.asarray(image_source)

    # 3. 定义 PyTorch 变换（预处理）
    transform = T.Compose([
        T.Resize((800, 800)),  # 统一调整大小
        T.ToTensor(),  # 转换为 PyTorch 张量
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化（ImageNet 标准）
    ])

    # 4. 应用变换
    image_transformed = transform(image_source)

    return image, image_transformed

model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./groundingdino_swint_ogc.pth")
# model = YOLOWorld("yolov8x-worldv2.pt")
# model = YOLOWorld("/home/dyfu/Project/Genesis/extend-G/yolo_world/")
# Define custom classes

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    image_data = data.get('image')
    classes = data.get('classes')
    print('classes:', classes)
    # model.set_classes(classes)   #设置要检测的内别名称
    if not image_data or not classes:
        return jsonify({'error': 'Missing image or classes'})

    # # 解码 Base64 图片数据
    # image_data = base64.b64decode(image_data)
    # image = Image.open(io.BytesIO(image_data))
    image_source, image  = load_image_from_base64(image_data)
    h, w, _ = image_source.shape
    print(h,w)
    # 使用模型进行预测
    # results = model.predict(source=image, conf=0.05)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=classes,
        box_threshold=0.2,
        text_threshold=0.2
    )
    annotated_frame,detection = my_annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("tmp.jpg", annotated_frame)
    print(boxes,'\n',logits,'\n',phrases)
    print(detection)
    detection = detection.xyxy  # 取出内部的 numpy array
    detection = detection.astype(object)
    detection = np.column_stack((detection, np.array(logits),phrases))
    print(detection)

    base64_image = image_to_base64(annotated_frame)
   
    # 准备请求数据
    data = {
        'image': base64_image,
        'Detections': detection.tolist()
    }

    return jsonify(data)

if __name__ == '__main__':
    app.run(host='127.0.0.1') #本地使用
    #app.run(host='0.0.0.0', port=5000) #作为服务器供你本地连接
