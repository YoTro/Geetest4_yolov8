"""
图像处理工具函数
"""
import logging
import requests
from typing import Tuple, Optional, Union
from PIL import Image
import io
import base64
import numpy as np
import cv2

def download_image(session: requests.Session, url: str) -> Optional[Image.Image]:
    """
    下载图片并转换为PIL Image。
    """
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content))
            return remove_transparency(img)
    except Exception as e:
        logging.getLogger(__name__).error(f"下载图片失败: {url}, 错误: {e}")
    return None

def remove_transparency(img: Image.Image, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    移除图片透明度，将透明背景替换为指定颜色
    """
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, bg_color)
        background.paste(img, (0, 0), img.convert('RGBA'))
        return background
    return img.convert('RGB')

def resize_with_padding(img: Image.Image, target_size: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    保持宽高比缩放图像，并用灰色填充。返回处理后的图像、缩放比例和填充尺寸。
    符合YOLOv5/v8的letterbox预处理方式。
    """
    target_w, target_h = target_size
    img_w, img_h = img.size
    
    ratio = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * ratio), int(img_h * ratio)
    
    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    padded_img = Image.new('RGB', (target_w, target_h), color)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))
    
    return np.array(padded_img), (ratio, ratio), (pad_x, pad_y)

def crop(img: Union[Image.Image, np.ndarray], area_coords: Tuple[int, int, int, int]) -> Image.Image:
    """
    裁剪图像区域
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.crop(area_coords)

def to_grayscale(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """转换为灰度图"""
    if isinstance(img, Image.Image):
        return np.array(img.convert('L'))
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def apply_adaptive_threshold(gray_img: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    应用自适应高斯阈值
    """
    return cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )

def apply_otsu_threshold(gray_img: np.ndarray) -> np.ndarray:
    """
    应用Otsu阈值
    """
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def apply_median_blur(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    应用中值滤波去噪
    """
    return cv2.medianBlur(img, kernel_size)

def image_to_base64(img: Image.Image, format: str = 'PNG') -> str:
    """
    将PIL图像转换为base64字符串
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_str: str) -> Image.Image:
    """
    将base64字符串转换为PIL图像
    """
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data)).convert('RGB')
