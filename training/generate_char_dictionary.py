# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import cv2
from PIL import ImageFont, Image
from tqdm import tqdm
from skimage.feature import hog
from config.settings import settings
from utils.text_drawing_utils import get_system_font_path, draw_centered_text, get_chinese_chars_from_unicode_range

# 配置保持一致
FONT_SIZE = 62
IMAGE_SIZE = (64, 65)
TARGET_SIZE = (64, 64) # 匹配 matcher.py 的处理尺寸

def extract_features_static(img_pil):
    """
    静态特征提取函数：不保存图片，直接转换特征
    """
    # 1. 预处理 (与 matcher.py 保持完全一致)
    if img_pil.mode == 'RGBA':
        # 创建一个纯白色背景的图像
        canvas = Image.new('RGBA', img_pil.size, (255, 255, 255, 255))
        # 将原图粘贴到白色背景上，使用原图的 alpha 通道作为掩码
        canvas.paste(img_pil, (0, 0), mask=img_pil)
        img_pil = canvas.convert('L')
    else:
        img_pil = img_pil.convert('L')
    img_gray = img_pil.resize(TARGET_SIZE, Image.LANCZOS)
    img_array = np.array(img_gray)

    # 2. 提取 HOG
    fd = hog(img_array, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), visualize=False, channel_axis=None)

    # 3. 提取 Projection
    _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
    h_proj = np.sum(binary, axis=1) / 255
    v_proj = np.sum(binary, axis=0) / 255
    h_proj = h_proj / (np.max(h_proj) + 1e-6)
    v_proj = v_proj / (np.max(v_proj) + 1e-6)
    proj = np.concatenate([h_proj, v_proj])

    return img_array, fd, proj

def generate_dictionary():
    """
    直接生成特征 pkl 文件，不再生成中间图片
    """
    font_path = get_system_font_path('SimHei')
    char_dict_full_path = os.path.join(settings.paths.base_dir, settings.paths.char_dict_path)
    output_dir = os.path.join(settings.paths.base_dir, settings.paths.dict_image_dir)
    cache_path = os.path.join(output_dir, "feature_cache.pkl")
    os.makedirs(output_dir, exist_ok=True)

    # 加载字符集逻辑...
    if os.path.exists(char_dict_full_path):
        with open(char_dict_full_path, 'r', encoding='utf-8') as f:
            characters = [line.strip() for line in f.readlines() if line.strip()]
    else:
        characters = get_chinese_chars_from_unicode_range(max_chars=1000)

    font = ImageFont.truetype(font_path, FONT_SIZE)
    dictionary_data = {}

    print(f"开始离线提取 {len(characters)} 个字符的特征...")
    pbar = tqdm(characters, desc="特征提取", unit="char")
    
    for char in pbar:
        pbar.set_postfix({"当前": char})
        
        # 1. 内存中绘制
        img_pil = draw_centered_text(char, font, image_size=IMAGE_SIZE)
        
        # 2. 直接提取特征
        img_array, fd, proj = extract_features_static(img_pil)
        
        # 3. 存入字典
        dictionary_data[char] = {
            'image_array': img_array,
            'hog_features': fd,
            'proj': proj
        }

    # 4. 一次性写入 pkl
    with open(cache_path, 'wb') as f:
        pickle.dump(dictionary_data, f)

    print(f"\n特征库已成功保存至: {cache_path}")
    print("现在你可以直接运行 ImageMatcher，无需生成图片文件。")

if __name__ == '__main__':
    generate_dictionary()