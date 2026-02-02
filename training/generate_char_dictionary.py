# -*- coding: utf-8 -*-
import os
from PIL import ImageFont
from tqdm import tqdm  
from config.settings import settings
from utils.text_drawing_utils import get_system_font_path, draw_centered_text, get_chinese_chars_from_unicode_range

# --- 配置 (从 settings 对象中获取) ---
FONT_SIZE = 62
IMAGE_SIZE = (64, 65)

def generate_dictionary():
    """
    根据字符集生成字典图片。
    如果字符集文件不存在，则自动生成一个默认的字符集作为后备。
    """
    # 1. 获取字体路径
    font_path = get_system_font_path('YuGothB')
    if not font_path:
        print("错误：未在系统中找到 'YuGothB' 字体。请确保已安装。")
        return

    # 2. 获取字符集文件路径和输出目录
    char_dict_full_path = os.path.join(settings.paths.base_dir, settings.paths.char_dict_path)
    output_full_path = os.path.join(settings.paths.base_dir, settings.paths.dict_image_dir)
    os.makedirs(output_full_path, exist_ok=True)
    
    # 3. 读取字符 (带后备逻辑)
    if os.path.exists(char_dict_full_path):
        print(f"从 '{char_dict_full_path}' 文件中读取字符集...")
        with open(char_dict_full_path, 'r', encoding='utf-8') as f:
            characters = [line.strip() for line in f.readlines() if line.strip()]
    else:
        print(f"警告：字符集文件 '{char_dict_full_path}' 未找到。")
        print("将自动生成一个默认的常用汉字集作为后备。")
        characters = get_chinese_chars_from_unicode_range(max_chars=1000) 
        try:
            with open(char_dict_full_path, 'w', encoding='utf-8') as f:
                for char in characters:
                    f.write(f"{char}\n")
            print(f"成功创建字符集文件: {char_dict_full_path}")
        except Exception as e:
            print(f"错误：保存字符集文件失败: {e}")

    if not characters:
        print("错误：字符集为空，无法生成字典。")
        return

    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except IOError:
        print(f"错误：无法加载字体文件 '{font_path}'。")
        return

    print(f"开始生成字符图片到 '{output_full_path}' ...")

    pbar = tqdm(characters, desc="生成进度", unit="char")
    
    for char in pbar:
        # 更新进度条右侧显示的当前字符信息
        pbar.set_postfix({"当前": char})
        
        # 使用工具函数绘制居中字符
        image = draw_centered_text(char, font, image_size=IMAGE_SIZE)
        
        # 保存图片
        file_path = os.path.join(output_full_path, f"char_{char}.png")
        image.save(file_path)

    print(f"\n所有字符图片已成功生成在 '{output_full_path}' 目录下。")

if __name__ == '__main__':
    generate_dictionary()