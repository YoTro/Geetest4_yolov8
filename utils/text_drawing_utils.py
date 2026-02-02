# -*- coding: utf-8 -*-
import requests # NEW: For downloading fonts
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import os
from config.settings import settings # NEW: For font_dir path

# Google Fonts URL for Noto Sans CJK SC Regular
msgothic_URL = "https://github.com/samuelclay/NewsBlur/raw/refs/heads/main/media/fonts/MS%20Gothic.ttf"
msgothic_FILENAME = "MS Gothic"

def get_system_font_path(font_name: str) -> str:
    """
    使用 matplotlib 从系统中查找指定名称的字体文件路径。
    如果找不到常用中文字体，则尝试下载一个开源字体作为备用。

    Args:
        font_name (str): 优先查找的字体名称, 例如 'SimHei'。

    Returns:
        str: 字体文件的绝对路径，如果找不到则返回 None。
    """
    # 1. 尝试查找用户指定的字体 (例如 SimHei)
    try:
        font_prop = fm.FontProperties(family=font_name)
        system_font_path = fm.findfont(font_prop)
        if font_name.lower() in system_font_path.lower() and 'ttc' in system_font_path:
            print(f"找到系统字体 '{font_name}': {system_font_path}")
            return system_font_path
    except Exception:
        pass # 继续尝试其他方法

    # 2. 尝试查找常用的开源中文字体 (例如 Noto Sans CJK SC)
    try:
        font_prop_fallback = fm.FontProperties(family=msgothic_FILENAME)
        system_font_path_fallback = fm.findfont(font_prop_fallback)
        if msgothic_FILENAME.lower() in system_font_path_fallback.lower() and 'ttc' in system_font_path_fallback:
            print(f"系统未找到 '{font_name}'，使用备用字体 '{msgothic_FILENAME}': {system_font_path_fallback}")
            return system_font_path_fallback
    except Exception:
        pass # 继续尝试下载

    # 3. 如果系统仍未找到合适字体，则尝试下载 Noto Sans CJK SC
    print(f"系统未找到 '{font_name}' 或 '{msgothic_FILENAME}' 字体，尝试下载备用字体...")
    font_dir = os.path.join(settings.paths.base_dir, settings.paths.font_dir)

    local_font_path = os.path.join(font_dir, font_name)+".ttc"

    if os.path.exists(local_font_path):
        print(f"已在 '{font_dir}' 中找到下载的字体: {local_font_path}")
        return local_font_path

    print(f"正在从 {font_name} 下载备用字体到 '{local_font_path}'...")
    try:
        os.makedirs(font_dir, exist_ok=True) # 确保字体目录存在
        response = requests.get(msgothic_URL, stream=True)
        response.raise_for_status() # 检查HTTP请求是否成功
        with open(local_font_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"字体下载完成: {local_font_path}")
        return local_font_path
    except requests.exceptions.RequestException as e:
        print(f"错误：下载字体失败: {e}")
    except Exception as e:
        print(f"错误：保存下载字体时发生异常: {e}")

    print("错误：未能找到或下载任何中文字体。")
    return None

def draw_centered_text(char: str, font: ImageFont.FreeTypeFont, image_size=(64, 65)) -> Image:
    """
    在透明背景上绘制一个居中的字符。

    Args:
        char (str): 需要绘制的单个字符。
        font (ImageFont.FreeTypeFont): 用于绘制的 PIL 字体对象。
        image_size (tuple): 输出图像的尺寸 (width, height)。

    Returns:
        PIL.Image.Image: 绘制了字符的 RGBA 图像对象。
    """
    # 创建一个支持透明度的 RGBA 图像
    image = Image.new('RGBA', image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 获取文字的边界框来计算居中位置
    try:
        # Pillow >= 10.0.0, textbbox is preferred
        bbox = draw.textbbox((0, 0), char, font=font)
    except AttributeError:
        # Pillow < 10.0.0, fallback to textsize
        w, h = draw.textsize(char, font=font)
        bbox = (0, 0, w, h)
        
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = (
        (image_size[0] - text_width) / 2 - bbox[0],
        (image_size[1] - text_height) / 2 - bbox[1]
    )

    # 在图片上绘制黑色文字
    draw.text(position, char, font=font, fill=(0, 0, 0, 255))
    
    return image


def get_chinese_chars_from_unicode_range(max_chars: int = 3500) -> list:
    """
    从 Unicode 的 CJK 统一汉字块中生成一个常用汉字列表。
    范围: U+4E00 到 U+9FA5.

    Args:
        max_chars (int): 返回的最大字符数。

    Returns:
        list: 生成的汉字字符列表。
    """
    chars = []
    start_code = 0x4E00
    end_code = 0x9FA5
    
    for code in range(start_code, end_code + 1):
        chars.append(chr(code))
    
    print(f"已从 Unicode 范围 {hex(start_code)} - {hex(end_code)} 生成 {len(chars)} 个汉字。")
    
    # 根据需要截取，以获得最常用的部分
    if max_chars > 0 and len(chars) > max_chars:
        chars = chars[:max_chars]
        print(f"已将字符集缩减至前 {max_chars} 个最常用字。")

    return chars
