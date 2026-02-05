# -*- coding: utf-8 -*-
import os
import json
import random
import platform
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from config.settings import settings
from utils.text_drawing_utils import get_system_font_path, get_chinese_chars_from_unicode_range # NEW: Import char generator

class TrOCRDataGenerator:
    def __init__(self, num_images: int, device: str = "cpu"):
        self.num_images = num_images
        
        # Output directory for generated synthetic data
        self.output_dir = os.path.join(settings.paths.base_dir, settings.paths.synthetic_ques_data_dir)
        self.image_dir = os.path.join(self.output_dir, "images")
        self.label_file = os.path.join(self.output_dir, "labels.jsonl") # Changed from metadata.jsonl for consistency

        # Background images directory
        self.background_dir = os.path.join(settings.paths.base_dir, settings.paths.background_images_dir)
        
        self.device = device
        self.font_path = get_system_font_path('SimHei') # NEW: Use the utility function
        self.char_dict_full_path = os.path.join(settings.paths.base_dir, settings.paths.char_dict_path)
        # --- 使用 Unicode 范围生成字符集 ---
        if os.path.exists(self.char_dict_full_path):
            with open(self.char_dict_full_path, 'r', encoding='utf-8') as f:
                self.char_set = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.char_set = get_chinese_chars_from_unicode_range(max_chars=1000)
        self.background_images = [
            os.path.join(self.background_dir, f)
            for f in os.listdir(self.background_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ]
        self._setup_dirs()

    def _setup_dirs(self):
        os.makedirs(self.image_dir, exist_ok=True)
        # Ensure labels.jsonl parent directory exists
        Path(self.label_file).parent.mkdir(parents=True, exist_ok=True)



    def _find_perspective_coeffs(self, src, dst):
        # src, dst: [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
        matrix = []
        for (x, y), (u, v) in zip(src, dst):
            matrix.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
            matrix.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        A = np.array(matrix, dtype=np.float32)
        B = np.array([p for pair in dst for p in pair], dtype=np.float32)

        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return res.tolist()

    def _apply_hollow_effect(self, img):
        """模拟极验4中笔画中心镂空、边缘发光的效果"""
        if random.random() < 0.5:
            return img
        r, g, b, a = img.split()
        mask = a.convert("L")
        
        # 边缘提取：膨胀减去腐蚀
        edge = mask.filter(ImageFilter.MaxFilter(3))
        inner = mask.filter(ImageFilter.MinFilter(3))
        hollow_mask = ImageChops.subtract(edge, inner)
        
        # 强化边缘
        hollow_mask = hollow_mask.point(lambda p: p * 2 if p > 50 else 0)
        
        return Image.merge("RGBA", (r, g, b, hollow_mask))
    def _apply_perspective_warp(self, img):
        # 随机生成四个角的扰动
        w, h = img.size
        max_dx = int(w * 0.15)
        max_dy = int(h * 0.15)

        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [
            (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)),
            (w + random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)),
            (w + random.randint(-max_dx, max_dx), h + random.randint(-max_dy, max_dy)),
            (random.randint(-max_dx, max_dx), h + random.randint(-max_dy, max_dy)),
        ]

        coeffs = self._find_perspective_coeffs(src, dst)
        return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    def _apply_geetest_noise(self, img):
        """添加极验特有的高频彩色碎点噪声"""
        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape
        
        # 随机生成彩色噪点图
        noise_mask = np.random.rand(h, w) > 0.92
        for i in range(3): # RGB三通道
            arr[noise_mask, i] = random.randint(0, 255)
            
        return Image.fromarray(arr.astype(np.uint8), 'RGBA')
    def _apply_stroke_thickness_variation(self, img):
        # 提取 alpha 作为笔画结构
        r, g, b, a = img.split()

        # 转成灰度 mask
        mask = a.convert("L")

        # 随机决定是“变粗”还是“变细”
        mode = random.choice(["thicken", "thin"])
        k = random.randint(1, 3)  # 强度

        if mode == "thicken":
            for _ in range(k):
                mask = mask.filter(ImageFilter.MaxFilter(3))  # 膨胀
        else:
            for _ in range(k):
                mask = mask.filter(ImageFilter.MinFilter(3))  # 腐蚀

        # 重新组合颜色 + 新 mask
        out = Image.merge("RGBA", (r, g, b, mask))
        return out
    def _apply_3d_and_distort(self, canvas):
        """增加文字的3D厚度感和局部扭曲"""
        # 1. 模拟厚度：位移叠加
        r, g, b, a = canvas.split()
        thickness_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        for i in range(1, 4):  # 偏移3层
            thickness_layer.paste(canvas, (i, i), canvas)
        
        # 将厚度层变暗
        thickness_layer = ImageChops.multiply(thickness_layer, 
                                              Image.new("RGBA", canvas.size, (50, 50, 50, 255)))
        canvas = Image.alpha_composite(thickness_layer, canvas)

        # 2. 局部形变 (使用 numpy 模拟水波纹)
        arr = np.array(canvas)
        rows, cols, _ = arr.shape
        # 生成正弦波偏移
        for i in range(rows):
            offset = int(5 * np.sin(2 * np.pi * i / 30)) # 振幅5，周期30
            arr[i] = np.roll(arr[i], offset, axis=0)
        
        return Image.fromarray(arr)

    def _apply_chromatic_aberration(self, img, intensity=None):
        """色差边缘"""
        if intensity is None:
            intensity = random.randint(2, 5)
        r, g, b, a = img.split()
        r = ImageChops.offset(r, intensity, 0)
        b = ImageChops.offset(b, -intensity, 0)
        return Image.merge("RGBA", (r, g, b, a))
    def _erode_dilate_mask(self, mask, radius=1, iters=1, mode="erode"):
        """
        简易形态学操作：对 mask 做侵蚀或膨胀
        mask: PIL L image
        """
        arr = np.array(mask) > 128
        for _ in range(iters):
            padded = np.pad(arr, 1, mode="constant", constant_values=False)
            new = arr.copy()
            for y in range(arr.shape[0]):
                for x in range(arr.shape[1]):
                    neigh = padded[y:y+3, x:x+3]
                    if mode == "erode":
                        new[y, x] = neigh.all()
                    else:  # dilate
                        new[y, x] = neigh.any()
            arr = new
        return Image.fromarray((arr * 255).astype(np.uint8), "L")

    def _apply_character_style(self, char, font, bbox):
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = int(max(bw, bh) * 0.8)
        canvas = Image.new("RGBA", (bw + pad*2, bh + pad*2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        off_x, off_y = pad - bbox[0], pad - bbox[1]

        # 1. 模拟极验高饱和度色彩
        # 使用 HSV 确保颜色足够“亮”
        h = random.random()
        s = random.uniform(0.7, 1.0)
        v = random.uniform(0.8, 1.0)
        from colorsys import hsv_to_rgb
        rgb = hsv_to_rgb(h, s, v)
        base_color = tuple(int(x * 255) for x in rgb)
        inner_color = (255 - base_color[0], 255 - base_color[1], 255 - base_color[2], 100)
        
        # 2. 绘制多层描边（立体感）
        for dist in range(random.randint(2, 4), 0, -1):
            stroke_color = (0, 0, 0, 200) if dist > 1 else (255, 255, 255, 255)
            draw.text((off_x, off_y), char, font=font, fill=base_color,
                      stroke_width=dist, stroke_fill=stroke_color)
            draw.text((off_x, off_y), char, font=font, fill=inner_color) # 填充淡色内部
        # 3. 镂空效果
        canvas = self._apply_hollow_effect(canvas)
        
        # 4. 色差偏移
        canvas = self._apply_chromatic_aberration(canvas, intensity=random.randint(1, 3))
        
        # 5. 彩色噪点
        canvas = self._apply_geetest_noise(canvas)
        # 6. 3D厚度感和局部扭曲
        canvas = self._apply_3d_and_distort(canvas)
        return canvas

    def generate(self):
        print(f"开始生成 TrOCR 数据集...")
        if not self.font_path:
            print("Error: No suitable Chinese font found.")
            return
        with open(self.label_file, mode='w', encoding='utf-8', newline='') as f:

            for i in tqdm(range(self.num_images), desc="数据生成进度", unit="img"):
                try:
                    bg = Image.open(random.choice(self.background_images)).convert("RGBA")
                    iw, ih = bg.size
                    draw_layer = Image.new("RGBA", (iw, ih), (0,0,0,0))
                    placed_info = []
                    num_chars_to_generate = 3
                    if len(self.char_set) < num_chars_to_generate:
                        print(f"Warning: Character set has only {len(self.char_set)} characters, but {num_chars_to_generate} were requested. Adjusting to {len(self.char_set)}.")
                        num_chars_to_generate = len(self.char_set)

                    chars = random.sample(self.char_set, num_chars_to_generate)
                    scale = ((iw /300) + (ih/200))/2
                    for char in chars:
                        f_size = int(random.uniform(45, 60)*scale)
                        font = ImageFont.truetype(self.font_path, f_size)
                        styled_img = self._apply_character_style(char, font, font.getbbox(char))
                        # 笔画厚度扰动
                        if random.random() < 0.7:
                            styled_img = self._apply_stroke_thickness_variation(styled_img)
                        # 几何破坏
                        if random.random() < 0.7:
                            styled_img = self._apply_perspective_warp(styled_img)
                        # 亮度调节
                        if random.random() < 0.3:
                            bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

                        rotated = styled_img.rotate(random.randint(-60, 60), expand=True)

                        rw, rh = rotated.size
                        # 确保文字不超出背景，且随机范围合法
                        if rh > ih * 0.8:
                            ratio = (ih * 0.8) / rh
                            rotated = rotated.resize((int(rw*ratio), int(rh*ratio)), Image.LANCZOS)
                            rw, rh = rotated.size

                        x_max = max(0, iw - rw)
                        y_min = int(ih * 0.15)
                        y_max = max(y_min, ih - rh - 10)

                        # 使用 try-except 保护随机数生成或直接使用三元运算
                        x = random.randint(0, x_max) if x_max > 0 else 0
                        y = random.randint(y_min, y_max) if y_max > y_min else y_min
                        
                        overlap = False
                        for p in placed_info:
                            if abs(x - p['x']) < rw * 0.45: overlap = True; break
                        
                        if not overlap:
                            draw_layer.paste(rotated, (x, y), rotated)
                            placed_info.append({'char': char, 'x': x})

                    if not placed_info: continue

                    # 按 X 坐标从左到右排序（TrOCR 识别顺序）
                    placed_info.sort(key=lambda x: x['x'])
                    full_text = "".join([p['char'] for p in placed_info])

                    final = Image.alpha_composite(bg, draw_layer).convert("RGB")
                    file_name = f"syn_{i:05d}.png"
                    final.save(os.path.join(self.image_dir, file_name))
                    entry = {
                        "file_name": file_name,
                        "text": full_text
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                except Exception as e:
                    tqdm.write(f"第 {i} 张图片生成出错: {e}")

        print(f"完成！生成了 {self.num_images} 张图片。标签: {self.label_file}")

if __name__ == '__main__':
    generator = TrOCRDataGenerator(num_images=5) # 建议生产环境设为 10000
    generator.generate()
