# -*- coding: utf-8 -*-
import os
import json
import random
import platform
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from config.settings import settings

class TrOCRDataGenerator:
    def __init__(self, num_images: int, device: str = "cpu"):
        self.num_images = num_images
        
        # Output directory for generated synthetic data
        self.output_dir = os.path.join(settings.paths.base_dir, settings.paths.synthetic_trocr_data_dir)
        self.image_dir = os.path.join(self.output_dir, "images")
        self.label_file = os.path.join(self.output_dir, "labels.jsonl") # Changed from metadata.jsonl for consistency

        # Background images directory
        self.background_dir = os.path.join(settings.paths.base_dir, settings.paths.background_images_dir)
        
        self.device = device
        self.font_path = self._get_system_font()
        # --- 使用 Unicode 范围生成字符集 ---
        self.char_set = self._get_chinese_chars_from_unicode_range()
        self.background_images = [
            os.path.join(self.background_dir, f)
            for f in os.path.listdir(self.background_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ]
        self._setup_dirs()

    def _setup_dirs(self):
        os.makedirs(self.image_dir, exist_ok=True)
        # Ensure labels.jsonl parent directory exists
        Path(self.label_file).parent.mkdir(parents=True, exist_ok=True)

    def _get_system_font(self):
        system = platform.system()
        paths = {
            "Darwin": ["/System/Library/Fonts/STHeiti Medium.ttc"],
            "Windows": ["C:/Windows/Fonts/msyh.ttc"],
            "Linux": ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]
        }.get(system, [])
        for path in paths:
            if os.path.exists(path): return path
        return None

    def _get_chinese_chars_from_unicode_range(self):
        """
        # CJK 统一汉字 (Common CJK Unified Ideographs) 块
        # 仅取 U+4E00 到 U+9FA5 的汉字，覆盖简体汉字
        """
        chars = []
        start_code = 0x4E00
        end_code = 0x9FA5
        
        for code in range(start_code, end_code + 1):
            char = chr(code)
            # 可以在此处添加过滤条件，例如排除不常用的字，或者确保是可打印字符
            # 对于大多数 CJK 范围内的字符，直接添加即可
            chars.append(char)
        
        print(f"已从 Unicode 范围 {hex(start_code)} - {hex(end_code)} 生成 {len(chars)} 个汉字。")
        
        # 如果需要更小的字符集，可以在此处进行切片或进一步筛选
        # 例如，为了控制验证码的复杂性，可以只取前 N 个或一个子集：
        chars = chars[:3500]  # 前 3500 个最常用简体字
        return chars
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

        neon_colors = [(255, 50, 50), (50, 255, 50), (255, 255, 0), (255, 0, 255)]
        base_color = random.choice(neon_colors)
        # 多层重影（Ghosting / Multi-layering）
        for i in range(random.randint(3, 5), 0, -1):
            alpha = 100 // i
            offset = i * 2
            draw.text((off_x + offset, off_y + offset), char, font=font, 
                      fill=base_color + (alpha,), stroke_width=random.randint(2, 4), stroke_fill=(0,0,0, alpha))
            draw.text((off_x - offset, off_y - offset), char, font=font, 
                      fill=base_color + (alpha,), stroke_width=random.randint(2, 4), stroke_fill=(0,0,0, alpha))

        draw.text((off_x, off_y), char, font=font, fill=base_color + (255,), 
                  stroke_width=random.randint(2, 4), stroke_fill=(255,255,255,255))
        canvas = self._apply_chromatic_aberration(canvas)
        # --- 局部腐蚀 / 焦裂 ---
        if random.random() < 0.6:
            r, g, b, a = canvas.split()
            mask = a.convert("L")
            m = np.array(mask)

            h, w = m.shape
            for _ in range(random.randint(2, 5)):
                cx = random.randint(0, w-1)
                cy = random.randint(0, h-1)
                r0 = random.randint(3, max(4, min(w, h) // 12))

                y0 = max(0, cy - r0)
                y1 = min(h, cy + r0)
                x0 = max(0, cx - r0)
                x1 = min(w, cx + r0)

                sub = m[y0:y1, x0:x1]
                if sub.size == 0:
                    continue

                # 简单腐蚀：阈值收缩
                edge = sub < 200
                sub[edge] = sub[edge] * random.uniform(0.2, 0.6)
                m[y0:y1, x0:x1] = sub

            new_mask = Image.fromarray(m.astype(np.uint8), "L")
            canvas = Image.merge("RGBA", (r, g, b, new_mask))
        # 非连续轮廓（像素级破碎）
        arr = np.array(canvas).astype(np.float32)
        noise = np.random.normal(0, 30, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(arr, 'RGBA')

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
