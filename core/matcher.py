# -*- coding: utf-8 -*-
import os
import cv2
import pickle
import numpy as np
from PIL import Image
import imagehash
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import settings # NEW: Import global settings

class ImageMatcher:
    def __init__(self, use_cache=True):
        """
        初始化图片匹配器。
        加载字典图片并预计算它们的特征。
        """
        self.dictionary_path = os.path.join(settings.paths.base_dir, settings.paths.dict_image_dir)
        # 定义缓存文件路径
        self.cache_path = os.path.join(self.dictionary_path, "feature_cache.pkl")
        self.target_size = (64, 64) # 统一尺寸变量
        self.dictionary = self._load_dictionary(use_cache)
        if self.dictionary:
            print(f"ImageMatcher 已初始化，从 '{self.dictionary_path}' 加载了 {len(self.dictionary)} 个字符。")
    def _preprocess_image(self, img_pil):
        """统一预处理：RGB -> 灰度 -> 填充背景 -> Resize"""
        # 如果是 RGBA，先复合到白色背景上，避免透明度干扰
        if img_pil.mode == 'RGBA':
            background = Image.new("RGB", img_pil.size, (255, 255, 255))
            background.paste(img_pil, mask=img_pil.split()[3])
            img_pil = background
        
        img_pil = img_pil.convert('L')
        img_pil = img_pil.resize(self.target_size, Image.LANCZOS) # 建议用 64x64 对称尺寸
        return img_pil

    def _load_dictionary(self, use_cache):
        """
        加载字典图片并预计算它们的 HOG 和 Projection 特征。
        """
        # 1. 尝试从缓存加载
        if use_cache and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"成功从缓存加载特征库: {self.cache_path}")
                return data
            except Exception as e:
                print(f"读取缓存失败，将重新计算: {e}")
        return {}
    def _extract_projection_features(self, img_array):
        """
        提取水平和垂直投影特征
        """
        # 1. 二值化处理（投影算法对二值化非常敏感）
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 2. 水平投影 (每一行像素之和)
        h_proj = np.sum(binary, axis=1) / 255
        # 3. 垂直投影 (每一列像素之和)
        v_proj = np.sum(binary, axis=0) / 255
        
        # 归一化，防止笔画粗细影响
        h_proj = h_proj / (np.max(h_proj) + 1e-6)
        v_proj = v_proj / (np.max(v_proj) + 1e-6)
        
        return np.concatenate([h_proj, v_proj])
    def _extract_features(self, image_input):
        """
        从目标图片中提取 SSIM、HOG 和 Projection 特征。
        image_input 可以是图片路径 (str) 或 PIL.Image 对象。
        """
        try:
            if isinstance(image_input, str):
                img_pil = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                img_pil = image_input
            else:
                raise ValueError("image_input 必须是文件路径字符串或 PIL.Image 对象。")

            img_gray = self._preprocess_image(img_pil)
            img_array = np.array(img_gray)

            fd, _ = hog(img_array, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=None)
            proj = self._extract_projection_features(img_array)

            return img_array, fd, proj
        except Exception as e:
            # 修改错误信息，使其更通用
            input_repr = image_input if isinstance(image_input, str) else "PIL Image object"
            print(f"提取目标图片 '{input_repr}' 特征失败: {e}")
            return None, None, None

    def find_best_match(self, target_image_path, weights=None):
        """
        查找与目标图片最匹配的字符（采用两阶段匹配优化）。

        Args:
            target_image_path (str): 待识别的目标图片路径。
            weights (dict, optional): 各相似度算法的权重。

        Returns:
            tuple: (最匹配的字符, 最高得分)，如果没有匹配则返回 (None, -1)。
        """
        if weights is None:
            weights = {'ssim': 1.0, 'hog': 1.0, 'proj': 1.0}

        target_img_array, target_hog_features, target_proj = self._extract_features(target_image_path)

        if target_img_array is None:
            return None, -1

        # --- 第一阶段: 使用 proj 进行快速粗筛 ---
        TOP_N_CANDIDATES = 25  # 定义粗筛后保留的候选项数量
        
        candidate_distances = []
        for char, data in self.dictionary.items():
            proj_distance = np.linalg.norm(target_proj - data['proj']) # 欧氏距离
            candidate_distances.append({'char': char, 'distance': proj_distance})
            
        # 根据 proj 距离从小到大排序
        candidate_distances.sort(key=lambda x: x['distance'])
        #print(candidate_distances)
        # 获取 Top N 候选项的字符列表
        top_candidates = [item['char'] for item in candidate_distances[:TOP_N_CANDIDATES]]

        # --- 第二阶段: 对 Top N 候选项进行精确匹配 ---
        best_char = None
        max_score = -1.0

        for char in top_candidates:
            data = self.dictionary[char]
            dict_img_array = data['image_array']
            dict_hog_features = data['hog_features']
            dict_proj = data['proj']

            # 1. SSIM 相似度 (计算昂贵)
            ssim_score = ssim(target_img_array, dict_img_array, data_range=255)

            # 2. HOG + 余弦相似度 (计算昂贵)
            hog_sim_score = cosine_similarity(target_hog_features.reshape(1, -1),
                                              dict_hog_features.reshape(1, -1))[0][0]

            # 3. proj 相似度 (计算非常快)
            dist = np.linalg.norm(target_proj - dict_proj)
            proj_sim_score = 1.0 / (1.0 + dist)

            # 加权求和
            total_score = (weights.get('ssim', 0) * ssim_score +
                           weights.get('hog', 0) * hog_sim_score +
                           weights.get('proj', 0) * proj_sim_score)

            if total_score > max_score:
                max_score = total_score
                best_char = char

        return best_char, max_score

if __name__ == '__main__':
    # 示例用法，实际运行时请通过 run_matching.py 调用
    print("这是一个模块文件，不建议直接运行。")
    print("请通过运行 'run_matching.py' 来查看效果。")
