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
from scipy.optimize import linear_sum_assignment
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
    def _segment_and_straighten_char(self, img_pil):
        """
        使用 GrabCut 分割前景并使用 minAreaRect 校正方向。
        返回4个旋转角度 (0, 90, 180, 270) 的 PIL 图像列表。
        """
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        mask = np.zeros(img_cv.shape[:2], np.uint8)
        rect = (1, 1, img_cv.shape[1] - 2, img_cv.shape[0] - 2)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img_cv, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        except Exception:
            # 如果 GrabCut 失败（例如，对于完全空白的图像），返回空列表
            return []

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 寻找前景轮廓
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # 找到最大轮廓并获取其最小面积矩形
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        
        angle = rect[2]
        size = tuple(map(int, rect[1]))
        center = tuple(map(int, rect[0]))

        # 根据角度调整，确保宽度大于高度
        if size[0] < size[1]:
            angle += 90
            size = (size[1], size[0])

        # 获取旋转矩阵并应用
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(img_cv, M, (img_cv.shape[1], img_cv.shape[0]))

        # 裁剪出校正后的字符区域
        img_cropped = cv2.getRectSubPix(img_rotated, size, center)
        
        if img_cropped is None:
            # 如果裁剪失败，通常是因为尺寸或中心点问题，返回空列表
            print(f"Debug: cv2.getRectSubPix for img_cropped returned None for size={size}, center={center}")
            return []

        # 为白色背景创建一个新的图像
        img_final_bgr = np.full((size[1], size[0], 3), (255, 255, 255), dtype=np.uint8)
        
        # 将 GrabCut 蒙版也进行旋转和裁剪
        mask_rotated = cv2.warpAffine(mask2, M, (mask2.shape[1], mask2.shape[0]))
        mask_cropped = cv2.getRectSubPix(mask_rotated, size, center)
        
        # 再次检查 mask_cropped 是否为 None，以防万一
        if mask_cropped is None:
            print(f"Debug: cv2.getRectSubPix for mask_cropped returned None for size={size}, center={center}")
            return []
        
        # 使用蒙版将前景粘贴到白色背景上
        img_final_bgr[mask_cropped == 1] = img_cropped[mask_cropped == 1]

        base_img = Image.fromarray(cv2.cvtColor(img_final_bgr, cv2.COLOR_BGR2RGB))

        # 生成四个旋转角度的图像
        rotated_images = []
        for rot_angle in [0, 90, 180, 270]:
            rotated_images.append(base_img.rotate(rot_angle, expand=True))
            
        return rotated_images

    def find_best_matches_for_main_image(self, main_char_images, ques_char_images, weights=None):
        """
        为从主图中提取的字符（复杂背景）找到最佳匹配。
        
        Args:
            main_char_images (list[PIL.Image]): 从主图中分割出的字符图片列表。
            ques_char_images (list[PIL.Image]): "ques" 中的字符图片列表。
            weights (dict, optional): 相似度算法的权重。

        Returns:
            list[tuple]: 一个匹配结果列表，每个元素是 (main_idx, ques_idx, score)。
        """
        if weights is None:
            weights = {'ssim': 1.0, 'hog': 1.0, 'proj': 1.0}

        num_main = len(main_char_images)
        num_ques = len(ques_char_images)

        # 1. 预处理 "ques" 图像特征
        ques_features = [self._extract_features(img) for img in ques_char_images]

        # 2. 构建相似度矩阵
        similarity_matrix = np.zeros((num_main, num_ques))

        for i, main_img in enumerate(main_char_images):
            # 2a. 分割和校正主图中的字符，并获得4个旋转版本
            rotated_main_images = self._segment_and_straighten_char(main_img)
            if not rotated_main_images:
                continue

            for j in range(num_ques):
                ques_img_array, ques_hog, ques_proj = ques_features[j]
                max_score_for_this_pair = -1.0

                # 2b. 遍历4个旋转版本，找到与当前 ques_char 最高的匹配分
                for rot_img in rotated_main_images:
                    main_img_array, main_hog, main_proj = self._extract_features(rot_img)
                    if main_img_array is None:
                        continue
                    
                    # --- 计算加权相似度得分 ---
                    ssim_score = ssim(main_img_array, ques_img_array, data_range=255)
                    hog_sim_score = cosine_similarity(main_hog.reshape(1, -1), ques_hog.reshape(1, -1))[0][0]
                    dist = np.linalg.norm(main_proj - ques_proj)
                    proj_sim_score = 1.0 / (1.0 + dist)
                    
                    total_score = (weights.get('ssim', 0) * ssim_score +
                                   weights.get('hog', 0) * hog_sim_score +
                                   weights.get('proj', 0) * proj_sim_score)
                    
                    if total_score > max_score_for_this_pair:
                        max_score_for_this_pair = total_score
                
                similarity_matrix[i, j] = max_score_for_this_pair

        # 3. 使用匈牙利算法求解最优匹配
        # linear_sum_assignment 需要成本矩阵，所以我们用最大相似度减去当前值
        cost_matrix = np.max(similarity_matrix) - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 4. 整理并返回结果
        matches = []
        for r, c in zip(row_ind, col_ind):
            score = similarity_matrix[r, c]
            matches.append((r, c, score))
            
        # 按分数降序排序
        matches.sort(key=lambda x: x[2], reverse=True)

        return matches

if __name__ == '__main__':
    # 示例用法，实际运行时请通过 run_matching.py 调用
    print("这是一个模块文件，不建议直接运行。")
    print("请通过运行 'run_matching.py' 来查看效果。")
