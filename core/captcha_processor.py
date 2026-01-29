"""
验证码处理器
整合了验证码处理的完整流程、模式切换和错误管理。
"""
import time
import logging
import requests
import numpy as np
import scipy.optimize
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

# 统一从config模块导入全局配置
from config import settings
from .gt4 import GeetestV4
# 导入重构后的函数式模块
from utils import image_processor, coordinate_utils
from . import yolo_inference, manual_fallback
from .trocr_recognizer import TrOCRRecognizer
from .paddle_recognizer import PaddleRecognizer, cosine_similarity

class CaptchaProcessor:
    """
    验证码处理器，作为核心业务流程的 orchestrator。
    - 管理自动（YOLO）和手动处理模式。
    - 内置错误计数和模式切换逻辑。
    """
    def __init__(self, session: Optional[requests.Session] = None):
        """
        初始化验证码处理器。
        """
        self.settings = settings
        self.session = session or requests.Session()
        self.logger = logging.getLogger(__name__)

        # 核心组件
        self.geetest = GeetestV4(self.settings.geetest.captcha_id, geetest_config=self.settings.geetest, session=self.session)
        self.yolo_model, self.yolo_class_names = yolo_inference.load_model(self.settings.yolo_inference, self.settings.paths)
        
        # 根据配置初始化OCR识别器
        self.ocr_recognizer = None
        if self.settings.ocr.engine == 'trocr':
            self.logger.info("初始化 TrOCR 识别器...")
            self.ocr_recognizer = TrOCRRecognizer(model_name=self.settings.ocr.trocr.model_name, device=self.settings.ocr.trocr.device)
        elif self.settings.ocr.engine == 'paddle':
            self.logger.info("初始化 PaddleOCR 识别器 (支持文本识别和特征提取)...")
            self.ocr_recognizer = PaddleRecognizer()
        else:
            self.logger.error(f"不支持的OCR引擎: {self.settings.ocr.engine}。请检查config/settings.py。")
            self.ocr_recognizer = None

        # 状态管理
        self.current_mode = "auto"
        self.consecutive_auto_failures = 0
        self.consecutive_manual_successes = 0

        self.logger.info("验证码处理器初始化完成。")

    def process(self, captcha_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        处理验证码的主入口点。
        """
        start_time = time.time()
        
        # 加载验证码
        load_data = self.geetest.load(captcha_id=captcha_id or self.settings.geetest.captcha_id, **kwargs)
        if load_data.get("status") != "success":
            return {"success": False, "error": "Failed to load captcha", "details": load_data}

        # 根据当前模式选择处理方式
        if self.current_mode == "auto":
            result = self._process_auto(load_data)
        else: # self.current_mode == "manual"
            result = self._process_manual(load_data)
        
        processing_time = time.time() - start_time
        result['total_time'] = processing_time
        self.logger.info(f"处理完成，耗时: {processing_time:.3f}s，模式: {self.current_mode}，成功: {result['success']}")

        # 更新模式
        self._update_mode(result['success'])
        
        return result

    def _process_auto(self, load_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用智能混合策略自动处理验证码：对每个字符独立判断，优先文本匹配，失败则启用相似度匹配。"""
        if self.yolo_model is None: return {"success": False, "error": "YOLO model not loaded.", "mode": "auto"}
        if self.ocr_recognizer is None: return {"success": False, "error": "Recognizer not initialized.", "mode": "auto"}

        # --- 1. 数据准备 ---
        image_urls = self.geetest.extract_image_urls(load_data)
        if not image_urls.get("ques_imgs"): return {"success": False, "error": "No 'ques_imgs' in captcha data.", "mode": "auto"}

        main_image = image_processor.download_image(self.session, image_urls["main_img"])
        if main_image is None: return {"success": False, "error": "Failed to download captcha image.", "mode": "auto"}

        ques_images = [image_processor.download_image(self.session, url) for url in image_urls["ques_imgs"]]
        if any(img is None for img in ques_images): return {"success": False, "error": "Failed to download a question image.", "mode": "auto"}

        detections = yolo_inference.detect(self.yolo_model, main_image, self.yolo_class_names, self.settings.yolo_inference)
        if not detections: return {"success": False, "error": "No objects detected by YOLO model.", "mode": "auto"}

        # --- 2. 识别与分组 ---
        self.logger.info("--- 步骤 1: 识别 'ques' 图片并分组 ---")
        solved_by_text = []
        unsolved_for_similarity = []
        for i, img in enumerate(ques_images):
            text = self.ocr_recognizer.recognize(img)
            if text and len(text) == 1:
                solved_by_text.append({'index': i, 'char': text})
                self.logger.info(f"Ques {i}: 文本识别成功 -> '{text}'")
            else:
                unsolved_for_similarity.append({'index': i, 'image': img, 'original_text': text})
                self.logger.warning(f"Ques {i}: 文本识别失败 (输出: '{text}'), 将使用相似度匹配。")

        # 对所有检测出的人物进行文本识别，以备后用
        for det in detections:
            det['class_name'] = self.ocr_recognizer.recognize(main_image.crop(det['bbox']))

        # --- 3. 优先处理文本匹配 ---
        self.logger.info("--- 步骤 2: 执行文本匹配 ---")
        final_coords = [None] * len(ques_images)
        
        available_dets_map = defaultdict(list)
        for det in detections:
            if det['class_name']:
                available_dets_map[det['class_name']].append(det)
        
        used_det_centers = set()

        for item in solved_by_text:
            char_to_find = item['char']
            if available_dets_map[char_to_find]:
                found_match = False
                for det in sorted(available_dets_map[char_to_find], key=lambda d: d['center'][0]):
                    if tuple(det['center']) not in used_det_centers:
                        final_coords[item['index']] = det['center']
                        used_det_centers.add(tuple(det['center']))
                        available_dets_map[char_to_find].remove(det)
                        self.logger.info(f"文本匹配: Ques {item['index']} ('{char_to_find}') -> 坐标 {det['center']}")
                        found_match = True
                        break
                if not found_match:
                    self.logger.warning(f"文本匹配: 未能为 Ques {item['index']} ('{char_to_find}') 找到一个未被占用的坐标。")
            else:
                self.logger.warning(f"文本匹配: 在图片中未检测到任何字符 '{char_to_find}'。")

        # --- 4. 对剩余部分进行相似度匹配 ---
        if unsolved_for_similarity:
            self.logger.info("--- 步骤 3: 对剩余字符执行相似度匹配 ---")
            if self.settings.ocr.engine != 'paddle':
                return {"success": False, "error": "Similarity matching requires PaddleOCR engine.", "mode": "auto"}

            remaining_dets = [d for d in detections if tuple(d['center']) not in used_det_centers]
            if len(remaining_dets) < len(unsolved_for_similarity):
                self.logger.error(f"相似度匹配失败：剩余检测区域 ({len(remaining_dets)}) 少于待匹配目标 ({len(unsolved_for_similarity)})。")
            else:
                ques_embeddings = [{'index': item['index'], 'embedding': self.ocr_recognizer.get_embedding(item['image'])} for item in unsolved_for_similarity]
                det_embeddings = [{'center': d['center'], 'embedding': self.ocr_recognizer.get_embedding(main_image.crop(d['bbox']))} for d in remaining_dets]

                similarity_threshold = self.settings.ocr.paddle.similarity_threshold
                cost_matrix = np.full((len(ques_embeddings), len(det_embeddings)), 2.0)
                for i in range(len(ques_embeddings)):
                    for j in range(len(det_embeddings)):
                        if ques_embeddings[i]['embedding'] is not None and det_embeddings[j]['embedding'] is not None:
                            sim = cosine_similarity(ques_embeddings[i]['embedding'], det_embeddings[j]['embedding'])
                            if sim >= similarity_threshold:
                                cost_matrix[i, j] = 1 - sim
                
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 1.0:
                        ques_item_index = ques_embeddings[r]['index']
                        det_center = det_embeddings[c]['center']
                        final_coords[ques_item_index] = det_center
                        self.logger.info(f"相似度匹配: Ques {ques_item_index} -> 坐标 {det_center}")

        # --- 5. 最终验证 ---
        if any(c is None for c in final_coords):
            self.logger.error(f"所有策略处理完毕，但未能为所有目标字符找到坐标。最终坐标: {final_coords}")
            return {"success": False, "error": "Could not find coordinates for all target characters.", "mode": "auto"}

        self.logger.info(f"最终确定的点击坐标顺序: {final_coords}")
        geetest_coords = coordinate_utils.convert_to_geetest_format(final_coords, (main_image.width, main_image.height))
        w_data = self.geetest.generate_w_data(load_data, userresponse=geetest_coords, passtime=int(time.time() * 1000) % 5000 + 2000)
        verify_result = self.geetest.verify(w=w_data['w'], load_data=load_data)

        success = verify_result.get("status") == "success"
        return {"success": success, "details": verify_result, "mode": "auto"}


    def _process_manual(self, load_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用手动模式处理验证码。"""
        image_urls = self.geetest.extract_image_urls(load_data)
        
        user_coords, passtime = manual_fallback.get_user_input_with_gui(
            main_image_url=image_urls["main_img"],
            ques_image_urls=image_urls.get("ques_imgs", []),
            session=self.session,
            timeout=self.settings.mode_switch.manual_timeout
        )

        if not user_coords:
            return {"success": False, "error": "User did not provide input or timed out.", "mode": "manual"}
        
        w_data = self.geetest.generate_w_data(load_data, userresponse=user_coords, passtime=passtime)
        verify_result = self.geetest.verify(w=w_data['w'], load_data=load_data)

        success = verify_result.get("status") == "success"
        return {"success": success, "details": verify_result, "mode": "manual"}
    
    def _update_mode(self, success: bool):
        """根据处理结果更新模式。"""
        if self.current_mode == "auto":
            if success:
                self.consecutive_auto_failures = 0
            else:
                self.consecutive_auto_failures += 1
                self.logger.warning(f"自动模式连续失败 {self.consecutive_auto_failures} 次。")
                if self.consecutive_auto_failures >= self.settings.mode_switch.max_auto_failures:
                    self.current_mode = "manual"
                    self.logger.error(f"自动模式失败达到阈值，切换到手动模式。")
                    self.consecutive_auto_failures = 0
        
        elif self.current_mode == "manual":
            if success:
                self.consecutive_manual_successes += 1
                self.logger.info(f"手动模式连续成功 {self.consecutive_manual_successes} 次。")
                if self.consecutive_manual_successes >= self.settings.mode_switch.min_success_for_switch:
                    self.current_mode = "auto"
                    self.logger.info(f"手动模式成功达到阈值，切换回自动模式。")
                    self.consecutive_manual_successes = 0
            else:
                self.consecutive_manual_successes = 0

    def _calculate_click_coords(self, detections: List[Dict], target_chars: List[str]) -> List[Tuple[int, int]]:
        """根据检测结果和目标文字计算点击坐标。"""
        sorted_detections = sorted(detections, key=lambda d: d['bbox'][0])
        
        char_map = defaultdict(list)
        for det in sorted_detections:
            char_map[det['class_name']].append(det['center'])
            
        click_coords = []
        for char in target_chars:
            if char_map[char]:
                click_coords.append(char_map[char].pop(0))
        
        return click_coords

    def _generate_random_click_coords(self, num_coords: int, image_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        生成随机点击坐标（仅用于调试流程）。
        """
        random_coords = []
        width, height = image_size
        import random
        for _ in range(num_coords):
            random_coords.append((random.randint(int(width * 0.1), int(width * 0.9)), random.randint(int(height * 0.1), int(height * 0.9))))
        return random_coords