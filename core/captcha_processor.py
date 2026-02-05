"""
验证码处理器
整合了验证码处理的完整流程、模式切换和错误管理。
"""
import os
import sys
import time
import random
import logging
import requests
import numpy as np
import scipy.optimize
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# 统一从config模块导入全局配置
from config import settings
from .gt4 import GeetestV4
# 导入重构后的函数式模块
from utils import image_processor, coordinate_utils
from . import yolo_inference, manual_fallback
from .trocr_recognizer import TrOCRRecognizer
from .paddle_recognizer import PaddleRecognizer, cosine_similarity
from .matcher import ImageMatcher # NEW: 导入 ImageMatcher


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

        # NEW: 初始化 ImageMatcher 用于 ques 图片识别
        self.image_matcher = ImageMatcher()
        if not self.image_matcher.dictionary:
            self.logger.warning("ImageMatcher 字典为空。ques 图片识别可能依赖 OCR 引擎或字典生成失败。")
        self.logger.info("验证码处理器初始化完成。")
        self.consecutive_auto_failures = 0
    def process(self, captcha_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        处理验证码的主入口点。
        """
        start_time = time.time()
        
        # 加载验证码
        load_data = self.geetest.load(captcha_id=captcha_id or self.settings.geetest.captcha_id, **kwargs)
        if load_data.get("status") != "success":
            return {"success": False, "error": "Failed to load captcha", "details": load_data}
        # 随机延迟
        time.sleep(max(0.5, 3 + random.uniform(-1.0, 1.0)))
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
        """使用智能混合策略自动处理验证码：优先文本匹配，对任何未匹配项启用相似度匹配。"""
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

        # --- 2. 识别所有相关文字 ---
        self.logger.info("--- 步骤 1: 识别 'ques' 图片和检测区域 ---")
        ques_data = [{'index': i, 'image': img} for i, img in enumerate(ques_images)]
        
        for item in ques_data:
            # 优先使用 ImageMatcher 识别 ques 图片
            if self.image_matcher.dictionary:
                char, score = self.image_matcher.find_best_match(
                    item['image'], self.settings.image_matcher.default_weights
                )
                if char and score >= self.settings.image_matcher.min_match_score:
                    item['char'] = char
                    item['score'] = score
                    item['recognized_by_imagematcher'] = True # NEW: Mark as recognized by ImageMatcher
                    self.logger.info(f"Ques {item['index']}: ImageMatcher 识别结果 -> '{item['char']}' (得分: {item['score']:.4f})")
                    continue # 成功匹配，跳过OCR回退
                else:
                    self.logger.warning(f"Ques {item['index']}: ImageMatcher 失败或得分过低 ({score:.4f} < {self.settings.image_matcher.min_match_score:.4f})，回退到 OCR 识别。")
            else:
                self.logger.warning(f"Ques {item['index']}: ImageMatcher 字典未加载，直接使用 OCR 识别。")
            
            # 回退到 OCR 识别
            text, score = self.ocr_recognizer.recognize(item['image'])
            item['char'] = text[0] if text else '' # Only take the first char for ques images
            item['score'] = score
            self.logger.info(f"Ques {item['index']}: OCR 识别结果 -> '{item['char']}' (置信度: {item['score']:.2f}, 原始识别: '{text}')")

        for i, det in enumerate(detections):
            det['det_index'] = i
            det['char'], det['score'] = self.ocr_recognizer.recognize(main_image.crop(det['bbox']))
            self.logger.debug(f"Det {det['det_index']}: 文本识别结果 -> '{det['char']}' (置信度: {det['score']:.2f})")

        # --- 3. 优先执行文本匹配 ---
        self.logger.info("--- 步骤 2: 优先执行文本匹配 ---")
        final_coords = [None] * len(ques_data)
        matched_det_indices = set() # 追踪已被匹配的检测区域的原始索引
        
        min_ocr_confidence_for_text_match = getattr(self.settings.ocr, self.settings.ocr.engine).min_auto_confidence # Use the same confidence for text matching

        available_dets_map = defaultdict(list)
        for det in detections:
            # Only consider detections with single char and sufficient confidence for text matching
            if det['char'] and len(det['char']) == 1 and det['score'] >= min_ocr_confidence_for_text_match:
                available_dets_map[det['char']].append(det)

        for ques_item in ques_data:
            char_to_find = ques_item['char']
            is_match_found = False
            
            # Determine appropriate confidence threshold for text matching
            confidence_meets_threshold = False
            if char_to_find and len(char_to_find) == 1:
                if ques_item.get('recognized_by_imagematcher', False):
                    # If recognized by ImageMatcher, it already passed ImageMatcher's min_match_score.
                    # So, it's considered good enough for text matching.
                    confidence_meets_threshold = True 
                else:
                    # If not recognized by ImageMatcher (i.e., by OCR or failed both), 
                    # then apply the OCR confidence threshold.
                    confidence_meets_threshold = (ques_item['score'] >= min_ocr_confidence_for_text_match)
                
                if confidence_meets_threshold:
                    if available_dets_map[char_to_find]:
                        sorted_dets = sorted(available_dets_map[char_to_find], key=lambda d: d['center'][0])
                        for det_candidate in sorted_dets:
                            if det_candidate['det_index'] not in matched_det_indices:
                                final_coords[ques_item['index']] = det_candidate['center']
                                matched_det_indices.add(det_candidate['det_index'])
                                available_dets_map[char_to_find].remove(det_candidate)
                                self.logger.info(f"文本匹配: Ques {ques_item['index']} ('{char_to_find}') -> 坐标 {det_candidate['center']}")
                                is_match_found = True
                                break
            
            if not is_match_found:
                # If text matching was not even attempted (e.g., ques OCR was bad) or it failed
                pass # This ques_item will remain None in final_coords and be picked up by similarity matching


        # --- 4. 对所有未匹配项进行相似度匹配 ---
        unmatched_ques_for_similarity = []
        for ques_item in ques_data:
            if final_coords[ques_item['index']] is None:
                unmatched_ques_for_similarity.append(ques_item)
        
        if unmatched_ques_for_similarity:
            self.logger.info(f"--- 步骤 3: 文本匹配未完成，对所有 {len(ques_data)} 个字符执行全局高级匹配 ---")
            
            # --- 全局高级匹配逻辑 ---
            # 策略：一旦需要高级匹配，就对所有图片进行全局最优分配，覆盖之前的文本匹配结果。
            main_char_images_to_match = [main_image.crop(det['bbox']) for det in detections]
            ques_char_images_to_match = ques_images # 使用所有原始ques图片

            if main_char_images_to_match and ques_char_images_to_match:
                self.logger.info(f"准备进行全局高级匹配: {len(ques_char_images_to_match)} 个 ques 图片 vs {len(main_char_images_to_match)} 个检测区域。")
                self.logger.info("调用 ImageMatcher 的 find_best_matches_for_main_image 进行匹配...")
                
                matches = self.image_matcher.find_best_matches_for_main_image(
                    main_char_images=main_char_images_to_match,
                    ques_char_images=ques_char_images_to_match,
                    weights=self.settings.image_matcher.default_weights
                )

                # 重置 final_coords，因为全局匹配的结果将完全覆盖它
                final_coords = [None] * len(ques_data)
                
                # 注意：这里的 main_idx 和 ques_idx 分别对应 `detections` 和 `ques_images` 列表的索引
                for main_idx, ques_idx, score in matches:
                    # 如果当前 ques_idx 已经有更高分的匹配，则跳过（不太可能发生，因为匈牙利算法已找到最优解）
                    if final_coords[ques_idx] is not None:
                        continue

                    matched_center = detections[main_idx]['center']
                    final_coords[ques_idx] = matched_center
                    self.logger.info(f"全局高级匹配: Ques {ques_idx} -> 坐标 {matched_center} (得分: {score:.4f})")
            else:
                self.logger.warning("未能为待匹配项生成有效的图像列表，跳过高级匹配。")
            # --- 全局高级匹配逻辑结束 ---

        # --- 5. 最终验证 ---
        # Debugging: Draw and save click points if enabled
        if settings.save_debug_images:
            debug_output_path = os.path.join(settings.paths.base_dir, settings.paths.debug_output_dir)
            if not os.path.exists(debug_output_path):
                os.makedirs(debug_output_path)
            
            self.logger.info(f"保存调试图片到: {debug_output_path}")
            # Ensure final_coords does not contain None before drawing
            valid_coords = [p for p in final_coords if p is not None]
            annotated_image = image_processor.draw_points_on_image(main_image, valid_coords, ques_images=ques_images)
            
            timestamp = int(time.time())
            filename = f"debug_image_{timestamp}.png"
            save_path = os.path.join(debug_output_path, filename)
            
            annotated_image.save(save_path)
            self.logger.info(f"调试图片已保存至: {save_path}")

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
        """使用手动模式处理验证码。(根据环境切换GUI/CLI)"""
        image_urls = self.geetest.extract_image_urls(load_data)
        
        main_image = image_processor.download_image(self.session, image_urls["main_img"])
        ques_images = [img for url in image_urls.get("ques_imgs", []) if (img := image_processor.download_image(self.session, url))]
        
        if not main_image:
            self.logger.error("手动模式：无法下载主验证码图片。" )
            return {"success": False, "error": "Manual mode: Failed to download main captcha image.", "mode": "manual"}
        
        # Determine if running in a headless Linux environment
        is_headless_linux = (sys.platform == 'linux' or sys.platform == 'linux2') and not os.environ.get('DISPLAY')

        if is_headless_linux:
            self.logger.info("检测到无头Linux环境，切换到CLI手动输入模式。" )
            # Print URLs for the user
            print("\n请在浏览器中打开以下链接查看验证码图片。" )
            print(f"\n主图 URL:\n{image_urls['main_img']}\n" )
            if image_urls.get("ques_imgs"):
                print("目标文字图片 URL:")
                for i, url in enumerate(image_urls["ques_imgs"]):
                    print(f"  {i+1}: {url}")
            
            # Call CLI input function
            user_raw_coords, passtime = manual_fallback.get_user_input_cli(num_points=len(ques_images))
            
            if not user_raw_coords:
                return {"success": False, "error": "User did not provide input in CLI mode.", "mode": "manual"}
            
            # Convert raw pixel coords to Geetest format
            user_coords = coordinate_utils.convert_to_geetest_format(
                user_raw_coords, container_size=(main_image.width, main_image.height)
            )

        else:
            self.logger.info("检测到GUI环境或非Linux系统，使用GUI手动输入模式。" )
            # Call GUI input function
            user_coords, passtime = manual_fallback.get_user_input_gui(
                main_image=main_image,
                ques_images=ques_images,
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
                self.logger.warning(f"自动模式连续失败 {self.consecutive_auto_failures} 次。" )
                if self.consecutive_auto_failures >= self.settings.mode_switch.max_auto_failures:
                    self.current_mode = "manual"
                    self.logger.error(f"自动模式失败达到阈值，切换到手动模式。" )
                    self.consecutive_auto_failures = 0
        
        elif self.current_mode == "manual":
            if success:
                self.consecutive_manual_successes += 1
                self.logger.info(f"手动模式连续成功 {self.consecutive_manual_successes} 次。" )
                if self.consecutive_manual_successes >= self.settings.mode_switch.min_success_for_switch:
                    self.current_mode = "auto"
                    self.logger.info(f"手动模式成功达到阈值，切换回自动模式。" )
                    self.consecutive_manual_successes = 0
            else:
                self.consecutive_manual_successes = 0

