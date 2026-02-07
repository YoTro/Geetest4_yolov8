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
from utils.proxy_manager import get_proxies
from . import yolo_inference, manual_fallback
from .trocr_recognizer import TrOCRRecognizer
from .paddle_recognizer import PaddleRecognizer
from .matcher import ImageMatcher # NEW: 导入 ImageMatcher


class CaptchaProcessor:
    """
    验证码处理器，作为核心业务流程的 orchestrator。
    - 管理自动（YOLO）和手动处理模式。
    - 内置错误计数和模式切换逻辑。
    """
    def __init__(self, session: Optional[requests.Session] = None, proxy_source: Optional[str] = None):
        """
        初始化验证码处理器。
        """
        self.settings = settings
        self.session = session or requests.Session()
        self.logger = logging.getLogger(__name__)

        # NEW: 添加代理配置
        if proxy_source:
            self.logger.info(f"正在从 '{proxy_source}' 加载代理...")
            proxies = get_proxies(proxy_source)
            if proxies:
                proxy = random.choice(proxies)
                self.session.proxies = {
                    "http": proxy,
                    "https": proxy,
                }
                self.logger.info(f"已成功配置代理: {proxy}")
            else:
                self.logger.warning("提供了代理来源，但未能加载任何代理。将不使用代理。")

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
        self.consecutive_manual_successes = 0 # NEW: 初始化手动模式连续成功计数
        self.current_mode = "auto" # NEW: 初始模式设置为自动
        
        # NEW: 统计相关
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0

    def batch_process(self, num_times: int = 1, captcha_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        批量处理验证码。
        """
        self.logger.info(f"开始批量处理 {num_times} 次验证码...")
        results = []
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0

        for i in range(num_times):
            self.total_attempts += 1
            self.logger.info(f"--- 批量处理: 第 {i+1}/{num_times} 次尝试 ---")
            result = self.process(captcha_id=captcha_id, **kwargs)
            results.append(result)
            
            if result['success']:
                self.successful_attempts += 1
                self.logger.info(f"第 {i+1} 次处理成功。")
            else:
                self.failed_attempts += 1
                self.logger.warning(f"第 {i+1} 次处理失败。")
            
            time.sleep(random.uniform(self.settings.geetest.min_batch_interval, self.settings.geetest.max_batch_interval))
        
        self.logger.info(f"批量处理完成。共处理 {self.total_attempts} 次。")
        if self.total_attempts > 0:
            success_rate = (self.successful_attempts / self.total_attempts) * 100
            failure_rate = (self.failed_attempts / self.total_attempts) * 100
            self.logger.info(f"统计结果: 成功 {self.successful_attempts} 次, 失败 {self.failed_attempts} 次。")
            self.logger.info(f"成功率: {success_rate:.2f}%, 失败率: {failure_rate:.2f}%.")
        return results

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
        """
        使用模块化策略自动处理验证码。
        该方法作为调度器，依次调用各个处理阶段。
        """
        if self.yolo_model is None: return {"success": False, "error": "YOLO model not loaded.", "mode": "auto"}
        if self.ocr_recognizer is None: return {"success": False, "error": "Recognizer not initialized.", "mode": "auto"}

        # 步骤 1: 数据准备 (下载图片, YOLO检测)
        main_image, ques_images, detections = self._prepare_data(load_data)
        if main_image is None or ques_images is None or detections is None:
            error_msg = "Data preparation failed: Could not acquire necessary images or detections."
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "mode": "auto"}

        # 步骤 2: 实体识别 (识别问题文字和检测框文字)
        ques_data, detections = self._recognize_entities(main_image, ques_images, detections)

        # 步骤 3: 坐标求解 (使用混合策略匹配坐标)
        final_coords = self._solve_coordinates(main_image, ques_images, detections, ques_data)
        if final_coords is None or any(c is None for c in final_coords):
            error_msg = f"Coordinate solving failed. Could not find coordinates for all targets. Coords: {final_coords}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "mode": "auto"}

        # 步骤 4: 提交验证
        return self._submit_solution(final_coords, load_data, main_image, ques_images)

    def _prepare_data(self, load_data: Dict[str, Any]) -> Tuple[Optional[Any], Optional[List[Any]], Optional[List[Dict]]]:
        """
        步骤 1: 数据准备。下载图片并运行YOLO检测。
        返回 (main_image, ques_images, detections)，任何失败则返回 None。
        """
        self.logger.info("--- 自动处理 步骤 1: 数据准备 ---")
        image_urls = self.geetest.extract_image_urls(load_data)
        if not image_urls.get("ques_imgs"):
            self.logger.error("No 'ques_imgs' found in captcha data.")
            return None, None, None

        main_image = image_processor.download_image(self.session, image_urls["main_img"])
        if main_image is None:
            self.logger.error("Failed to download main captcha image.")
            return None, None, None

        ques_images = [image_processor.download_image(self.session, url) for url in image_urls["ques_imgs"]]
        if any(img is None for img in ques_images):
            self.logger.error("Failed to download one or more question images.")
            return None, None, None

        detections = yolo_inference.detect(self.yolo_model, main_image, self.yolo_class_names, self.settings.yolo_inference)
        if not detections:
            self.logger.warning("No objects detected by YOLO model.")
            # 返回空列表而不是None，因为后续步骤可能仍需处理图片
            return main_image, ques_images, [] 
        
        self.logger.info(f"数据准备完成: 1张主图, {len(ques_images)}张问题图, {len(detections)}个检测目标。")
        return main_image, ques_images, detections

    def _recognize_entities(self, main_image: Any, ques_images: List[Any], detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        步骤 2: 实体识别。对所有图片进行文字识别。
        """
        self.logger.info("--- 自动处理 步骤 2: 实体识别 ---")
        ques_data = [{'index': i, 'image': img} for i, img in enumerate(ques_images)]
        
        for item in ques_data:
            # 优先使用 ImageMatcher
            if self.image_matcher.dictionary:
                char, score = self.image_matcher.find_best_match(item['image'], self.settings.image_matcher.default_weights)
                if char and score >= self.settings.image_matcher.min_match_score:
                    item.update({'char': char, 'score': score, 'recognized_by': 'imagematcher'})
                    self.logger.info(f"Ques {item['index']}: ImageMatcher 识别 -> '{item['char']}' (得分: {item['score']:.4f})")
                    continue
                self.logger.warning(f"Ques {item['index']}: ImageMatcher 失败或低分 ({score:.4f})，回退到 OCR。")
            
            # 回退到 OCR
            text, score = self.ocr_recognizer.recognize(item['image'])
            item.update({'char': text[0] if text else '', 'score': score, 'recognized_by': 'ocr'})
            self.logger.info(f"Ques {item['index']}: OCR 识别 -> '{item['char']}' (置信度: {item['score']:.2f})")

        for i, det in enumerate(detections):
            det['det_index'] = i
            det['char'], det['score'] = self.ocr_recognizer.recognize(main_image.crop(det['bbox']))
            self.logger.debug(f"Det {det['det_index']}: OCR 识别 -> '{det['char']}' (置信度: {det['score']:.2f})")
        
        return ques_data, detections

    def _solve_coordinates(self, main_image: Any, ques_images: List[Any], detections: List[Dict], ques_data: List[Dict]) -> Optional[List[Optional[Tuple[int, int]]]]:
        """
        步骤 3: 坐标求解。使用混合策略匹配坐标。
        """
        self.logger.info("--- 自动处理 步骤 3: 坐标求解 ---")
        final_coords = [None] * len(ques_data)
        
        # 策略1: 优先文本匹配
        self.logger.info("求解策略 1: 优先文本匹配。")
        matched_det_indices = set()
        min_ocr_conf = getattr(self.settings.ocr, self.settings.ocr.engine).min_auto_confidence
        
        available_dets_map = defaultdict(list)
        for det in detections:
            if det['char'] and len(det['char']) == 1 and det['score'] >= min_ocr_conf:
                available_dets_map[det['char']].append(det)

        for item in ques_data:
            char, score, source = item['char'], item['score'], item['recognized_by']
            
            passes_threshold = (source == 'imagematcher' and score >= self.settings.image_matcher.min_match_score) or \
                               (source == 'ocr' and score >= min_ocr_conf)

            if char and len(char) == 1 and passes_threshold and available_dets_map[char]:
                sorted_dets = sorted(available_dets_map[char], key=lambda d: d['center'][0])
                for det_candidate in sorted_dets:
                    if det_candidate['det_index'] not in matched_det_indices:
                        final_coords[item['index']] = det_candidate['center']
                        matched_det_indices.add(det_candidate['det_index'])
                        available_dets_map[char].remove(det_candidate)
                        self.logger.info(f"文本匹配成功: Ques {item['index']} ('{char}') -> 坐标 {det_candidate['center']}")
                        break

        # 策略2: 如果有任何未匹配项，则执行全局高级匹配
        if any(c is None for c in final_coords):
            self.logger.info("求解策略 2: 文本匹配未完成，执行全局高级匹配。")
            
            main_char_images = [main_image.crop(det['bbox']) for det in detections]
            if not main_char_images or not ques_images:
                self.logger.warning("未能为高级匹配生成有效图像列表，跳过。")
                return final_coords

            self.logger.info(f"调用 ImageMatcher 进行全局匹配: {len(ques_images)} vs {len(main_char_images)}")
            matches = self.image_matcher.find_best_matches_for_main_image(
                main_char_images=main_char_images,
                ques_char_images=ques_images,
                weights=self.settings.image_matcher.default_weights
            )

            # 全局匹配结果将完全覆盖之前的任何结果
            final_coords = [None] * len(ques_data)
            for main_idx, ques_idx, score in matches:
                if final_coords[ques_idx] is None: # 匈牙利算法确保每个ques只有一个最优匹配
                    final_coords[ques_idx] = detections[main_idx]['center']
                    self.logger.info(f"全局高级匹配: Ques {ques_idx} -> 坐标 {detections[main_idx]['center']} (得分: {score:.4f})")
        
        return final_coords

    def _submit_solution(self, final_coords: List[Tuple[int, int]], load_data: Dict[str, Any], main_image: Any, ques_images: List[Any]) -> Dict[str, Any]:
        """
        步骤 4: 提交验证。
        """
        self.logger.info("--- 自动处理 步骤 4: 提交验证 ---")
        self.logger.info(f"最终确定的点击坐标顺序: {final_coords}")

        # 保存调试图像
        if settings.save_debug_images:
            debug_path = os.path.join(settings.paths.base_dir, settings.paths.debug_output_dir)
            os.makedirs(debug_path, exist_ok=True)
            annotated_image = image_processor.draw_points_on_image(main_image, final_coords, ques_images=ques_images)
            save_path = os.path.join(debug_path, f"debug_image_{int(time.time())}.png")
            annotated_image.save(save_path)
            self.logger.info(f"调试图片已保存至: {save_path}")

        geetest_coords = coordinate_utils.convert_to_geetest_format(final_coords, (main_image.width, main_image.height))
        w_data = self.geetest.generate_w_data(load_data, userresponse=geetest_coords, passtime=int(time.time() * 1000) % 5000 + 2000)
        verify_result = self.geetest.verify(w=w_data['w'], load_data=load_data)

        success = verify_result.get("status") == "success" and verify_result.get("data", {}).get("result") != "fail"
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

        success = verify_result.get("status") == "success" and verify_result.get("result") != "fail"
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

