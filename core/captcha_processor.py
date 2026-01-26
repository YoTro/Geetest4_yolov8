"""
验证码处理器
整合了验证码处理的完整流程、模式切换和错误管理。
"""
import time
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# 统一从config模块导入全局配置
from config import settings
from .gt4 import GeetestV4
# 导入重构后的函数式模块
from utils import image_processor, coordinate_utils
from . import yolo_inference, manual_fallback
from .trocr_recognizer import TrOCRRecognizer
from .paddle_recognizer import PaddleRecognizer

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
            self.logger.info("初始化 PaddleOCR 识别器...")
            self.ocr_recognizer = PaddleRecognizer(model_name=self.settings.ocr.paddle.model_dir)
        else:
            self.logger.error(f"不支持的OCR引擎: {self.settings.ocr.engine}。请检查config/settings.py。")
            self.ocr_recognizer = None # Ensure it's None if not supported

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
        """使用自动（YOLO+OCR）模式处理验证码。"""
        if self.yolo_model is None:
            return {"success": False, "error": "YOLO model not loaded.", "mode": "auto"}
        if self.ocr_recognizer is None:
            return {"success": False, "error": "OCR recognizer not initialized.", "mode": "auto"}

        image_urls = self.geetest.extract_image_urls(load_data)
        
        # 1. 识别目标文字
        target_chars = []
        if not image_urls.get("ques_imgs"):
            return {"success": False, "error": "No 'ques_imgs' found in captcha data.", "mode": "auto"}

        for ques_url in image_urls["ques_imgs"]:
            ques_image = image_processor.download_image(self.session, ques_url)
            if ques_image is None:
                self.logger.error(f"Failed to download question image: {ques_url}")
                return {"success": False, "error": f"Failed to download question image: {ques_url}", "mode": "auto"}
            
            recognized_char = self.ocr_recognizer.recognize(ques_image)
            if recognized_char:
                target_chars.append(recognized_char)
            else:
                self.logger.warning(f"OCR failed to recognize text from question image: {ques_url}")

        if not target_chars:
            return {"success": False, "error": "OCR failed to recognize any target characters from question images.", "mode": "auto"}
        self.logger.info(f"识别出的目标文字: {target_chars}")

        # 2. 下载主图片并检测所有文字位置
        main_image = image_processor.download_image(self.session, image_urls["main_img"])
        if main_image is None:
            self.logger.error("Failed to download captcha image.")
            return {"success": False, "error": "Failed to download captcha image.", "mode": "auto"}
        
        self.logger.debug(f"Downloaded main image: {main_image.size} {main_image.mode}")

        detections = yolo_inference.detect(self.yolo_model, main_image, self.yolo_class_names, self.settings.yolo_inference)
        self.logger.debug(f"YOLO raw detections: {detections}")
        if not detections:
            return {"success": False, "error": "No objects detected by YOLO model.", "mode": "auto"}

        # 3. 对每个检测到的区域进行OCR识别
        recognized_detections = []
        for det in detections:
            bbox = det['bbox']
            char_image = main_image.crop(bbox)
            recognized_text = self.ocr_recognizer.recognize(char_image)
            
            if recognized_text:
                self.logger.debug(f"OCR recognized '{recognized_text}' for bbox {bbox}")
                det['class_name'] = recognized_text
                recognized_detections.append(det)
            else:
                self.logger.warning(f"OCR failed to recognize text for bbox {bbox}. Skipping this detection.")
        
        if not recognized_detections:
            return {"success": False, "error": "OCR failed to recognize any characters from detected regions.", "mode": "auto"}

        # 4. 计算点击坐标
        click_coords = self._calculate_click_coords(recognized_detections, target_chars)
        
        if len(click_coords) != len(target_chars):
             self.logger.warning(f"目标字符数 ({len(target_chars)}) 与匹配到的坐标数 ({len(click_coords)}) 不匹配。")
             if not click_coords:
                 return {"success": False, "error": "Could not find coordinates for any of the target characters.", "mode": "auto"}

        # 5. 验证
        geetest_coords = coordinate_utils.convert_to_geetest_format(click_coords, (main_image.width, main_image.height))
        w_data = self.geetest.generate_w_data(load_data, userresponse=geetest_coords, passtime=int(time.time() * 1000) % 5000 + 2000)
        verify_result = self.geetest.verify(w=w_data['w'], load_data=load_data)

        success = verify_result.get("status") == "success"
        return {"success": success, "details": verify_result, "mode": "auto"}

    def _process_manual(self, load_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用手动模式处理验证码。"""
        image_urls = self.geetest.extract_image_urls(load_data)
        
        # 获取用户输入
        user_coords, passtime = manual_fallback.get_user_input_with_gui(
            main_image_url=image_urls["main_img"],
            ques_image_urls=image_urls.get("ques_imgs", []),
            session=self.session,
            timeout=self.settings.mode_switch.manual_timeout
        )

        if not user_coords:
            return {"success": False, "error": "User did not provide input or timed out.", "mode": "manual"}
        
        # 验证
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
                    self.consecutive_auto_failures = 0 # 重置计数器
        
        elif self.current_mode == "manual":
            if success:
                self.consecutive_manual_successes += 1
                self.logger.info(f"手动模式连续成功 {self.consecutive_manual_successes} 次。")
                if self.consecutive_manual_successes >= self.settings.mode_switch.min_success_for_switch:
                    self.current_mode = "auto"
                    self.logger.info(f"手动模式成功达到阈值，切换回自动模式。")
                    self.consecutive_manual_successes = 0 # 重置计数器
            else:
                self.consecutive_manual_successes = 0

    def _calculate_click_coords(self, detections: List[Dict], target_chars: List[str]) -> List[Tuple[int, int]]:
        """根据检测结果和目标文字计算点击坐标。"""
        # 按x坐标对检测结果排序，模拟阅读顺序
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

