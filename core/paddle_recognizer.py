import os
import cv2
import paddle
import numpy as np
import logging
from PIL import Image
from config import settings
from paddle.inference import Config, create_predictor
from typing import Tuple
import yaml
from libs.ppocr.modeling.architectures import build_model
from libs.ppocr.utils.save_load import load_pretrained_params

class PaddleRecognizer:
    def __init__(self, model_name=None):
        self.logger = logging.getLogger(__name__)
        self.ocr_cfg = settings.ocr.paddle
        self.path_cfg = settings.paths

        try:
            inference_model_dir = model_name if model_name else self.ocr_cfg.inference_model_dir
            self.recognizer = PaddleRecInfer(
                model_dir=inference_model_dir,
                dict_path=self.ocr_cfg.char_dict_path,
                use_gpu=self.ocr_cfg.use_gpu,
            )
        except Exception as e:
            self.logger.error(f"PaddleRecognizer: Failed to initialize standard inference engine: {e}", exc_info=True)
            self.recognizer = None
            
    def recognize(self, image) -> Tuple[str, float]: # Return text and confidence
        if self.recognizer is None: return "", 0.0
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.recognizer.recognize(image)
        
class CTCLabelDecoder:
    def __init__(self, dict_path, use_space_char=True):
        logger = logging.getLogger(__name__)
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                self.character = [line.strip("\n") for line in f]
            if use_space_char: self.character.append(" ")
            self.blank_idx = 0
            self.idx2char = [""] + self.character
        except Exception as e:
            logger.error(f"CTCLabelDecoder 初始化失败: {e}", exc_info=True)
            self.character, self.idx2char, self.blank_idx = [], [""], 0

    def decode(self, preds: np.ndarray) -> Tuple[str, float]: # Return text and confidence
        if not self.character: return "", 0.0
        
        # preds shape: [seq_len, num_classes]
        # Softmax might not be applied, so raw logits can be large.
        # We need probabilities for confidence. Apply softmax if it's not already.
        # However, for confidence based on argmax, just use the value from the logit.
        
        pred_idxs = preds.argmax(axis=1)
        # Use softmax to get probabilities for confidence if preds are logits
        if np.any(preds > 1.0) or np.any(preds < 0.0): # Heuristic to check if they are logits
            probabilities = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
            pred_probs = np.max(probabilities, axis=1) # Get max probability for each time step
        else:
            pred_probs = np.max(preds, axis=1) # Assume they are already probabilities
            
        last_idx = -1
        result = []
        confidences = []

        for i, idx in enumerate(pred_idxs):
            if idx != self.blank_idx and idx != last_idx:
                result.append(self.idx2char[idx])
                confidences.append(pred_probs[i]) # Store confidence for non-blank, non-repeated
            last_idx = idx
        
        if confidences:
            avg_confidence = np.mean(confidences)
        else:
            avg_confidence = 0.0
            
        return "".join(result), float(avg_confidence)

class PaddleRecInfer:
    def __init__(self, model_dir, dict_path, use_gpu=True):
        self.logger = logging.getLogger(__name__)
        self.ocr_cfg = settings.ocr.paddle
        self.image_shape = [3, 32, 32] # Use the correct shape for CRNN

        try:
            self.decoder = CTCLabelDecoder(dict_path, use_space_char=self.ocr_cfg.use_space_char)
            if not self.decoder.character: raise RuntimeError("CTCLabelDecoder 初始化失败。")

            pir_model_path = os.path.join(model_dir, "inference.pirmdl")
            
            if os.path.exists(pir_model_path):
                self.logger.info(f"检测到 PIR 单文件模型，从 {pir_model_path} 加载。\n")
                config = Config(pir_model_path)
            else:
                self.logger.info("未找到 PIR 模型，尝试加载旧版双文件模型。\n")
                model_file = os.path.join(model_dir, "inference.json")
                params_file = os.path.join(model_dir, "inference.pdiparams")
                if not os.path.exists(model_file):
                    model_file = os.path.join(model_dir, "inference.pdmodel")
                if not os.path.exists(model_file) or not os.path.exists(params_file):
                    raise FileNotFoundError(f"在 '{model_dir}' 中找不到有效的推理模型文件对 (.pdmodel/.json + .pdiparams)。")
                
                config = Config(model_file, params_file)

            if use_gpu == "GPU": 
                config.enable_use_gpu(200, 0)
            else: 
                config.disable_gpu()
                config.disable_mkldnn() # For CPU, explicitly disable MKL-DNN (OneDNN) if it's causing issues.
                config.set_cpu_math_library_num_threads(1) 
                # 禁用一些可能导致死锁的优化策略
                config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
            config.switch_ir_optim(True)

            self.predictor = create_predictor(config)
            self.input_name = self.predictor.get_input_names()[0]
            self.output_name = self.predictor.get_output_names()[0]
        except Exception as e:
            self.logger.error(f"PaddleRecInfer 初始化失败 (模型目录: {model_dir}): {e}", exc_info=True)
            self.predictor = None

    def _resize_norm_img(self, img):
        imgC, imgH, imgW = self.image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        new_w = min(int(imgH * ratio), imgW)
        
        resized = cv2.resize(img, (new_w, imgH))
        padded = np.zeros((imgH, imgW, 3), dtype=np.uint8)
        padded[:, :new_w, :] = resized
        
        padded = padded.astype("float32") / 255.0
        padded = (padded - 0.5) / 0.5
        padded = padded.transpose(2, 0, 1)
        return padded

    def recognize(self, image) -> Tuple[str, float]: # Return text and confidence
        if self.predictor is None or image is None: return "", 0.0
        if len(image.shape) != 3 or image.shape[2] != 3: raise ValueError("输入必须是 BGR 三通道图像")

        img = self._resize_norm_img(image)
        img = np.expand_dims(img, axis=0)

        input_tensor = self.predictor.get_input_handle(self.input_name)
        input_tensor.copy_from_cpu(img)
        self.predictor.run()
        output_tensor = self.predictor.get_output_handle(self.output_name)
        preds = output_tensor.copy_to_cpu()[0]
        
        return self.decoder.decode(preds)

if __name__ == "__main__":
    pass
