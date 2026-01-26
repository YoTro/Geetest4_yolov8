import os
import cv2
import paddle
import numpy as np
import logging # Import logging
from PIL import Image
from config import settings
from paddle.inference import Config, create_predictor


class PaddleRecognizer:
    def __init__(self, model_name=None):
        logger = logging.getLogger(__name__) # Get logger
        ocr_cfg = settings.ocr.paddle

        model_dir = model_name if model_name else ocr_cfg.model_dir
        
        try:
            self.recognizer = PaddleRecInfer(
                model_dir=model_dir,
                dict_path=ocr_cfg.char_dict_path,
                image_shape=(3, 48, 64),
                use_gpu=ocr_cfg.use_gpu,
            )
        except Exception as e:
            logger.error(f"PaddleRecognizer 初始化失败: {e}", exc_info=True)
            self.recognizer = None # Set to None if initialization fails

    def recognize(self, image):
        if self.recognizer is None:
            return "" # Return empty string if recognizer is not initialized
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.recognizer.recognize(image)

class CTCLabelDecoder:
    """
    PaddleOCR 风格 CTC 解码器（支持中文）
    """
    def __init__(self, dict_path, use_space_char=True):
        logger = logging.getLogger(__name__) # Get logger
        try:
            if not os.path.exists(dict_path):
                raise FileNotFoundError(f"字符字典文件未找到: {dict_path}")
            with open(dict_path, "r", encoding="utf-8") as f:
                self.character = [line.strip("\n") for line in f]

            if use_space_char:
                self.character.append(" ")

            self.blank_idx = 0
            self.idx2char = [""] + self.character
        except Exception as e:
            logger.error(f"CTCLabelDecoder 初始化失败 (字典文件: {dict_path}): {e}", exc_info=True)
            self.character = [] # Ensure it's an empty list to prevent further errors
            self.idx2char = [""]
            self.blank_idx = 0


    def decode(self, preds):
        """
        preds: [T, C] softmax 后
        """
        if not self.character: # Check if decoder was initialized successfully
            return ""

        pred_idxs = preds.argmax(axis=1)

        last_idx = -1
        result = []
        for idx in pred_idxs:
            if idx != self.blank_idx and idx != last_idx:
                result.append(self.idx2char[idx])
            last_idx = idx
        return "".join(result)


class PaddleRecInfer:
    """
    直接使用 paddle.inference 的 Rec-only 推理器
    """
    def __init__(
        self,
        model_dir,
        dict_path,
        image_shape=(3, 48, 160),
        use_gpu=True,
    ):
        logger = logging.getLogger(__name__) # Get logger
        self.image_shape = image_shape
        
        try:
            self.decoder = CTCLabelDecoder(dict_path)
            if not self.decoder.character: # Check if decoder failed to init
                raise RuntimeError("CTCLabelDecoder 初始化失败，无法继续。")

            model_file = os.path.join(model_dir, "inference.json")
            params_file = os.path.join(model_dir, "inference.pdiparams")

            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Paddle 推理模型文件未找到: {model_file}")
            if not os.path.exists(params_file):
                raise FileNotFoundError(f"Paddle 推理参数文件未找到: {params_file}")

            config = Config(model_file, params_file)

            if use_gpu == "GPU": # Assuming use_gpu can be "GPU" or "CPU"
                config.enable_use_gpu(200, 0)
            else:
                config.disable_gpu()

            config.enable_memory_optim()
            config.switch_ir_optim(True)

            self.predictor = create_predictor(config)
            self.input_name = self.predictor.get_input_names()[0]
            self.output_name = self.predictor.get_output_names()[0]
        except Exception as e:
            logger.error(f"PaddleRecInfer 初始化失败 (模型目录: {model_dir}): {e}", exc_info=True)
            self.predictor = None # Set predictor to None to indicate failure

    def _resize_norm_img(self, img):
        """
        等价于 PaddleOCR 的 RecResizeImg
        """
        imgC, imgH, imgW = self.image_shape

        h, w = img.shape[:2]
        ratio = w / float(h)
        new_w = int(imgH * ratio)
        new_w = min(new_w, imgW)

        resized = cv2.resize(img, (new_w, imgH))
        padded = np.zeros((imgH, imgW, 3), dtype=np.uint8)
        padded[:, :new_w, :] = resized

        padded = padded.astype("float32") / 255.0
        padded = (padded - 0.5) / 0.5
        padded = padded.transpose(2, 0, 1)
        return padded

    def recognize(self, image):
        """
        image: BGR numpy.ndarray
        """
        if self.predictor is None: # Check if predictor was initialized successfully
            return ""

        if image is None:
            return ""

        if len(image.shape) == 3 and image.shape[2] == 3:
            img = image
        else:
            raise ValueError("输入必须是 BGR 三通道图像")

        img = self._resize_norm_img(img)
        img = np.expand_dims(img, axis=0)

        input_tensor = self.predictor.get_input_handle(self.input_name)
        input_tensor.copy_from_cpu(img)

        self.predictor.run()

        output_tensor = self.predictor.get_output_handle(self.output_name)
        preds = output_tensor.copy_to_cpu()

        preds = preds[0]  # [T, C]
        return self.decoder.decode(preds)
if __name__ == "__main__":
    pass