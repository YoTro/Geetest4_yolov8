import os
import cv2
import paddle
import numpy as np
import logging
from PIL import Image
from config import settings
from paddle.inference import Config, create_predictor
import yaml
# Since we are loading a training model, we need the model-building components
from libs.ppocr.modeling.architectures import build_model
from libs.ppocr.utils.load_static_weights import load_static_weights

# --- New utility function for similarity calculation ---
def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

class PaddleRecognizer:
    def __init__(self, model_name=None):
        self.logger = logging.getLogger(__name__)
        self.ocr_cfg = settings.ocr.paddle
        self.path_cfg = settings.paths

        # Keep the original inference engine for the 'recognize' method
        try:
            inference_model_dir = model_name if model_name else self.ocr_cfg.inference_model_dir
            self.recognizer = PaddleRecInfer(
                model_dir=inference_model_dir,
                dict_path=self.ocr_cfg.char_dict_path,
                image_shape=(3, 64, 64),
                use_gpu=self.ocr_cfg.use_gpu,
            )
        except Exception as e:
            self.logger.error(f"PaddleRecognizer: Failed to initialize standard inference engine: {e}", exc_info=True)
            self.recognizer = None

        # Placeholder for the feature extraction model, loaded on demand
        self.feature_extractor = None
        self.feature_extractor_checkpoint = self.ocr_cfg.trained_model_dir # e.g., "runs/paddle_train/best_model"

    def _load_feature_extractor(self):
        """Lazy-loads the model for feature extraction."""
        self.logger.info("--- 正在加载用于特征提取的 PaddleOCR 模型 ---")
        try:
            # Load the architecture from the template config
            template_path = os.path.join(self.path_cfg.base_dir, "config", "paddle_ocr_template.yml")
            with open(template_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # --- Critical Change: Remove the Head to get features ---
            config['Architecture']['Head'] = None
            
            # Fill in required global settings for model building
            config['Global'] = {
                'character_dict_path': os.path.join(self.path_cfg.base_dir, self.ocr_cfg.char_dict_path),
                'use_space_char': self.ocr_cfg.use_space_char,
                'max_text_length': self.ocr_cfg.max_text_length
            }

            model = build_model(config['Architecture'])

            # Load the trained weights from the checkpoint
            checkpoint_path = os.path.join(self.path_cfg.base_dir, self.feature_extractor_checkpoint, "student")
            if not os.path.exists(f"{checkpoint_path}.pdparams"):
                 checkpoint_path = os.path.join(self.path_cfg.base_dir, self.feature_extractor_checkpoint, "best_accuracy")
            
            self.logger.info(f"从以下位置加载权重: {checkpoint_path}")
            load_static_weights(model, f"{checkpoint_path}.pdparams")
            
            model.eval()
            self.feature_extractor = model
            self.logger.info("--- 特征提取模型加载成功 ---")
        except Exception as e:
            self.logger.error(f"加载特征提取模型失败: {e}", exc_info=True)
            self.feature_extractor = None

    def get_embedding(self, image):
        """
        获取图像的特征向量 (embedding)。
        image: PIL Image or BGR numpy.ndarray
        """
        if self.feature_extractor is None:
            self._load_feature_extractor()
            if self.feature_extractor is None:
                return None # Failed to load

        # Preprocess the image
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Use the same preprocessing as the inference engine
        preprocessed_img = self.recognizer._resize_norm_img(image)
        img_tensor = paddle.to_tensor(np.expand_dims(preprocessed_img, axis=0))

        # Get feature embedding
        with paddle.no_grad():
            features = self.feature_extractor(img_tensor)
        
        # Global average pooling to get a fixed-size vector from sequence
        # The output of the Neck is typically [batch, sequence_length, feature_dim]
        embedding = paddle.mean(features, axis=1).numpy()
        
        return embedding

    def recognize(self, image):
        if self.recognizer is None:
            return ""
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
        image_shape=(3, 64, 64),
        use_gpu=True,
    ):
        logger = logging.getLogger(__name__) # Get logger
        self.image_shape = image_shape
        
        try:
            self.decoder = CTCLabelDecoder(dict_path)
            if not self.decoder.character: # Check if decoder failed to init
                raise RuntimeError("CTCLabelDecoder 初始化失败，无法继续。")

            model_file = os.path.join(model_dir, "inference.pdmodel")
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