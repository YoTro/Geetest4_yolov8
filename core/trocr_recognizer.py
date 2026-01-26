"""
TrOCR 模型处理器
封装了TrOCR模型，用于对图像中的文本进行识别。
"""
import torch
import logging # Import logging
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer, GenerationConfig
from typing import Optional
from config.settings import settings, TrOCRTrainingConfig

class TrOCRRecognizer:
    """
    使用TrOCR模型进行文本识别的封装类。
    """
    def __init__(self, model_name: str = "microsoft/trocr-base-stage1", device: Optional[str] = None):
        """
        初始化TrOCR识别器。

        Args:
            model_name (str): Hugging Face Hub上的预训练模型名称。
            device (Optional[str]): 指定运行设备 ('cuda' or 'cpu')。如果为None，则自动检测。
        """
        self.model = None # Initialize model to None
        logger = logging.getLogger(__name__) # Get logger

        try:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            # Workaround for models that fail to load preprocessor_config.json directly
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            except OSError:
                logger.warning("Warning: Could not load image processor from model dir, using default ViT config.")
                self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.processor = TrOCRProcessor(image_processor=self.image_processor, tokenizer=tokenizer)
            
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            # 对于 BERT 类型的解码器，通常使用 cls_token_id (101) 作为起始
            
            self.model.config.decoder_start_token_id = tokenizer.cls_token_id
            self.model.config.pad_token_id = tokenizer.pad_token_id
            
            # 设置最大长度和光束搜索参数（可选）
            self.model.config.max_length = 16
            self.model.config.num_beams = 4
            logger.info(f"TrOCR model '{model_name}' loaded on {self.device}.")
        except Exception as e:
            logger.error(f"TrOCRRecognizer 初始化失败 (模型名称: {model_name}): {e}", exc_info=True)
            self.model = None # Ensure model is None if initialization fails


    def recognize(self, image: Image.Image) -> str:
        """
        识别给定图像中的文本。

        Args:
            image (Image.Image): PIL图像对象。

        Returns:
            str: 识别出的文本。
        """
        if self.model is None:
            logging.getLogger(__name__).warning("TrOCR 模型未加载，无法执行识别。")
            return "" # Return empty string if model is not initialized

        if image.mode != "RGB":
            image = image.convert("RGB")
            
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

if __name__ == '__main__':
    pass # 避免在没有图片时出错
