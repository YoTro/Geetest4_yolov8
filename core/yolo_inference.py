"""
YOLOv8 模型推理函数
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from config.settings import YOLOInferenceConfig, PathConfig
from utils import image_processor, coordinate_utils

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.getLogger(__name__).error("ultralytics 未安装，YOLO功能将不可用。请运行 'pip install ultralytics'")

def load_model(config: YOLOInferenceConfig, paths: PathConfig):
    logger = logging.getLogger(__name__)
    if not YOLO_AVAILABLE:
        return None, []

    model_path = paths.get_model_path(config.model_path)
    if not Path(model_path).exists():
        logger.error(f"模型文件不存在: {model_path}")
        return None, []

    try:
        logger.info(f"加载YOLO模型: {model_path}")
        device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(model_path)
        model.to(device)
        logger.info(f"模型已加载到设备: {device}")
        logger.debug(f"模型类别名称: {model.names}")

        logger.info("预热模型...")
        dummy_image = np.random.randint(0, 255, (config.imgsz, config.imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            model(dummy_image, verbose=False)
        logger.info("模型预热完成。")

        return model, model.names

    except Exception as e:
        logger.error(f"加载YOLO模型失败: {e}", exc_info=True)
        return None, []

def detect(
    model: 'YOLO',
    image: np.ndarray,
    class_names: Dict[int, str],
    config: YOLOInferenceConfig
) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    try:
        logger.debug(f"Detecting on image size: {image.shape if hasattr(image, 'shape') else image.size}, YOLO input size: {config.imgsz}")
        logger.debug(f"YOLO conf_thres: {config.conf}, iou_thres: {config.iou}")
        results = model(
            image,
            conf=config.conf,
            iou=config.iou,
            imgsz=config.imgsz,
            verbose=False
        )

        detections = []
        for res in results:
            for box in res.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().tolist()
                class_id = int(box.cls[0])
                detections.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'confidence': float(box.conf[0]),
                    'class_id': class_id,
                    'class_name': class_names.get(class_id, str(class_id)),
                    'center': coordinate_utils.bbox_to_center((x_min, y_min, x_max, y_max)),
                })
        
        detections.sort(key=lambda d: d['bbox'][0])
        return detections

    except Exception as e:
        logger.error(f"YOLO检测失败: {e}", exc_info=True)
        return []
