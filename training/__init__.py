"""
训练模块
=========

该模块包含了与模型生命周期相关的所有功能，包括：
- 数据集准备与增强
- 模型训练
- 模型验证与评估
- 数据收集
- 文字区域提取 (半自动标注)
- TrOCR模型训练
- 半自动标注工具
"""
from . import dataset_preparation
from . import train_yolo
from . import data_collector
from . import text_extractor
from . import train_trocr
from . import semi_auto_labeler
from . import synthetic_data_generator

__all__ = [
    "dataset_preparation",
    "train_yolo",
    "data_collector",
    "text_extractor",
    "train_trocr",
    "semi_auto_labeler",
    "synthetic_data_generator"
]