"""
数据增强工具模块，基于 Albumentations 库。
提供用于YOLO模型训练和验证的标准化数据增强流水线。
"""
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List

def get_yolo_augmentation_pipeline(
    img_size: Tuple[int, int] = (640, 640),
    training: bool = True
) -> A.Compose:
    """
    获取YOLO风格的数据增强流水线。
    
    Args:
        img_size: 目标图像尺寸 (height, width)
        training: 是否为训练模式。True会应用多种增强，False只进行缩放和转换。
        
    Returns:
        一个Albumentations的Compose对象。
    """
    if training:
        # 为训练模式定义一个强大的增强流水线
        transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.GaussNoise(p=0.1),
            A.Blur(blur_limit=3, p=0.1),
            A.Cutout(num_holes=8, max_h_size=40, max_w_size=40, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.1
        ))
    else:
        # 验证/测试模式：只进行缩放、归一化和张量转换
        transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    return transform

def apply_augmentation(
    image: np.ndarray,
    bboxes: List[List[float]],
    class_labels: List[int],
    transform: A.Compose
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    对单个图像和其边界框应用数据增强。
    
    Args:
        image: 输入图像 (H, W, C)
        bboxes: 边界框列表，YOLO格式
        class_labels: 类别标签列表
        transform: Albumentations增强流水线
        
    Returns:
        增强后的图像、边界框和标签
    """
    transformed = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )
    
    return (
        transformed['image'],
        transformed['bboxes'],
        transformed['class_labels']
    )
