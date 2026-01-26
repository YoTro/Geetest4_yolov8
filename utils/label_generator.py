"""
标签生成与处理工具函数
"""
import os
import yaml
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

def coords_to_yolo_format(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    将边界框坐标(xmin, ymin, xmax, ymax)转换为YOLO格式的相对坐标(x_center, y_center, width, height)
    """
    img_width, img_height = image_size
    x_min, y_min, x_max, y_max = bbox
    
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return (x_center, y_center, width, height)

def yolo_to_coords(
    yolo_bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    将YOLO格式的相对坐标转换为绝对像素坐标(xmin, ymin, xmax, ymax)
    """
    img_width, img_height = image_size
    x_center, y_center, width, height = yolo_bbox
    
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    
    return (x_min, y_min, x_max, y_max)

def save_yolo_label_file(
    label_path: Union[str, Path],
    annotations: List[Tuple[int, float, float, float, float]]
):
    """
    将YOLO格式的标注保存到.txt文件
    每行为: class_id x_center y_center width height
    """
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_path, 'w', encoding='utf-8') as f:
        for ann in annotations:
            line = f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n"
            f.write(line)

def load_yolo_label_file(label_path: Union[str, Path]) -> List[Tuple[int, float, float, float, float]]:
    """
    从.txt文件加载YOLO格式的标注
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    
    annotations = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.read().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                annotations.append(
                    (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                )
    return annotations

def create_dataset_yaml(output_path: Union[str, Path], dataset_root: str, class_names: List[str]):
    """
    创建并保存YOLOv5/v8格式的dataset.yaml文件
    """
    dataset_root_abs = str(Path(dataset_root).absolute())
    
    data_config = {
        'path': dataset_root_abs,
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'test': os.path.join('images', 'test'),
        'nc': len(class_names),
        'names': class_names
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"数据集配置文件已创建于: {output_path}")

def visualize_yolo_bboxes(
    image_path: Union[str, Path],
    label_path: Union[str, Path],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None
) -> Image.Image:
    """
    可视化图像上的YOLO边界框
    """
    img = Image.open(image_path).convert("RGB")
    annotations = load_yolo_label_file(label_path)
    
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        bbox = yolo_to_coords((x_center, y_center, width, height), (img_width, img_height))
        
        color = colors[class_id % len(colors)]
        draw.rectangle(bbox, outline=color, width=2)
        
        class_name = class_names[class_id]
        
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((bbox[0], bbox[1]), class_name, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], bbox[1]), class_name, fill="white", font=font)
        
    if output_path:
        img.save(output_path)
        
    return img
