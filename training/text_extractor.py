"""
使用训练好的YOLO模型，根据有无gt.jsonl标注文件，智能地提取文字区域或单字。
- 如果gt.jsonl存在，则生成带单字标注的字符数据集。
- 如果gt.jsonl不存在，则仅裁剪出图片中的整个文本区域。
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
import cv2
from tqdm import tqdm
import numpy as np

# 确保能导入项目根目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings
from core.yolo_inference import load_model, detect

def _extract_chars_with_labels(model, class_names, yolo_config, input_image_dir, output_base_dir):
    """
    私有函数：当gt.jsonl存在时，提取单字并生成标注。
    """
    print("--- 检测到 gt.jsonl, 执行单字提取与标注任务 ---")
    # 注意：gt.jsonl 和 images/ 目录应该在同一个父目录下
    input_root_dir = Path(input_image_dir).parent
    gt_path = input_root_dir / "gt.jsonl"
    images_path = input_image_dir
    output_images_dir = Path(output_base_dir) / "images"
    output_labels_path = Path(output_base_dir) / "labels.jsonl"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = [json.loads(line) for line in f]
    
    print(f"已加载 {len(gt_data)} 条标注。")

    new_labels = []
    processed_count = 0
    skipped_mismatch = 0

    for item in tqdm(gt_data, desc="生成单字数据"):
        image_filename = item['file_name']
        text_label = item['text']
        image_path = os.path.join(images_path, image_filename)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        results = detect(model, image, class_names, yolo_config)
        boxes = [res['bbox'] for res in results] # Detect函数已排序

        if len(boxes) != len(text_label):
            skipped_mismatch += 1
            continue

        for i, (box, char) in enumerate(zip(boxes, text_label)):
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_char_img = image[y_min:y_max, x_min:x_max]
            
            if cropped_char_img.size == 0:
                continue

            base_name, ext = os.path.splitext(image_filename)
            new_filename = f"{base_name}_char_{i}{ext}"
            new_image_path = output_images_dir / new_filename
            cv2.imwrite(str(new_image_path), cropped_char_img)

            new_labels.append({"file_name": new_filename, "text": char})
        
        processed_count += 1

    with open(output_labels_path, 'w', encoding='utf-8') as f:
        for label in new_labels:
            f.write(json.dumps(label, ensure_ascii=False) + '\\n')

    print("--- 单字数据集生成完成 ---")
    return {
        "success": True, 
        "message": f"处理图片: {processed_count}, 跳过: {skipped_mismatch}, 生成样本: {len(new_labels)}",
        "output_images": str(output_images_dir),
        "output_labels": str(output_labels_path)
    }

def _extract_regions_only(model, class_names, yolo_config, input_image_dir, output_base_dir):
    """
    私有函数：当gt.jsonl不存在时，仅提取文本区域。
    """
    print("--- 未检测到 gt.jsonl, 执行文本区域提取任务 ---")
    images_path = input_image_dir
    output_regions_dir = Path(output_base_dir) / "images"
    output_regions_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in Path(images_path).glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    if not image_files:
        return {"success": False, "error": f"在 {images_path} 中未找到图片文件。"}

    cropped_count = 0
    for img_path in tqdm(image_files, desc="提取文本区域"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        detections = detect(model, image, class_names, yolo_config)

        for i, det in enumerate(detections):
            box = det['bbox']
            x1, y1, x2, y2 = map(int, box)
            cropped_region = image[y1:y2, x1:x2]
            
            if cropped_region.size == 0:
                continue

            cropped_filename = f"{img_path.stem}_region{i}.png"
            cropped_image_path = output_regions_dir / cropped_filename
            cv2.imwrite(str(cropped_image_path), cropped_region)
            cropped_count += 1
            
    return {"success": True, "message": f"成功提取 {cropped_count} 个文本区域，保存在 {output_regions_dir}"}


def extract_text_regions(
    yolo_model_path: str,
    input_image_dir: str,
    output_base_dir: str,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    使用YOLO模型，根据有无gt.jsonl，智能提取文字区域或单字。
    """
    try:
        # 1. 加载YOLO模型 (统一入口)
        print("--- 正在加载YOLO模型 ---")
        yolo_config = settings.yolo_inference
        yolo_config.model_path = os.path.basename(yolo_model_path)
        yolo_config.device = device
        yolo_config.conf = confidence_threshold
        yolo_config.iou = iou_threshold
        path_config = settings.paths

        model, class_names = load_model(yolo_config, path_config)
        if model is None:
            return {"success": False, "error": f"无法从 {yolo_model_path} 加载YOLO模型。"}
        print("--- YOLO模型加载成功 ---")

        # 2. 检查gt.jsonl文件，决定执行哪个任务
        # 假设gt.jsonl和images文件夹在同一个父目录下
        input_root_dir = Path(input_image_dir).parent
        gt_path = input_root_dir / "gt.jsonl"
        
        if gt_path.exists():
            # 执行带标注的单字提取
            return _extract_chars_with_labels(model, class_names, yolo_config, input_image_dir, output_base_dir)
        else:
            # 执行无标注的区域提取
            return _extract_regions_only(model, class_names, yolo_config, input_image_dir, output_base_dir)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    # 此脚本现在通过 main.py 的命令行界面调用，这里的示例仅供参考
    pass
