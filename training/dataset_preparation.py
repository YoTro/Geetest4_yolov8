"""
数据集准备、处理、分割和增强的工具函数。
"""
import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from PIL import Image

from utils import data_augmentation, label_generator
from config import settings

def _create_directory_structure(output_dir: Path):
    """为数据集创建标准的YOLO目录结构"""
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

def _split_and_copy_files(
    image_files: List[Path],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float
) -> Dict[str, int]:
    """将文件分割并复制到 train/val/test 目录"""
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    for split_name, files in splits.items():
        image_dest_dir = output_dir / "images" / split_name
        label_dest_dir = output_dir / "labels" / split_name
        for img_path in tqdm(files, desc=f"Copying {split_name} set"):
            label_path = img_path.parent / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(img_path, image_dest_dir)
                shutil.copy2(label_path, label_dest_dir)
    
    return {name: len(files) for name, files in splits.items()}

def augment_training_set(
    dataset_dir: Path,
    target_multiplier: int
) -> int:
    """对训练集进行数据增强。"""
    train_image_dir = dataset_dir / "images" / "train"
    train_label_dir = dataset_dir / "labels" / "train"
    image_files = list(train_image_dir.glob("*.png")) + list(train_image_dir.glob("*.jpg"))
    
    if not image_files or target_multiplier <= 1:
        return 0

    aug_pipeline = data_augmentation.get_yolo_augmentation_pipeline(training=True)
    augmented_count = 0
    
    num_to_generate = (target_multiplier - 1) * len(image_files)

    for i in tqdm(range(num_to_generate), desc="Augmenting data"):
        image_path = random.choice(image_files)
        label_path = train_label_dir / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            continue
            
        img = np.array(Image.open(image_path).convert("RGB"))
        annotations = label_generator.load_yolo_label_file(label_path)
        
        if not annotations:
            continue
            
        bboxes = [ann[1:] for ann in annotations]
        class_labels = [ann[0] for ann in annotations]
        
        try:
            transformed = aug_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
            
            aug_img_name = f"{image_path.stem}_aug_{i}.png"
            aug_label_name = f"{image_path.stem}_aug_{i}.txt"
            
            # 将张量转回可保存的图像格式
            transformed_image_np = transformed['image'].permute(1, 2, 0).numpy()
            transformed_image_np = np.clip(transformed_image_np * 255, 0, 255).astype(np.uint8)
            Image.fromarray(transformed_image_np).save(train_image_dir / aug_img_name)
            
            aug_annotations = [(label, *bbox) for label, bbox in zip(transformed['class_labels'], transformed['bboxes'])]
            label_generator.save_yolo_label_file(train_label_dir / aug_label_name, aug_annotations)
            
            augmented_count += 1
        except Exception:
            # Augmentation can fail if all bboxes are removed
            pass
            
    return augmented_count

def prepare_dataset_from_source(
    source_dir: str,
    dataset_dir: str,
    train_ratio: float,
    val_ratio: float,
    augment_multiplier: int
):
    """从原始数据源准备YOLO数据集的完整流程。"""
    source_path = Path(source_dir)
    dataset_path = Path(dataset_dir)
    
    print("Step 1: Creating directory structure...")
    _create_directory_structure(dataset_path)

    print("Step 2: Finding and copying valid image-label pairs...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 假设源目录中包含images和labels两个子目录
    source_images_dir = source_path / "images"
    source_labels_dir = source_path / "labels"
    
    valid_image_files = []
    if source_images_dir.exists() and source_labels_dir.exists():
        for img_path in source_images_dir.glob("*"):
            if img_path.suffix.lower() in image_extensions:
                label_path = source_labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    valid_image_files.append(img_path)
    
    print(f"Found {len(valid_image_files)} valid image-label pairs.")
    if not valid_image_files:
        print("Error: No valid pairs found. Ensure source directory has 'images' and 'labels' subdirectories.")
        return

    print("Step 3: Splitting dataset...")
    split_stats = _split_and_copy_files(valid_image_files, dataset_path, train_ratio, val_ratio)
    print(f"Dataset split: Train={split_stats['train']}, Val={split_stats['val']}, Test={split_stats['test']}")

    if augment_multiplier > 1:
        print("Step 4: Augmenting training set...")
        augmented_count = augment_training_set(dataset_path, augment_multiplier)
        print(f"Generated {augmented_count} new augmented images.")
        
    print("Step 5: Creating dataset.yaml file...")
    # 使用来自全局配置的类别名称
    class_names = settings.dataset.names
    
    label_generator.create_dataset_yaml(
        output_path=dataset_path / "dataset.yaml",
        dataset_root=str(dataset_path),
        class_names=class_names
    )
    
    print("\nDataset preparation complete!")