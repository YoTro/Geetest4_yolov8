"""
半自动标注工具，使用EasyOCR提供初步识别结果，用户进行确认或修改。
新版支持全自动处理和两种保存模式。
"""
import os
import sys # Import sys for platform check
import shutil
import easyocr
import cv2
from pathlib import Path
from typing import List, Dict, Any
import logging
import argparse
from collections import defaultdict

def run_semi_auto_labeler(
    input_base_dir: str,
    lang_list: List[str] = ['ch_sim', 'en'],
    conf_threshold: float = 0.5,
    save_mode: str = 'single', # 'single' or 'consolidated'
    output_dir: str = 'data/dataset/paddle' # Used for consolidated mode
) -> Dict[str, Any]:
    """
    半自动标注工具。

    Args:
        input_base_dir (str): 包含 'images/{train,val}' 的基础目录。
        lang_list (List[str]): EasyOCR 语言列表。
        conf_threshold (float): EasyOCR 识别置信度阈值。
        save_mode (str): 保存模式: 'single' (一图一文件) 或 'consolidated' (合并为 paddle 格式)。
        output_dir (str): 'consolidated' 模式下的输出目录。

    Returns:
        Dict[str, Any]: 操作结果。
    """
    logger = logging.getLogger(__name__)
    
    # Check for headless environment
    is_headless_env = (sys.platform == 'linux' or sys.platform == 'linux2') and not os.environ.get('DISPLAY')
    try:
        reader = easyocr.Reader(lang_list)
        logger.info(f"--- 启动半自动标注工具 (模式: {save_mode}) ---")

        all_labels = defaultdict(list) # {'train': [('Path(img)', 'label'), ...], 'val': [...]} 
        total_images_processed = 0
        auto_labeled_count = 0
        manually_labeled_count = 0
        skipped_count = 0
        
        consolidated_output_paths = {}

        for split in ["train", "val"]:
            image_dir = Path(input_base_dir) / "images" / split
            label_dir_single = Path(input_base_dir) / "labels" / split
            
            if not image_dir.exists():
                logger.warning(f"图像目录不存在: {image_dir}。跳过 {split} 分割。")
                continue

            if save_mode == 'single':
                label_dir_single.mkdir(parents=True, exist_ok=True)

            image_paths = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if not image_paths:
                logger.info(f"信息: 图像目录 {image_dir} 为空。跳过 {split} 分割。")
                continue
            
            total_images_processed += len(image_paths)

            # --- Pass 1: Auto-labeling ---
            logger.info(f"\n--- 自动标注阶段 ({split} set, 共 {len(image_paths)} 张图片) ---")
            unlabeled_images = [] # stores img_path objects
            for img_path in image_paths:
                try:
                    # For EasyOCR, readtext expects a path or numpy array. It has issues with PIL images directly.
                    img_cv = cv2.imread(str(img_path))
                    if img_cv is None:
                        logger.warning(f"无法读取图片 {img_path.name}，跳过 EasyOCR 处理。")
                        unlabeled_images.append(img_path)
                        continue

                    result = reader.readtext(img_cv) # Pass numpy array
                    # Filter results by confidence and non-empty text, ensuring single character
                    suggestions = [text for (bbox, text, prob) in result if prob > conf_threshold and text.strip()]
                    
                    if len(suggestions) == 1 and len(suggestions[0]) == 1: # Single confident character
                        label = suggestions[0]
                        logger.info(f"自动标注: {img_path.name} -> '{label}'")
                        all_labels[split].append((img_path, label))
                        auto_labeled_count += 1
                    else:
                        unlabeled_images.append(img_path)
                except Exception as e:
                    logger.error(f"EasyOCR 处理图片 {img_path.name} 时出错: {e}")
                    unlabeled_images.append(img_path)
            
            # --- Pass 2: Manual Labeling ---
            if unlabeled_images:
                if is_headless_env:
                    logger.warning(f"无头Linux环境，跳过 {len(unlabeled_images)} 张图片的GUI手动标注。")
                    skipped_count += len(unlabeled_images) # Mark these as skipped from manual
                else:
                    logger.info(f"\n--- 手动标注阶段 ({split} set, {len(unlabeled_images)} 张图片待处理) ---")
                    cv2.namedWindow("Image to Label", cv2.WINDOW_NORMAL) # Create resizable window
                    for img_path in unlabeled_images:
                        img_cv = cv2.imread(str(img_path))
                        if img_cv is None: 
                            logger.warning(f"无法读取图片 {img_path.name}，跳过手动标注。")
                            skipped_count += 1
                            continue

                        cv2.imshow("Image to Label", img_cv)
                        cv2.moveWindow("Image to Label", 10, 10) # Position window
                        
                        print(f"\n图片: {img_path.name}")
                        user_input = input("请输入正确的文字内容 (输入 's' 跳过, 'q' 退出): ").strip()

                        if user_input.lower() == 'q':
                            logger.info("用户选择退出。")
                            cv2.destroyAllWindows()
                            # Summarize collected labels up to this point
                            final_msg = (
                                f"标注流程中断！\n"
                                f"  总共处理图片: {total_images_processed} 张\n"
                                f"  自动标注图片: {auto_labeled_count} 张\n"
                                f"  手动标注图片: {manually_labeled_count} 张\n"
                                f"  跳过图片 (未标注): {skipped_count} 张\n"
                                f"  成功标注图片: {auto_labeled_count + manually_labeled_count} 张\n"
                            )
                            return {"success": True, "message": final_msg}
                        
                        if user_input and user_input.lower() != 's':
                            all_labels[split].append((img_path, user_input))
                            manually_labeled_count += 1
                            logger.info(f"已保存标签: '{user_input}'")
                        else:
                            skipped_count += 1
                            logger.info(f"用户选择跳过 {img_path.name}。")
                    cv2.destroyAllWindows()
            else:
                logger.info(f"'{split}' set 中没有图片需要手动标注。")

        # --- Final Save Step ---
        if save_mode == 'single':
            logger.info("\n--- 正在保存为单个 .txt 文件 (每张图片一个标签文件) ---")
            for split, labels in all_labels.items():
                label_dir = Path(input_base_dir) / "labels" / split
                label_dir.mkdir(parents=True, exist_ok=True)
                for img_path, label in labels:
                    label_file_path = label_dir / f"{img_path.stem}.txt"
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        f.write(label)
            logger.info(f"单个 .txt 文件保存完成。标签文件保存在 '{Path(input_base_dir) / 'labels'}' 目录下。")

        elif save_mode == 'consolidated':
            logger.info("\n--- 正在合并为 PaddleOCR 格式的标签文件 ---")
            consolidated_output_path = Path(output_dir)
            consolidated_output_path.mkdir(parents=True, exist_ok=True)
            
            for split, labels in all_labels.items():
                output_file_path = consolidated_output_path / f"rec_gt_{split}.txt"
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for img_path, label in labels:
                        # Construct relative path from the output dir to the image
                        # This assumes images are in input_base_dir/images/{split}/
                        # The path in rec_gt.txt should be like 'images/train/img.png' relative to input_base_dir
                        
                        relative_img_path_for_label_file = img_path.relative_to(Path(input_base_dir)).as_posix()
                        f.write(f"{relative_img_path_for_label_file}\t{label}\n")
                consolidated_output_paths[split] = str(output_file_path)
                logger.info(f"已生成合并文件: {output_file_path}")

        # --- Final Summary ---
        final_summary_message = (
            f"标注流程完成！\n"
            f"  总共处理图片: {total_images_processed} 张\n"
            f"  自动标注图片: {auto_labeled_count} 张\n"
            f"  手动标注图片: {manually_labeled_count} 张\n"
            f"  跳过图片 (未标注): {skipped_count} 张\n"
            f"  成功标注图片: {auto_labeled_count + manually_labeled_count} 张\n"
        )
        if save_mode == 'consolidated':
            final_summary_message += f"  合并标签文件路径:\n"
            for split, path in consolidated_output_paths.items():
                final_summary_message += f"    - {split}: {path}\n"
        else:
            final_summary_message += f"  标签文件保存在 '{Path(input_base_dir) / 'labels'}' 目录下。\n"

        logger.info(final_summary_message)
        return {"success": True, "message": final_summary_message}

    except Exception as e:
        logger.error(f"半自动标注工具运行失败: {e}", exc_info=True)
        cv2.destroyAllWindows()
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    pass