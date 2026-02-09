"""
半自动标注工具，使用 EasyOCR 提供初步识别结果，用户进行确认或修改。
支持全自动处理和两种保存模式。
"""
import os
import sys
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
    save_mode: str = 'single',
    output_dir: str = 'data/dataset/paddle'
) -> Dict[str, Any]:
    """
    半自动标注工具。

    Args:
        input_base_dir (str): 包含 'images/{train,val}' 的基础目录。
        lang_list (List[str]): EasyOCR 语言列表。
        conf_threshold (float): EasyOCR 识别置信度阈值。
        save_mode (str): 保存模式: 'single' 或 'consolidated'。
        output_dir (str): 'consolidated' 模式下的输出目录。

    Returns:
        Dict[str, Any]: 操作结果。
    """
    logger = logging.getLogger(__name__)
    
    is_headless_env = (sys.platform.startswith('linux')) and not os.environ.get('DISPLAY')
    if is_headless_env:
        logger.warning("检测到无头Linux环境。手动标注GUI将被禁用。\n")

    try:
        reader = easyocr.Reader(lang_list)
        logger.info(f"--- 启动半自动标注工具 (模式: {save_mode}) ---")

        all_labels = defaultdict(list)
        total_images_processed = 0
        auto_labeled_count = 0
        manually_labeled_count = 0
        skipped_count = 0
        
        consolidated_output_paths = {}

        for split in ["train", "val"]:
            image_dir = Path(input_base_dir) / "images" / split
            if not image_dir.exists():
                logger.warning(f"图像目录不存在: {image_dir}。跳过 {split} 分割。\n")
                continue

            image_paths = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if not image_paths:
                logger.info(f"信息: 图像目录 {image_dir} 为空。跳过 {split} 分割。\n")
                continue
            
            total_images_processed += len(image_paths)

            # --- Pass 1: Auto-labeling with EasyOCR ---
            logger.info(f"\n--- 自动标注阶段 ({split} set, 共 {len(image_paths)} 张图片) ---")
            unlabeled_images = []
            for img_path in image_paths:
                try:
                    img_cv = cv2.imread(str(img_path))
                    if img_cv is None:
                        unlabeled_images.append(img_path)
                        continue
                    
                    result = reader.readtext(img_cv)
                    suggestions = [res for res in result if res[2] > conf_threshold and res[1].strip()]
                    
                    if len(suggestions) == 1 and len(suggestions[0][1]) == 1: # Single confident character
                        label = suggestions[0][1]
                        score = suggestions[0][2]
                        logger.info(f"自动标注: {img_path.name} -> '{label}' (置信度: {score:.2f})")
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
                    logger.warning(f"无头Linux环境，跳过 {len(unlabeled_images)} 张图片的GUI手动标注。\n")
                    skipped_count += len(unlabeled_images)
                else:
                    logger.info(f"\n--- 手动标注阶段 ({split} set, {len(unlabeled_images)} 张图片待处理) ---")
                    try:
                        cv2.namedWindow("Image to Label", cv2.WINDOW_NORMAL)
                        for img_path in unlabeled_images:
                            img_cv = cv2.imread(str(img_path))
                            if img_cv is None: 
                                skipped_count += 1
                                continue

                            cv2.imshow("Image to Label", img_cv)
                            cv2.moveWindow("Image to Label", 10, 10)
                            cv2.waitKey(1)
                            
                            print(f"\n图片: {img_path.name}")
                            user_input = input("请输入正确的文字内容 (输入 's' 跳过, 'q' 退出): ").strip()

                            if user_input.lower() == 'q':
                                logger.info("用户选择退出。\n")
                                cv2.destroyAllWindows()
                                return {"success": True, "message": "标注中断。"}
                            
                            if user_input and user_input.lower() != 's':
                                all_labels[split].append((img_path, user_input))
                                manually_labeled_count += 1
                            else:
                                skipped_count += 1
                        cv2.destroyAllWindows()
                    except Exception as e:
                         logger.error(f"手动标注GUI失败: {e}. 在无头系统上请不要手动标注。\n")
                         skipped_count += len(unlabeled_images)
            else:
                logger.info(f"'{split}' set 中没有图片需要手动标注。\n")

        # --- Final Save & Summary ---
        final_summary_message = (
            f"标注流程完成！\n"
            f"  总共发现图片: {total_images_processed} 张\n"
            f"  自动标注图片: {auto_labeled_count} 张\n"
            f"  手动标注图片: {manually_labeled_count} 张\n"
            f"  跳过图片 (未标注): {skipped_count} 张\n"
            f"  成功标注总数: {auto_labeled_count + manually_labeled_count} 张\n"
        )

        if save_mode == 'single':
            label_dir_single = Path(input_base_dir) / "labels"
            label_dir_single.mkdir(parents=True, exist_ok=True)
            for split, labels in all_labels.items():
                for img_path, label in labels:
                    label_file_path = label_dir_single / split / f"{img_path.stem}.txt"
                    label_file_path.parent.mkdir(exist_ok=True)
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        f.write(label)
            final_summary_message += f"  标签文件保存在 '{label_dir_single}' 目录下。\n"

        elif save_mode == 'consolidated':
            # Base directory for all consolidated labels (e.g., output_dir/labels)
            consolidated_labels_root = Path(output_dir) / "labels" 
            consolidated_labels_root.mkdir(parents=True, exist_ok=True) # Ensure base exists

            for split, labels in all_labels.items():
                if not labels: continue
                
                # Create split-specific directory (e.g., output_dir/labels/train)
                split_output_dir = consolidated_labels_root / split
                split_output_dir.mkdir(parents=True, exist_ok=True)

                output_file_path = split_output_dir / f"rec_gt_{split}.txt" # e.g., output_dir/labels/train/rec_gt_train.txt
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for img_path, label in labels:
                        # Image path should be relative to input_base_dir's images folder
                        # Example: images/train/some_image.png\tLabel
                        relative_img_path = img_path.relative_to(Path(input_base_dir)).as_posix()
                        f.write(f"{relative_img_path}\t{label}\n")
                consolidated_output_paths[split] = str(output_file_path)
            
            final_summary_message += f"  合并标签文件路径:\n"
            for split, path in consolidated_output_paths.items():
                final_summary_message += f"    - {split}: {path}\n"
        
        logger.info(final_summary_message)
        return {"success": True, "message": final_summary_message}

    except Exception as e:
        logger.error(f"半自动标注工具运行失败: {e}", exc_info=True)
        if 'is_headless_env' in locals() and not is_headless_env:
            cv2.destroyAllWindows()
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="增强的半自动标注工具，使用EasyOCR进行推荐。" )
    parser.add_argument("--input-dir", required=True, help="包含 'images/{train,val}' 的基础目录。" )
    parser.add_argument("--lang", default="ch_sim,en", help="EasyOCR识别的语言列表，用逗号分隔。" )
    parser.add_argument("--conf-thres", type=float, default=0.5, help="EasyOCR识别结果的置信度阈值。" )
    parser.add_argument("--save-mode", default="single", choices=['single', 'consolidated'], help="保存模式")
    parser.add_argument("--output-dir", default="data/dataset/paddle", help="'consolidated' 模式下的输出目录。" )
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    lang_list = [lang.strip() for lang in args.lang.split(',')]
    
    run_semi_auto_labeler(
        input_base_dir=args.input_dir,
        lang_list=lang_list,
        conf_threshold=args.conf_thres,
        save_mode=args.save_mode,
        output_dir=args.output_dir
    )