import os
import easyocr
import cv2 # EasyOCR uses OpenCV
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

def run_semi_auto_labeler(
    input_base_dir: str,
    lang_list: List[str] = ['ch_sim', 'en'],
    conf_threshold: float = 0.5, # EasyOCR recognition confidence threshold
) -> Dict[str, Any]:
    """
    半自动标注工具，使用EasyOCR提供初步识别结果，用户进行确认或修改。

    Args:
        input_base_dir (str): 包含裁剪图片的基础目录 (e.g., 'data/dataset_trocr').
                                期望结构: input_base_dir/images/{train,val}/img.png
        lang_list (List[str]): EasyOCR识别的语言列表。
        conf_threshold (float): EasyOCR识别结果的置信度阈值。

    Returns:
        Dict[str, Any]: 包含标注结果的字典。
    """
    logger = logging.getLogger(__name__)
    
    try:
        reader = easyocr.Reader(lang_list)
        
        splits = ["train", "val"]
        labeled_count = 0
        skipped_count = 0
        re_labeled_count = 0

        logger.info(f"--- 启动半自动标注工具 ---")
        logger.info(f"标注目录: {input_base_dir}")
        logger.info(f"EasyOCR 语言: {lang_list}")

        for split in splits:
            image_dir = Path(input_base_dir) / "images" / split
            label_dir = Path(input_base_dir) / "labels" / split
            label_dir.mkdir(parents=True, exist_ok=True) # Ensure label directory exists

            if not image_dir.exists():
                logger.warning(f"图像目录不存在: {image_dir}。跳过 {split} 分割。 ולא קיים")
                continue

            image_paths = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if not image_paths:
                logger.info(f"信息: 图像目录 {image_dir} 为空。跳过 {split} 分割。")
                continue
            
            logger.info(f"\n--- 开始标注 {split} 分割 ({len(image_paths)} 张图片) ---")

            for i, img_path in enumerate(image_paths):
                label_file_path = label_dir / f"{img_path.stem}.txt"
                
                existing_text = ""
                if label_file_path.exists():
                    with open(label_file_path, 'r', encoding='utf-8') as f:
                        existing_text = f.read().strip()
                    logger.info(f"图片 {i+1}/{len(image_paths)} - {img_path.name}\n已存在标签: '{existing_text}'。")
                    response = input(f"要重新标注吗？(y/N/s(skip)): ").strip().lower()
                    if response == 'n' or response == '':
                        skipped_count += 1
                        continue
                    elif response == 's':
                        skipped_count += 1
                        continue
                    else:
                        re_labeled_count += 1
                
                # Load image for display
                img_cv = cv2.imread(str(img_path))
                if img_cv is None:
                    logger.warning(f"无法读取图片 {img_path.name}。跳过。")
                    continue
                
                # Perform OCR
                img_for_ocr = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                try:
                    
                    result = reader.readtext(img_for_ocr)
                except ValueError as e:
                    
                    logger.warning(f"RGB 识别失败，尝试灰度图模式: {e}")
                    img_grey = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    result = reader.readtext(img_grey)
                
                # Extract text from OCR result (EasyOCR returns (bbox, text, confidence))
                ocr_suggestion = " ".join([text for (bbox, text, prob) in result if prob > conf_threshold])
                
                # Display image and get user input
                # Use a specific window name to manage it
                window_name = "Image to Label"
                cv2.imshow(window_name, img_cv)
                
                # Move window to top-left to be visible
                cv2.moveWindow(window_name, 10, 10) 
                
                print(f"图片: {img_path.name}")
                print(f"EasyOCR 建议: '{ocr_suggestion}'")
                user_input = input("请输入正确的文字内容 (留空则使用 OCR 建议，输入 's' 跳过，输入 'q' 退出): ").strip()

                if user_input.lower() == 'q':
                    logger.info("用户选择退出。")
                    cv2.destroyAllWindows()
                    return {"success": True, "message": f"标注中断。已标注 {labeled_count} 张图片，跳过 {skipped_count} 张图片，重新标注 {re_labeled_count} 张图片。"}

                if user_input.lower() == 's':
                    skipped_count += 1
                    logger.info("用户选择跳过此图片。")
                elif user_input: # User provided input
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        f.write(user_input)
                    labeled_count += 1
                    logger.info(f"已保存标签: '{user_input}'")
                elif ocr_suggestion: # Use OCR suggestion if user left empty and OCR found something
                    with open(label_file_path, 'w', encoding='utf-8') as f:
                        f.write(ocr_suggestion)
                    labeled_count += 1
                    logger.info(f"已使用 OCR 建议保存标签: '{ocr_suggestion}'")
                else: # No user input, no OCR suggestion
                    skipped_count += 1
                    logger.info("未输入内容且 OCR 无建议，跳过此图片。")
            
            cv2.destroyAllWindows() # Close window after each split
            
        return {"success": True, "message": f"标注完成！共标注 {labeled_count} 张图片，跳过 {skipped_count} 张图片，重新标注 {re_labeled_count} 张图片。"}

    except Exception as e:
        logger.error(f"半自动标注工具运行失败: {e}", exc_info=True)
        cv2.destroyAllWindows()
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    # Example usage:
    # run_semi_auto_labeler(input_base_dir="data/dataset_trocr")
    pass
