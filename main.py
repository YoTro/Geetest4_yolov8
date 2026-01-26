"""
极验验证码自动化处理主程序。
提供一个统一的命令行入口来运行不同的任务。
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from PIL import Image

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录和 libs 目录加入 sys.path，确保能正确导入所有模块
# 使用 insert(0, ...) 确保这些路径有最高的导入优先级
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
libs_dir = os.path.join(current_dir, 'libs')
if libs_dir not in sys.path:
    sys.path.insert(0, libs_dir)

from config import settings
from logs import setup_logging
from core.captcha_processor import CaptchaProcessor
from core.trocr_recognizer import TrOCRRecognizer
from core.paddle_recognizer import PaddleRecognizer
from training import train_yolo, dataset_preparation, data_collector, train_trocr, train_paddleocr, text_extractor, semi_auto_labeler
from training.synthetic_data_generator import TrOCRDataGenerator # 导入合成数据生成器

def main():
    """主函数，解析命令行参数并分发任务。"""
    parser = argparse.ArgumentParser(
        description="极验验证码自动化处理系统",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="可执行的命令", required=True)

    # --- 文本识别命令 ---
    recognize_parser = subparsers.add_parser("recognize", help="使用OCR模型识别图像中的文本")
    recognize_parser.add_argument("--image", required=True, help="要识别的图像文件的路径")
    recognize_parser.add_argument("--engine", default="trocr", choices=["trocr", "paddle"], help="要使用的OCR引擎 (默认: trocr)")
    recognize_parser.add_argument("--model", default="microsoft/trocr-base-stage1", help="要使用的TrOCR模型名称 (仅当 engine='trocr' 时有效)")

    # --- 验证命令 ---
    process_parser = subparsers.add_parser("run", help="运行验证码处理器，处理单个验证码")
    process_parser.add_argument("--mode", choices=["auto", "manual"], default="auto", help="处理模式 (默认: auto)")
    process_parser.add_argument("--captcha-id", help=f"要处理的验证码ID (默认: {settings.geetest.captcha_id})")
    process_parser.add_argument("--yolo-model", help=f"要使用的YOLO模型路径 (覆盖config中的设置，默认: {settings.yolo_inference.model_path})")
    process_parser.add_argument("--ocr-engine", choices=["trocr", "paddle"], help=f"要使用的OCR引擎 (覆盖config中的设置，默认: {settings.ocr.engine})")
    process_parser.add_argument("--trocr-model", help=f"要使用的TrOCR模型名称或路径 (覆盖config中的设置，默认: {settings.ocr.trocr.model_name})")
    process_parser.add_argument("--paddle-model-dir", help=f"要使用的PaddleOCR推理模型目录 (覆盖config中的设置，默认: {settings.ocr.paddle.model_dir})")
    
    # --- 数据集准备命令 ---
    prep_parser = subparsers.add_parser("prepare", help="从原始数据文件夹准备YOLOv8数据集")
    prep_parser.add_argument("--source", required=True, help="包含图像和标签的源数据目录")
    prep_parser.add_argument("--output", default=settings.paths.yolo_dataset_dir, help="处理后数据集的输出目录")
    prep_parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集所占比例 (默认: 0.8)")
    prep_parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集所占比例 (默认: 0.1)")
    prep_parser.add_argument("--augment", type=int, default=1, help="数据增强倍数 (1表示不增强)")



    # --- 模型训练命令 (YOLOv8) ---
    train_parser = subparsers.add_parser("train", help="训练一个新的YOLOv8模型")
    train_parser.add_argument("--data", default=settings.paths.get_yolo_dataset_yaml_path(), help="数据集的 data.yaml 配置文件路径")
    train_parser.add_argument("--model", default="yolov8n.pt", help="基础模型 (e.g., yolov8n.pt) 或恢复训练的模型路径")
    train_parser.add_argument("--epochs", type=int, help=f"训练轮数 (默认: {settings.yolo_training.epochs})")
    train_parser.add_argument("--batch", type=int, help=f"批次大小 (默认: {settings.yolo_training.batch})")
    train_parser.add_argument("--imgsz", type=int, help=f"图像尺寸 (默认: {settings.yolo_training.imgsz})")
    train_parser.add_argument("--device", default=settings.yolo_training.device, help="训练设备 (e.g., 'cpu', '0')")

    # --- TrOCR模型训练命令 ---
    train_trocr_parser = subparsers.add_parser("train_trocr", help="训练TrOCR文字识别模型")
    train_trocr_parser.add_argument("--dataset-dir", default=settings.paths.trocr_dataset_dir, help="包含图像和标签的数据集目录")
    train_trocr_parser.add_argument("--epochs", type=int, help=f"训练轮数 (默认: {settings.ocr.trocr.epochs})")
    train_trocr_parser.add_argument("--batch-size", type=int, help=f"批次大小 (默认: {settings.ocr.trocr.batch_size})")
    train_trocr_parser.add_argument("--learning-rate", type=float, help=f"学习率 (默认: {settings.ocr.trocr.learning_rate})")
    train_trocr_parser.add_argument("--model-name", help=f"基础TrOCR模型名称 (默认: {settings.ocr.trocr.model_name})")
    train_trocr_parser.add_argument("--output-dir", help=f"模型输出目录 (默认: {settings.ocr.trocr.output_dir})")
    train_trocr_parser.add_argument("--device", help=f"训练设备 (默认: {settings.ocr.trocr.device})")
    train_trocr_parser.add_argument("--resume", action="store_true", help="是否从上一次的断点恢复训练")

    # --- PaddleOCR模型训练命令 ---
    train_paddle_parser = subparsers.add_parser("train_paddle", help="训练PaddleOCR文字识别模型 (需要PaddleOCR的训练脚本)")
    train_paddle_parser.add_argument("--train-label-file", required=True, help="训练集标签文件的路径 (e.g., 'data/train_list.txt')")
    train_paddle_parser.add_argument("--val-label-file", required=True, help="验证集标签文件的路径 (e.g., 'data/val_list.txt')")
    

    # --- 文本区域提取命令 ---
    extract_parser = subparsers.add_parser("extract_text_regions", help="使用YOLO模型从图像中提取文字区域")
    extract_parser.add_argument("--yolo-model", required=True, help="训练好的YOLO模型路径 (e.g., 'runs/detect/train/weights/best.pt')")
    extract_parser.add_argument("--input-dir", default="data/raw/images", help="包含原始图像的输入目录")
    extract_parser.add_argument("--output-dir", default="data/dataset/trocr", help="保存裁剪图片的输出目录")
    extract_parser.add_argument("--conf-thres", type=float, default=0.5, help="YOLO检测的置信度阈值")
    extract_parser.add_argument("--iou-thres", type=float, default=0.45, help="YOLO检测的IoU阈值")
    extract_parser.add_argument("--device", default="cpu", help="推理设备 ('cpu' 或 'mps'/'cuda')")

    # --- 半自动标注命令 ---
    labeler_parser = subparsers.add_parser("semi_auto_labeler", help="运行半自动标注工具，为裁剪出的文字图片添加标签")
    labeler_parser.add_argument("--input-dir", default=settings.paths.trocr_dataset_dir, help="包含裁剪图片的基础目录")
    labeler_parser.add_argument("--lang", default="ch_sim,en", help="EasyOCR识别的语言列表，用逗号分隔")

    # --- 模型评估命令 ---
    eval_parser = subparsers.add_parser("evaluate", help="评估模型在验证集上的性能")
    eval_parser.add_argument("--model", default=settings.yolo_inference.model_path, help="要评估的模型文件 (位于 data/models/ 下)")
    eval_parser.add_argument("--data", default=settings.paths.get_yolo_dataset_yaml_path(), help="数据集的 data.yaml 配置文件路径")
    eval_parser.add_argument("--device", default=settings.yolo_inference.device, help="评估设备")

    # --- 数据收集命令 ---
    collect_parser = subparsers.add_parser("collect", help="收集训练图片，支持代理或延时降频")
    collect_parser.add_argument("--samples", type=int, default=100, help="要收集的样本数量")
    collect_parser.add_argument("--output", default=settings.data_collection.output_dir, help="收集数据的输出目录")
    collect_parser.add_argument("--proxy-source", help="[可选] 代理来源 (URL或文件路径)。如果提供，将启用多线程代理模式")
    collect_parser.add_argument("--captcha-id", default=settings.geetest.captcha_id, help="要采集的Geetest验证码ID")
    collect_parser.add_argument("--workers", type=int, default=10, help="并发线程数 (仅在代理模式下生效)")
    collect_parser.add_argument("--delay", type=float, default=3.0, help="无代理模式下，两次请求之间的基础延迟秒数")

    # --- 新增：合成数据生成命令 ---
    generate_parser = subparsers.add_parser("generate_synthetic", help="生成用于TrOCR训练的合成验证码图像")
    generate_parser.add_argument("--num-images", type=int, default=100, help="要生成的合成图像数量 (默认: 100)")
    generate_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="指定运行设备: 'cpu', 'cuda' (NVIDIA GPU) 或 'mps' (Apple Silicon GPU)" )
    # Note: Paths for backgrounds, fonts, etc., are handled internally by SyntheticDataGenerator using settings.

    # --- 全局参数 ---
    parser.add_argument("--log-level", default=settings.logging.level, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="全局日志级别")

    args = parser.parse_args()

    # 1. 配置日志系统
    settings.logging.level = args.log_level.upper()
    setup_logging(settings.logging)
    logger = logging.getLogger("main")

    # 2. 根据命令分发任务
    try:
        if args.command == "recognize":
            logger.info(f"--- 使用 {args.engine.upper()} 引擎进行文本识别 ---")
            try:
                image = Image.open(args.image)
            except FileNotFoundError:
                logger.error(f"图像文件未找到: {args.image}")
                return

            recognizer = None
            if args.engine == "trocr":
                recognizer = TrOCRRecognizer(model_name=args.model)
            elif args.engine == "paddle":
                recognizer = PaddleRecognizer(model_name=args.model)
            
            if not recognizer:
                logger.error(f"无法初始化引擎 '{args.engine}'。")
                return

            text = recognizer.recognize(image)
            
            print("\n" + "="*50)
            print(f"识别结果 ({args.engine.upper()}):")
            print(f"  文件: {args.image}")
            print(f"  文本: {text}")
            print("="*50)

        elif args.command == "run":
            logger.info(f"--- 在 {args.mode} 模式下启动验证码处理器 ---")
            
            # Apply command-line overrides to settings
            if args.yolo_model:
                settings.yolo_inference.model_path = args.yolo_model
                logger.info(f"YOLO模型路径已通过命令行参数覆盖为: {settings.yolo_inference.model_path}")
            if args.ocr_engine:
                settings.ocr.engine = args.ocr_engine
                logger.info(f"OCR引擎已通过命令行参数覆盖为: {settings.ocr.engine}")
            if args.trocr_model and settings.ocr.engine == "trocr": # Only apply if TrOCR is the selected engine
                settings.ocr.trocr.model_name = args.trocr_model
                logger.info(f"TrOCR模型名称已通过命令行参数覆盖为: {settings.ocr.trocr.model_name}")
            if args.paddle_model_dir and settings.ocr.engine == "paddle": # Only apply if PaddleOCR is the selected engine
                settings.ocr.paddle.model_dir = args.paddle_model_dir
                logger.info(f"PaddleOCR推理模型目录已通过命令行参数覆盖为: {settings.ocr.paddle.model_dir}")

            processor = CaptchaProcessor()
            processor.current_mode = args.mode
            result = processor.process(captcha_id=args.captcha_id)
            
            print("\n" + "="*50)
            print(f"{args.mode} 模式运行结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("="*50)

        elif args.command == "prepare":
            logger.info("--- 开始准备数据集 ---")
            dataset_preparation.prepare_dataset_from_source(
                source_dir=args.source,
                dataset_dir=args.output,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                augment_multiplier=args.augment
            )
            logger.info("--- 数据集准备完成 ---")
        
        elif args.command == "collect":
            logger.info("--- 开始数据收集 ---")
            data_collector.run_collection_pipeline(
                num_samples=args.samples,
                output_dir=args.output,
                captcha_id=args.captcha_id,
                proxy_source=args.proxy_source,
                max_workers=args.workers,
                delay=args.delay
            )
            logger.info("--- 数据收集完成 ---")

        elif args.command == "train":
            logger.info("--- 开始模型训练 ---")
            if args.epochs: settings.yolo_training.epochs = args.epochs
            if args.batch: settings.yolo_training.batch = args.batch
            if args.imgsz: settings.yolo_training.imgsz = args.imgsz
            
            result = train_yolo.train(
                training_config=settings.yolo_training,
                path_config=settings.paths,
                model_name=args.model,
                data_yaml_path=args.data,
                device=args.device
            )
            if result.get("success"):
                logger.info(f"训练成功！最佳模型保存在: {result.get('best_model_path')}")
            else:
                logger.error(f"训练失败: {result.get('error')}")
            
        elif args.command == "extract_text_regions":
            logger.info("--- 开始提取文字区域 ---")
            result = text_extractor.extract_text_regions(
                yolo_model_path=args.yolo_model,
                input_image_dir=args.input_dir,
                output_base_dir=args.output_dir,
                confidence_threshold=args.conf_thres,
                iou_threshold=args.iou_thres,
                device=args.device
            )
            if result["success"]:
                print(result["message"])
            else:
                print(f"提取文字区域失败: {result['error']}")
            logger.info("--- 文字区域提取完成 ---")

        elif args.command == "semi_auto_labeler":
            logger.info("--- 开始半自动标注 ---")
            lang_list = [lang.strip() for lang in args.lang.split(',')]
            result = semi_auto_labeler.run_semi_auto_labeler(
                input_base_dir=args.input_dir,
                lang_list=lang_list
            )
            if result["success"]:
                print(result["message"])
            else:
                print(f"半自动标注失败: {result['error']}")
            logger.info("--- 半自动标注完成 ---")
    
        elif args.command == "train_trocr":
            logger.info("--- 开始TrOCR模型训练 ---")
            if args.epochs: settings.ocr.trocr.epochs = args.epochs
            if args.batch_size: settings.ocr.trocr.batch_size = args.batch_size
            if args.learning_rate: settings.ocr.trocr.learning_rate = args.learning_rate
            if args.model_name: settings.ocr.trocr.model_name = args.model_name
            if args.output_dir: settings.ocr.trocr.output_dir = args.output_dir
            if args.device: settings.ocr.trocr.device = args.device

            train_trocr.train_trocr_model(
                config=settings.ocr.trocr,
                dataset_dir=args.dataset_dir
            )
            logger.info("--- TrOCR模型训练完成 ---")

        elif args.command == "train_paddle":
            logger.info("--- 开始 PaddleOCR 模型训练配置 ---")
            result = train_paddleocr.train_paddle_model(
                train_label_file=args.train_label_file,
                val_label_file=args.val_label_file,
                config=settings.ocr.paddle,
                path_config=settings.paths
            )
            if result.get("success"):
                logger.info(result.get("message"))
            else:
                logger.error(f"PaddleOCR 训练失败: {result.get('error')}")

        elif args.command == "evaluate":
            logger.info("--- 开始模型评估 ---")
            result = train_yolo.validate(
                inference_config=settings.yolo_inference,
                path_config=settings.paths,
                data_yaml_path=args.data,
                model_name=args.model,
                device=args.device
            )
            if result.get("success"):
                print("评估结果:")
                for key, value in result["metrics"].items():
                    print(f"  {key}: {value:.4f}")
            else:
                logger.error(f"评估失败: {result.get('error')}")

        elif args.command == "export":
            logger.info("--- 开始模型导出 ---")
            result = train_yolo.export(
                path_config=settings.paths,
                model_name=args.model,
                export_format=args.format
            )
            if result.get("success"):
                logger.info(f"模型已成功导出到: {result.get('exported_model_path')}")
            else:
                logger.error(f"导出失败: {result.get('error')}")

        # --- 新增：处理 generate_synthetic 命令 ---
        elif args.command == "generate_synthetic":
            logger.info(f"--- 开始生成 {args.num_images} 张合成验证码图像 ---")
            try:
                # Instantiate the generator
                generator = TrOCRDataGenerator(num_images=args.num_images)
                
                # Run the generation process
                generator.generate()
                
                logger.info("--- 合成数据生成完成 ---")
            except ImportError:
                logger.error("无法导入 SyntheticDataGenerator。请确保 'training/synthetic_data_generator.py' 文件存在且可导入。")
            except FileNotFoundError as e:
                logger.error(f"合成数据生成过程中文件未找到: {e}")
            except Exception as e:
                logger.error(f"合成数据生成时发生错误: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"程序顶层异常: {e}", exc_info=True)

if __name__ == "__main__":
    main()