"""
YOLOv8 模型训练、验证和导出的函数
"""
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from ultralytics import YOLO
from config.settings import YOLOTrainingConfig, YOLOInferenceConfig, PathConfig

def train(
    training_config: YOLOTrainingConfig,
    path_config: PathConfig,
    model_name: str,
    data_yaml_path: str,
    device: str
) -> Dict[str, Any]:
    """
    训练一个新的YOLOv8模型。

    Args:
        training_config: 包含所有训练超参数的配置对象。
        path_config: 包含项目路径的配置对象。
        model_name: 基础模型文件 (e.g., 'yolov8n.pt')。
        data_yaml_path: 数据集配置文件的路径。
        device: 训练设备 ('cpu', '0', etc.)。

    Returns:
        一个包含训练结果的字典。
    """
    try:
        model = YOLO(model_name)
        
        # 将 dataclass 转换为 ultralytics 期望的字典格式
        yolo_args = {
            'data': data_yaml_path,
            'epochs': training_config.epochs,
            'batch': training_config.batch,
            'imgsz': training_config.imgsz,
            'workers': training_config.workers,
            'device': device,
            'optimizer': training_config.optimizer,
            'patience': training_config.patience,
            'save_period': training_config.save_period,
            'project': training_config.project,
            'name': training_config.name,
            'exist_ok': training_config.exist_ok,
            'val': training_config.val,
            'deterministic': training_config.deterministic,
            'single_cls': training_config.single_cls,
            'rect': training_config.rect,
            'resume': training_config.resume,
            'amp': training_config.amp,
            'half': False, # Explicitly disable half-precision for MPS compatibility
            'fraction': training_config.fraction,
            'profile': training_config.profile,
            'pretrained': training_config.pretrained,
            'lr0': training_config.lr0,
            'lrf': training_config.lrf,
            'momentum': training_config.momentum,
            'weight_decay': training_config.weight_decay,
        }
        
        results = model.train(**yolo_args)
        
        # 训练完成后，将最佳模型复制到 models 目录
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        final_model_path = Path(path_config.get_model_path(path_config.best_model_name))
        shutil.copy2(best_model_path, final_model_path)
        
        return {
            "success": True,
            "message": "Training completed successfully.",
            "best_model_path": str(final_model_path),
            "results": results.results_dict
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

def validate(
    inference_config: YOLOInferenceConfig,
    path_config: PathConfig,
    data_yaml_path: str,
    model_name: str,
    device: str
) -> Dict[str, Any]:
    """
    在验证集上评估YOLOv8模型。

    Args:
        inference_config: 推理配置。
        path_config: 路径配置。
        data_yaml_path: 数据集配置文件路径。
        model_name: 要评估的模型文件名 (e.g., 'best.pt')。
        device: 评估设备。

    Returns:
        包含评估指标的字典。
    """
    try:
        model_path = path_config.get_model_path(model_name)
        model = YOLO(model_path)
        
        metrics = model.val(
            data=data_yaml_path,
            imgsz=inference_config.imgsz,
            conf=inference_config.conf,
            iou=inference_config.iou,
            device=device
        )
        
        return {
            "success": True,
            "metrics": {
                "map50": metrics.box.map50,
                "map50_95": metrics.box.map,
                "precision": metrics.box.p[0], # p is a list
                "recall": metrics.box.r[0],   # r is a list
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def export(
    path_config: PathConfig,
    model_name: str,
    export_format: str = "onnx"
) -> Dict[str, Any]:
    """
    将YOLOv8模型导出为指定格式 (e.g., ONNX, TensorRT)。

    Args:
        path_config: 路径配置。
        model_name: 要导出的模型文件名 (e.g., 'best.pt')。
        export_format: 目标格式。

    Returns:
        包含导出结果的字典。
    """
    try:
        model_path = path_config.get_model_path(model_name)
        model = YOLO(model_path)
        
        exported_path = model.export(format=export_format)
        
        # 将导出的模型移动到 models 目录
        final_path = Path(path_config.model_dir) / Path(exported_path).name
        shutil.move(exported_path, final_path)
        
        return {
            "success": True,
            "exported_model_path": str(final_path)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
