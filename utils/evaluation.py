"""
评估工具函数
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
from pathlib import Path

# 使用dataclass来定义清晰的数据结构
@dataclass
class DetectionResult:
    """检测结果的数据模型"""
    bbox: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    confidence: float
    class_id: int

@dataclass
class GroundTruth:
    """真实标注的数据模型"""
    bbox: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    class_id: int

def calculate_iou(bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> float:
    """
    计算IoU（交并比）
    """
    inter_x1 = max(bbox1[0], bbox2[0])
    inter_y1 = max(bbox1[1], bbox2[1])
    inter_x2 = min(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_ap(detections: List[DetectionResult], ground_truths: List[GroundTruth], iou_threshold: float = 0.5) -> float:
    """
    计算单个类别的平均精度（AP）
    """
    if not ground_truths:
        return 0.0
    
    detections.sort(key=lambda x: x.confidence, reverse=True)
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    gt_matched = [False] * len(ground_truths)
    
    for i, detection in enumerate(detections):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truths):
            if gt_matched[j]:
                continue
            iou = calculate_iou(detection.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
            
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # 使用面积下的曲线计算AP
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(len(recalls) - 1):
        if recalls[i+1] != recalls[i]:
            ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
            
    return ap

def calculate_map(
    all_detections: List[DetectionResult],
    all_ground_truths: List[GroundTruth],
    iou_threshold: float = 0.5
) -> float:
    """
    计算平均精度均值（mAP）
    """
    class_ids = set(gt.class_id for gt in all_ground_truths)
    if not class_ids:
        return 0.0
        
    aps = []
    for class_id in sorted(list(class_ids)):
        detections_per_class = [d for d in all_detections if d.class_id == class_id]
        gt_per_class = [gt for gt in all_ground_truths if gt.class_id == class_id]
        
        ap = calculate_ap(detections_per_class, gt_per_class, iou_threshold)
        aps.append(ap)
        
    return np.mean(aps) if aps else 0.0

def evaluate_captcha_performance(
    predictions: List[List[Tuple[int, int]]],
    ground_truths: List[List[Tuple[int, int]]],
    tolerance: int = 10
) -> Dict[str, Any]:
    """
    评估验证码点击任务的性能
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("预测和真实值的数量必须相同")
    
    total_cases = len(predictions)
    correct_cases = 0
    total_clicks = 0
    correct_clicks = 0
    
    for pred_clicks, gt_clicks in zip(predictions, ground_truths):
        if len(pred_clicks) == len(gt_clicks):
            is_case_correct = True
            for pred_click, gt_click in zip(pred_clicks, gt_clicks):
                total_clicks += 1
                distance = np.sqrt((pred_click[0] - gt_click[0])**2 + (pred_click[1] - gt_click[1])**2)
                if distance <= tolerance:
                    correct_clicks += 1
                else:
                    is_case_correct = False
            
            if is_case_correct:
                correct_cases += 1
    
    return {
        'total_cases': total_cases,
        'correct_cases': correct_cases,
        'case_accuracy': correct_cases / total_cases if total_cases > 0 else 0,
        'total_clicks': total_clicks,
        'correct_clicks': correct_clicks,
        'click_accuracy': correct_clicks / total_clicks if total_clicks > 0 else 0,
        'tolerance': tolerance
    }

def save_evaluation_results(results: Dict[str, Any], output_path: Union[str, Path]):
    """
    保存评估结果到JSON文件
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存到: {output_path}")

def load_evaluation_results(input_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从JSON文件加载评估结果
    """
    input_path = Path(input_path)
    if not input_path.exists():
        return {}
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)
