"""
坐标处理工具函数
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def convert_to_geetest_format(
    pixel_coords: List[Tuple[int, int]],
    container_size: Tuple[int, int] = (300, 200)
) -> List[List[int]]:
    """
    将像素坐标转换为极验格式坐标
    """
    container_width, container_height = container_size
    geetest_coords = []
    
    for x, y in pixel_coords:
        relative_x = (x / container_width) * 100
        relative_y = (y / container_height) * 100
        final_x = round(relative_x * 100)
        final_y = round(relative_y * 100)
        geetest_coords.append([final_x, final_y])
    
    return geetest_coords

def convert_from_geetest_format(
    geetest_coords: List[List[int]],
    container_size: Tuple[int, int] = (300, 200)
) -> List[Tuple[int, int]]:
    """
    将极验格式坐标转换为像素坐标
    """
    container_width, container_height = container_size
    pixel_coords = []
    
    for x_percent, y_percent in geetest_coords:
        relative_x = x_percent / 100.0
        relative_y = y_percent / 100.0
        x = int(relative_x * container_width)
        y = int(relative_y * container_height)
        pixel_coords.append((x, y))
    
    return pixel_coords

def sort_coordinates_by_reading_order(
    coords: List[Tuple[int, int]],
    tolerance: int = 20
) -> List[Tuple[int, int]]:
    """
    按阅读顺序（从左到右，从上到下）排序坐标
    """
    if not coords:
        return []
    
    sorted_by_y = sorted(coords, key=lambda c: c[1])
    
    rows = []
    current_row = [sorted_by_y[0]]
    current_y = sorted_by_y[0][1]
    
    for coord in sorted_by_y[1:]:
        if abs(coord[1] - current_y) <= tolerance:
            current_row.append(coord)
        else:
            current_row.sort(key=lambda c: c[0])
            rows.append(current_row)
            current_row = [coord]
            current_y = coord[1]
    
    if current_row:
        current_row.sort(key=lambda c: c[0])
        rows.append(current_row)
    
    result = [item for row in rows for item in row]
    return result

def calculate_distance(
    coord1: Tuple[int, int],
    coord2: Tuple[int, int]
) -> float:
    """
    计算两点之间的欧氏距离
    """
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def find_closest_coord(
    target: Tuple[int, int],
    coords: List[Tuple[int, int]],
    max_distance: Optional[float] = None
) -> Tuple[Optional[int], Optional[float]]:
    """
    查找最近的坐标
    """
    if not coords:
        return None, None
    
    distances = [calculate_distance(target, coord) for coord in coords]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    
    if max_distance is not None and min_distance > max_distance:
        return None, None
    
    return min_index, min_distance

def bbox_to_center(
    bbox: Tuple[int, int, int, int]
) -> Tuple[int, int]:
    """
    将边界框转换为中心点坐标 (xmin, ymin, xmax, ymax) -> (x_center, y_center)
    """
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return center_x, center_y

def bboxes_to_centers(bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """将边界框列表转换为中心点列表"""
    return [bbox_to_center(bbox) for bbox in bboxes]
