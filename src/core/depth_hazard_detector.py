#!/usr/bin/env python3
"""
深度危险检测模块 - BlindStar决策系统扩展组件
基于深度信息检测YOLO无法识别的几何危险，如深坑、台阶、路面破损等
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import scipy.ndimage as ndimage
from scipy import signal

logger = logging.getLogger(__name__)


class DepthHazardType(Enum):
    """深度危险类型枚举"""
    DEEP_HOLE = "deep_hole"        # 深坑
    STEP_DOWN = "step_down"        # 向下台阶
    STEP_UP = "step_up"            # 向上台阶
    SLOPE_DOWN = "slope_down"      # 下坡
    SLOPE_UP = "slope_up"          # 上坡
    SURFACE_BREAK = "surface_break" # 路面破损
    CLIFF = "cliff"                # 悬崖边缘
    UNKNOWN_DEPTH = "unknown_depth" # 深度异常


@dataclass
class DepthHazard:
    """深度危险信息"""
    hazard_type: DepthHazardType
    severity: float  # 危险程度 0-1
    center_point: Tuple[int, int]  # 危险中心点 (x, y)
    bounding_box: Tuple[int, int, int, int]  # 危险区域 (x1, y1, x2, y2)
    depth_change: float  # 深度变化量(米)
    confidence: float  # 检测置信度 0-1
    distance_to_camera: float  # 距离摄像头距离(米)
    description: str  # 危险描述


class DepthHazardDetector:
    """深度危险检测器 - 基于深度图检测几何危险"""
    
    def __init__(self):
        """初始化深度危险检测器"""
        # 检测参数 - 调整为更保守的设置
        self.min_depth_change = 0.5  # 最小深度变化阈值(米) - 提高阈值减少误报
        self.max_detection_distance = 6.0  # 最大检测距离(米) - 减少检测范围
        self.min_hazard_area = 200  # 最小危险区域面积(像素) - 提高面积要求
        self.gradient_threshold = 0.8  # 深度梯度阈值 - 提高梯度要求
        
        # 不同危险类型的阈值 - 调整为更保守的设置
        self.hazard_thresholds = {
            DepthHazardType.DEEP_HOLE: {
                'min_depth_change': 0.8,  # 深坑至少0.8米深 - 提高要求
                'max_distance': 4.0,      # 减少检测距离
                'severity_factor': 1.0
            },
            DepthHazardType.STEP_DOWN: {
                'min_depth_change': 0.4,  # 台阶至少0.4米 - 提高要求
                'max_distance': 2.5,      # 减少检测距离
                'severity_factor': 0.8
            },
            DepthHazardType.STEP_UP: {
                'min_depth_change': 0.4,  # 提高要求
                'max_distance': 2.5,      # 减少检测距离
                'severity_factor': 0.6
            },
            DepthHazardType.CLIFF: {
                'min_depth_change': 1.2,  # 悬崖至少1.2米落差 - 提高要求
                'max_distance': 5.0,      # 减少检测距离
                'severity_factor': 1.0
            }
        }
        
        logger.info("✅ 深度危险检测器初始化完成")
    
    def _calculate_depth_gradients(self, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算深度图的梯度"""
        # 使用Sobel算子计算x和y方向的梯度
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return gradient_magnitude, np.arctan2(grad_y, grad_x)
    
    def _detect_depth_discontinuities(self, depth_map: np.ndarray) -> np.ndarray:
        """检测深度不连续区域"""
        # 应用高斯滤波减少噪声
        smoothed_depth = cv2.GaussianBlur(depth_map, (5, 5), 1.0)
        
        # 计算深度梯度
        gradient_magnitude, _ = self._calculate_depth_gradients(smoothed_depth)
        
        # 找到梯度较大的区域
        discontinuity_mask = gradient_magnitude > self.gradient_threshold
        
        # 形态学操作清理噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        discontinuity_mask = cv2.morphologyEx(
            discontinuity_mask.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel
        )
        
        return discontinuity_mask
    
    def _analyze_depth_region(self, depth_map: np.ndarray, 
                            region_mask: np.ndarray) -> Dict[str, float]:
        """分析深度区域的统计特征"""
        region_depths = depth_map[region_mask > 0]
        
        if len(region_depths) == 0:
            return {}
        
        return {
            'mean_depth': np.mean(region_depths),
            'median_depth': np.median(region_depths),
            'min_depth': np.min(region_depths),
            'max_depth': np.max(region_depths),
            'depth_std': np.std(region_depths),
            'depth_range': np.max(region_depths) - np.min(region_depths)
        }
    
    def _classify_hazard_type(self, region_stats: Dict[str, float], 
                            surrounding_stats: Dict[str, float],
                            region_center: Tuple[int, int],
                            frame_height: int) -> DepthHazardType:
        """根据深度特征分类危险类型"""
        if not region_stats or not surrounding_stats:
            return DepthHazardType.UNKNOWN_DEPTH
        
        depth_change = region_stats['mean_depth'] - surrounding_stats['mean_depth']
        depth_variance = region_stats['depth_std']
        
        # 判断是否在画面下方（更可能是地面危险）
        is_ground_level = region_center[1] > frame_height * 0.6
        
        # 深坑检测：区域深度明显大于周围
        if depth_change > 0.5 and is_ground_level:
            if depth_change > 1.0:
                return DepthHazardType.CLIFF
            else:
                return DepthHazardType.DEEP_HOLE
        
        # 台阶检测：深度有明显但不极端的变化
        elif abs(depth_change) > 0.2 and abs(depth_change) < 1.0:
            if depth_change > 0:
                return DepthHazardType.STEP_DOWN
            else:
                return DepthHazardType.STEP_UP
        
        # 坡度检测：深度变化较缓但持续
        elif abs(depth_change) > 0.1 and depth_variance > 0.3:
            if depth_change > 0:
                return DepthHazardType.SLOPE_DOWN
            else:
                return DepthHazardType.SLOPE_UP
        
        # 路面破损：深度变化不规律
        elif depth_variance > 0.4:
            return DepthHazardType.SURFACE_BREAK
        
        return DepthHazardType.UNKNOWN_DEPTH
    
    def _calculate_hazard_severity(self, hazard_type: DepthHazardType,
                                 depth_change: float,
                                 distance: float,
                                 area: float) -> float:
        """计算危险严重程度"""
        if hazard_type not in self.hazard_thresholds:
            return 0.5
        
        thresholds = self.hazard_thresholds[hazard_type]
        
        # 基础严重程度
        base_severity = thresholds['severity_factor']
        
        # 深度变化影响
        depth_factor = min(1.0, abs(depth_change) / thresholds['min_depth_change'])
        
        # 距离影响（越近越危险）
        distance_factor = max(0.1, 1.0 - (distance / thresholds['max_distance']))
        
        # 面积影响（面积越大越危险）
        area_factor = min(1.0, area / 1000.0)  # 标准化到1000像素
        
        severity = base_severity * depth_factor * distance_factor * (0.7 + 0.3 * area_factor)
        
        return min(1.0, severity)
    
    def _generate_hazard_description(self, hazard: DepthHazard) -> str:
        """生成危险描述"""
        distance_text = f"{hazard.distance_to_camera:.1f}米"
        
        descriptions = {
            DepthHazardType.DEEP_HOLE: f"前方{distance_text}处有深坑，深度{abs(hazard.depth_change):.1f}米",
            DepthHazardType.STEP_DOWN: f"前方{distance_text}处有向下台阶，高度{abs(hazard.depth_change):.1f}米",
            DepthHazardType.STEP_UP: f"前方{distance_text}处有向上台阶，高度{abs(hazard.depth_change):.1f}米",
            DepthHazardType.SLOPE_DOWN: f"前方{distance_text}处有下坡，坡度较大",
            DepthHazardType.SLOPE_UP: f"前方{distance_text}处有上坡，坡度较大",
            DepthHazardType.SURFACE_BREAK: f"前方{distance_text}处路面不平，请小心",
            DepthHazardType.CLIFF: f"危险！前方{distance_text}处有悬崖，落差{abs(hazard.depth_change):.1f}米",
            DepthHazardType.UNKNOWN_DEPTH: f"前方{distance_text}处深度异常，请注意"
        }
        
        return descriptions.get(hazard.hazard_type, f"前方{distance_text}处有未知危险")
    
    def detect_depth_hazards(self, depth_map: np.ndarray, 
                           frame_info: Optional[Dict[str, Any]] = None) -> List[DepthHazard]:
        """检测深度图中的几何危险"""
        if depth_map is None or depth_map.size == 0:
            return []
        
        hazards = []
        frame_height, frame_width = depth_map.shape
        
        try:
            # 检测深度不连续区域
            discontinuity_mask = self._detect_depth_discontinuities(depth_map)
            
            # 找到连通区域
            num_labels, labels = cv2.connectedComponents(discontinuity_mask)
            
            for label in range(1, num_labels):  # 跳过背景(0)
                region_mask = (labels == label)
                region_area = np.sum(region_mask)
                
                # 过滤太小的区域
                if region_area < self.min_hazard_area:
                    continue
                
                # 获取区域边界框
                coords = np.where(region_mask)
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                
                # 计算区域中心
                center_y = int((y_min + y_max) / 2)
                center_x = int((x_min + x_max) / 2)
                
                # 分析区域深度特征
                region_stats = self._analyze_depth_region(depth_map, region_mask)
                
                # 分析周围区域作为对比
                # 扩展区域边界来获取周围区域
                expand_size = 20
                y_min_exp = max(0, y_min - expand_size)
                y_max_exp = min(frame_height, y_max + expand_size)
                x_min_exp = max(0, x_min - expand_size)
                x_max_exp = min(frame_width, x_max + expand_size)
                
                surrounding_mask = np.zeros_like(region_mask)
                surrounding_mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = True
                surrounding_mask = surrounding_mask & (~region_mask)  # 排除区域本身
                
                surrounding_stats = self._analyze_depth_region(depth_map, surrounding_mask)
                
                if not region_stats or not surrounding_stats:
                    continue
                
                # 计算距离摄像头的距离
                distance_to_camera = region_stats['mean_depth']
                
                # 过滤距离太远的危险
                if distance_to_camera > self.max_detection_distance:
                    continue
                
                # 分类危险类型
                hazard_type = self._classify_hazard_type(
                    region_stats, surrounding_stats, (center_x, center_y), frame_height
                )
                
                # 计算深度变化
                depth_change = region_stats['mean_depth'] - surrounding_stats['mean_depth']
                
                # 过滤深度变化太小的区域
                if abs(depth_change) < self.min_depth_change:
                    continue
                
                # 计算危险严重程度
                severity = self._calculate_hazard_severity(
                    hazard_type, depth_change, distance_to_camera, region_area
                )
                
                # 创建危险对象
                hazard = DepthHazard(
                    hazard_type=hazard_type,
                    severity=severity,
                    center_point=(center_x, center_y),
                    bounding_box=(x_min, y_min, x_max, y_max),
                    depth_change=depth_change,
                    confidence=min(1.0, severity * 1.2),  # 置信度基于严重程度
                    distance_to_camera=distance_to_camera,
                    description=""  # 稍后生成
                )
                
                # 生成描述
                hazard.description = self._generate_hazard_description(hazard)
                
                hazards.append(hazard)
                
                logger.debug(f"检测到深度危险: {hazard.hazard_type.value} "
                           f"at ({center_x}, {center_y}), "
                           f"severity: {severity:.2f}, "
                           f"distance: {distance_to_camera:.1f}m")
            
            # 按严重程度和距离排序
            hazards.sort(key=lambda h: (h.severity, -h.distance_to_camera), reverse=True)
            
            logger.info(f"检测到 {len(hazards)} 个深度危险")
            
        except Exception as e:
            logger.error(f"深度危险检测过程中出错: {e}")
            return []
        
        return hazards
    
    def update_parameters(self, **kwargs):
        """更新检测参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"深度危险检测参数已更新: {key} = {value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        return {
            "min_depth_change": self.min_depth_change,
            "max_detection_distance": self.max_detection_distance,
            "min_hazard_area": self.min_hazard_area,
            "gradient_threshold": self.gradient_threshold
        }
