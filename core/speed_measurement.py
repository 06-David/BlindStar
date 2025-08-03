"""
速度测量模块
使用光流法和物体跟踪来计算物体的移动速度
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SpeedInfo:
    """速度信息数据类"""
    object_id: int
    speed_mps: float  # 速度 (米/秒)
    speed_kmh: float  # 速度 (公里/小时)
    velocity_x: float  # X方向速度分量
    velocity_y: float  # Y方向速度分量
    acceleration: float = 0.0  # 加速度
    confidence: float = 1.0  # 速度测量置信度

class OpticalFlowSpeedMeasurement:
    """基于光流的速度测量器"""
    
    def __init__(self, 
                 pixels_per_meter: float = 100.0,
                 max_tracking_distance: float = 50.0,
                 min_tracking_frames: int = 3):
        """
        初始化速度测量器
        
        Args:
            pixels_per_meter: 每米对应的像素数（用于像素到实际距离的转换）
            max_tracking_distance: 最大跟踪距离（像素）
            min_tracking_frames: 最小跟踪帧数
        """
        self.pixels_per_meter = pixels_per_meter
        self.max_tracking_distance = max_tracking_distance
        self.min_tracking_frames = min_tracking_frames
        
        # 跟踪历史
        self.tracking_history: Dict[int, List[Dict]] = {}
        self.next_object_id = 1
        self.previous_frame = None
        self.previous_timestamp = None
        
        logger.info(f"Speed measurement initialized with {pixels_per_meter} pixels/meter")
    
    def update_tracking(self, frame: np.ndarray, detections: List) -> List:
        """
        更新物体跟踪并计算速度
        
        Args:
            frame: 当前帧
            detections: 检测结果列表
            
        Returns:
            更新了速度信息的检测结果列表
        """
        current_timestamp = time.time()
        
        try:
            # 为检测结果分配或更新物体ID
            detections = self._assign_object_ids(detections)
            
            # 计算速度
            if self.previous_frame is not None and self.previous_timestamp is not None:
                time_delta = current_timestamp - self.previous_timestamp
                if time_delta > 0:
                    detections = self._calculate_speeds(detections, time_delta)
            
            # 更新跟踪历史
            self._update_tracking_history(detections, current_timestamp)
            
            # 保存当前帧信息
            self.previous_frame = frame.copy()
            self.previous_timestamp = current_timestamp
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in speed tracking: {e}")
            return detections
    
    def _assign_object_ids(self, detections: List) -> List:
        """为检测结果分配物体ID"""
        if not detections:
            return detections
        
        # 简单的基于位置的ID分配
        for detection in detections:
            bbox = detection.bbox
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # 查找最近的已跟踪物体
            best_match_id = None
            min_distance = float('inf')
            
            for obj_id, history in self.tracking_history.items():
                if history:
                    last_pos = history[-1]
                    distance = np.sqrt((center_x - last_pos['center_x'])**2 + 
                                     (center_y - last_pos['center_y'])**2)
                    
                    if distance < min_distance and distance < self.max_tracking_distance:
                        min_distance = distance
                        best_match_id = obj_id
            
            # 分配ID
            if best_match_id is not None:
                object_id = best_match_id
            else:
                object_id = self.next_object_id
                self.next_object_id += 1
                self.tracking_history[object_id] = []
            
            # 添加ID到检测结果
            detection.object_id = object_id
        
        return detections
    
    def _calculate_speeds(self, detections: List, time_delta: float) -> List:
        """计算物体速度"""
        for detection in detections:
            if not hasattr(detection, 'object_id'):
                continue
                
            obj_id = detection.object_id
            
            if obj_id in self.tracking_history and len(self.tracking_history[obj_id]) > 0:
                # 获取当前位置
                bbox = detection.bbox
                current_x = (bbox[0] + bbox[2]) / 2
                current_y = (bbox[1] + bbox[3]) / 2
                
                # 获取上一帧位置
                last_pos = self.tracking_history[obj_id][-1]
                prev_x = last_pos['center_x']
                prev_y = last_pos['center_y']
                
                # 计算像素位移
                dx_pixels = current_x - prev_x
                dy_pixels = current_y - prev_y
                
                # 转换为实际距离（米）
                dx_meters = dx_pixels / self.pixels_per_meter
                dy_meters = dy_pixels / self.pixels_per_meter
                
                # 计算速度
                velocity_x = dx_meters / time_delta  # m/s
                velocity_y = dy_meters / time_delta  # m/s
                speed_mps = np.sqrt(velocity_x**2 + velocity_y**2)
                speed_kmh = speed_mps * 3.6  # 转换为 km/h
                
                # 计算加速度（如果有足够的历史数据）
                acceleration = 0.0
                if len(self.tracking_history[obj_id]) >= 2:
                    prev_speed = self.tracking_history[obj_id][-1].get('speed_mps', 0)
                    acceleration = (speed_mps - prev_speed) / time_delta
                
                # 创建速度信息
                speed_info = SpeedInfo(
                    object_id=obj_id,
                    speed_mps=speed_mps,
                    speed_kmh=speed_kmh,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    acceleration=acceleration,
                    confidence=self._calculate_confidence(obj_id)
                )
                
                # 添加到检测结果
                detection.speed_info = speed_info
        
        return detections
    
    def _update_tracking_history(self, detections: List, timestamp: float):
        """更新跟踪历史"""
        current_ids = set()
        
        for detection in detections:
            if hasattr(detection, 'object_id'):
                obj_id = detection.object_id
                current_ids.add(obj_id)
                
                bbox = detection.bbox
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # 获取速度信息
                speed_mps = 0.0
                if hasattr(detection, 'speed_info') and detection.speed_info:
                    speed_mps = detection.speed_info.speed_mps
                
                # 添加到历史记录
                history_entry = {
                    'timestamp': timestamp,
                    'center_x': center_x,
                    'center_y': center_y,
                    'speed_mps': speed_mps,
                    'class_name': detection.class_name
                }
                
                if obj_id not in self.tracking_history:
                    self.tracking_history[obj_id] = []
                
                self.tracking_history[obj_id].append(history_entry)
                
                # 限制历史记录长度
                max_history = 10
                if len(self.tracking_history[obj_id]) > max_history:
                    self.tracking_history[obj_id] = self.tracking_history[obj_id][-max_history:]
        
        # 清理不再跟踪的物体（超过一定时间没有出现）
        max_missing_time = 5.0  # 5秒
        ids_to_remove = []
        
        for obj_id in self.tracking_history:
            if obj_id not in current_ids:
                if self.tracking_history[obj_id]:
                    last_seen = self.tracking_history[obj_id][-1]['timestamp']
                    if timestamp - last_seen > max_missing_time:
                        ids_to_remove.append(obj_id)
        
        for obj_id in ids_to_remove:
            del self.tracking_history[obj_id]
    
    def _calculate_confidence(self, obj_id: int) -> float:
        """计算速度测量的置信度"""
        if obj_id not in self.tracking_history:
            return 0.5
        
        history_length = len(self.tracking_history[obj_id])
        
        # 基于跟踪历史长度的置信度
        if history_length >= self.min_tracking_frames:
            return min(1.0, history_length / 10.0)
        else:
            return 0.3 + (history_length / self.min_tracking_frames) * 0.4
    
    def get_tracking_stats(self) -> Dict:
        """获取跟踪统计信息"""
        return {
            'total_tracked_objects': len(self.tracking_history),
            'active_objects': len([h for h in self.tracking_history.values() if h]),
            'next_object_id': self.next_object_id
        }
    
    def reset_tracking(self):
        """重置跟踪状态"""
        self.tracking_history.clear()
        self.next_object_id = 1
        self.previous_frame = None
        self.previous_timestamp = None
        logger.info("Speed tracking reset")
