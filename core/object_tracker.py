#!/usr/bin/env python3
"""
物体跟踪模块
用于跟踪物体运动，计算速度和加速度信息
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TrackedObject:
    """跟踪的物体信息"""
    object_id: int
    class_id: int
    class_name: str
    positions: List[Tuple[float, float, float]]  # (x, y, timestamp)
    confidences: List[float]
    distances: List[Optional[float]]  # 距离历史
    
    # 当前状态
    current_velocity_x: float = 0.0
    current_velocity_y: float = 0.0
    current_speed: float = 0.0
    current_acceleration: float = 0.0
    movement_direction: float = 0.0
    movement_status: str = "静止"
    
    # 跟踪状态
    last_seen_frame: int = 0
    tracking_confidence: float = 1.0
    is_active: bool = True


class SimpleObjectTracker:
    """简单的物体跟踪器"""
    
    def __init__(self, 
                 max_distance_threshold: float = 100.0,
                 max_missing_frames: int = 30,
                 min_tracking_frames: int = 3,
                 fps: float = 30.0):
        """
        初始化物体跟踪器
        
        Args:
            max_distance_threshold: 最大匹配距离阈值（像素）
            max_missing_frames: 最大丢失帧数
            min_tracking_frames: 最小跟踪帧数（用于计算速度）
            fps: 视频帧率
        """
        self.max_distance_threshold = max_distance_threshold
        self.max_missing_frames = max_missing_frames
        self.min_tracking_frames = min_tracking_frames
        self.fps = fps
        
        # 跟踪状态
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 1
        self.current_frame = 0
        
    def update(self, 
               detections: List[Dict], 
               frame_number: int, 
               timestamp: float) -> List[Dict]:
        """
        更新跟踪器并返回带有跟踪信息的检测结果
        
        Args:
            detections: 当前帧的检测结果
            frame_number: 帧编号
            timestamp: 时间戳
            
        Returns:
            带有跟踪信息的检测结果
        """
        self.current_frame = frame_number
        
        # 匹配检测结果到已跟踪的物体
        matched_objects, unmatched_detections = self._match_detections(detections, timestamp)
        
        # 更新已匹配的物体
        for obj_id, detection in matched_objects:
            self._update_tracked_object(obj_id, detection, frame_number, timestamp)
        
        # 创建新的跟踪物体
        for detection in unmatched_detections:
            self._create_new_tracked_object(detection, frame_number, timestamp)
        
        # 清理丢失的物体
        self._cleanup_lost_objects()
        
        # 为检测结果添加跟踪信息
        enhanced_detections = self._enhance_detections_with_tracking(detections)
        
        return enhanced_detections
    
    def _match_detections(self, 
                         detections: List[Dict], 
                         timestamp: float) -> Tuple[List[Tuple[int, Dict]], List[Dict]]:
        """匹配检测结果到已跟踪的物体"""
        matched_objects = []
        unmatched_detections = []
        used_object_ids = set()
        
        for detection in detections:
            center_x = detection.get('center', [0, 0])[0]
            center_y = detection.get('center', [0, 0])[1]
            class_id = detection.get('class_id', -1)
            
            best_match_id = None
            best_distance = float('inf')
            
            # 寻找最佳匹配
            for obj_id, tracked_obj in self.tracked_objects.items():
                if (obj_id in used_object_ids or 
                    not tracked_obj.is_active or 
                    tracked_obj.class_id != class_id):
                    continue
                
                # 计算距离
                if tracked_obj.positions:
                    last_x, last_y, _ = tracked_obj.positions[-1]
                    distance = math.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                    
                    if distance < self.max_distance_threshold and distance < best_distance:
                        best_distance = distance
                        best_match_id = obj_id
            
            if best_match_id is not None:
                matched_objects.append((best_match_id, detection))
                used_object_ids.add(best_match_id)
            else:
                unmatched_detections.append(detection)
        
        return matched_objects, unmatched_detections
    
    def _update_tracked_object(self, 
                              obj_id: int, 
                              detection: Dict, 
                              frame_number: int, 
                              timestamp: float):
        """更新已跟踪的物体"""
        tracked_obj = self.tracked_objects[obj_id]
        
        # 更新位置历史
        center = detection.get('center', [0, 0])
        tracked_obj.positions.append((center[0], center[1], timestamp))
        tracked_obj.confidences.append(detection.get('confidence', 0.0))
        tracked_obj.distances.append(detection.get('distance_meters'))
        tracked_obj.last_seen_frame = frame_number
        
        # 限制历史长度
        max_history = max(self.min_tracking_frames * 2, 10)
        if len(tracked_obj.positions) > max_history:
            tracked_obj.positions = tracked_obj.positions[-max_history:]
            tracked_obj.confidences = tracked_obj.confidences[-max_history:]
            tracked_obj.distances = tracked_obj.distances[-max_history:]
        
        # 计算速度和加速度
        self._calculate_motion_parameters(tracked_obj)
    
    def _create_new_tracked_object(self, 
                                  detection: Dict, 
                                  frame_number: int, 
                                  timestamp: float):
        """创建新的跟踪物体"""
        center = detection.get('center', [0, 0])
        
        tracked_obj = TrackedObject(
            object_id=self.next_object_id,
            class_id=detection.get('class_id', -1),
            class_name=detection.get('class_name', 'unknown'),
            positions=[(center[0], center[1], timestamp)],
            confidences=[detection.get('confidence', 0.0)],
            distances=[detection.get('distance_meters')],
            last_seen_frame=frame_number
        )
        
        self.tracked_objects[self.next_object_id] = tracked_obj
        self.next_object_id += 1
    
    def _cleanup_lost_objects(self):
        """清理丢失的物体"""
        lost_objects = []
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            frames_missing = self.current_frame - tracked_obj.last_seen_frame
            if frames_missing > self.max_missing_frames:
                lost_objects.append(obj_id)
            elif frames_missing > 0:
                tracked_obj.is_active = False
            else:
                tracked_obj.is_active = True
        
        for obj_id in lost_objects:
            del self.tracked_objects[obj_id]
    
    def _calculate_motion_parameters(self, tracked_obj: TrackedObject):
        """计算运动参数"""
        if len(tracked_obj.positions) < 2:
            return
        
        positions = tracked_obj.positions
        
        # 计算当前速度（使用最近的两个位置）
        if len(positions) >= 2:
            x1, y1, t1 = positions[-2]
            x2, y2, t2 = positions[-1]
            
            dt = t2 - t1
            if dt > 0:
                tracked_obj.current_velocity_x = (x2 - x1) / dt
                tracked_obj.current_velocity_y = (y2 - y1) / dt
                tracked_obj.current_speed = math.sqrt(
                    tracked_obj.current_velocity_x**2 + tracked_obj.current_velocity_y**2
                )
                
                # 计算运动方向（角度）
                if tracked_obj.current_speed > 1.0:  # 只有在明显移动时才计算方向
                    tracked_obj.movement_direction = math.degrees(
                        math.atan2(tracked_obj.current_velocity_y, tracked_obj.current_velocity_x)
                    )
        
        # 计算加速度（使用最近的三个位置）
        if len(positions) >= 3:
            x1, y1, t1 = positions[-3]
            x2, y2, t2 = positions[-2]
            x3, y3, t3 = positions[-1]
            
            dt1 = t2 - t1
            dt2 = t3 - t2
            
            if dt1 > 0 and dt2 > 0:
                v1 = math.sqrt(((x2-x1)/dt1)**2 + ((y2-y1)/dt1)**2)
                v2 = math.sqrt(((x3-x2)/dt2)**2 + ((y3-y2)/dt2)**2)
                
                tracked_obj.current_acceleration = (v2 - v1) / ((dt1 + dt2) / 2)
        
        # 确定运动状态
        speed_threshold = 2.0  # 像素/秒
        accel_threshold = 5.0  # 像素/秒²
        
        if tracked_obj.current_speed < speed_threshold:
            tracked_obj.movement_status = "静止"
        elif abs(tracked_obj.current_acceleration) < accel_threshold:
            tracked_obj.movement_status = "匀速"
        elif tracked_obj.current_acceleration > accel_threshold:
            tracked_obj.movement_status = "加速"
        else:
            tracked_obj.movement_status = "减速"
    
    def _enhance_detections_with_tracking(self, detections: List[Dict]) -> List[Dict]:
        """为检测结果添加跟踪信息"""
        enhanced_detections = []
        
        for detection in detections:
            enhanced_detection = detection.copy()
            
            # 查找对应的跟踪物体
            center = detection.get('center', [0, 0])
            class_id = detection.get('class_id', -1)
            
            matched_obj = None
            for tracked_obj in self.tracked_objects.values():
                if (tracked_obj.class_id == class_id and 
                    tracked_obj.is_active and 
                    tracked_obj.positions):
                    
                    last_x, last_y, _ = tracked_obj.positions[-1]
                    distance = math.sqrt((center[0] - last_x)**2 + (center[1] - last_y)**2)
                    
                    if distance < self.max_distance_threshold:
                        matched_obj = tracked_obj
                        break
            
            if matched_obj:
                # 添加跟踪信息
                enhanced_detection['object_id'] = matched_obj.object_id
                enhanced_detection['velocity_x'] = matched_obj.current_velocity_x
                enhanced_detection['velocity_y'] = matched_obj.current_velocity_y
                enhanced_detection['speed_pixels_per_second'] = matched_obj.current_speed
                enhanced_detection['acceleration'] = matched_obj.current_acceleration
                enhanced_detection['movement_direction'] = matched_obj.movement_direction
                enhanced_detection['movement_status'] = matched_obj.movement_status
                
                # 如果有距离信息，计算实际速度
                if (matched_obj.distances and 
                    matched_obj.distances[-1] is not None and 
                    matched_obj.current_speed > 0):
                    
                    # 简单的像素到米的转换（基于距离）
                    distance_m = matched_obj.distances[-1]
                    if distance_m > 0:
                        # 假设物体在图像中的大小与距离成反比
                        # 这是一个简化的转换，实际应用中需要更精确的标定
                        bbox = detection.get('bbox', [0, 0, 0, 0])
                        object_size_pixels = math.sqrt((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2)
                        
                        if object_size_pixels > 0:
                            # 估算像素到米的比例
                            pixels_per_meter = object_size_pixels / (distance_m * 0.1)  # 简化估算
                            speed_mps = matched_obj.current_speed / pixels_per_meter
                            
                            enhanced_detection['speed_meters_per_second'] = speed_mps
                            enhanced_detection['speed_kmh'] = speed_mps * 3.6
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def get_tracking_stats(self) -> Dict:
        """获取跟踪统计信息"""
        active_objects = sum(1 for obj in self.tracked_objects.values() if obj.is_active)
        total_objects = len(self.tracked_objects)
        
        return {
            "active_objects": active_objects,
            "total_tracked_objects": total_objects,
            "next_object_id": self.next_object_id
        }
