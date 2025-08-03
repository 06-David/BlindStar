"""
视频帧分析模块
记录和分析视频处理过程中每一帧的详细信息，包括物体检测、距离测量和速度信息
"""

import json
import csv
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FrameObjectInfo:
    """单个物体的帧信息"""
    frame_number: int
    timestamp: float
    object_id: Optional[int]
    class_name: str
    class_id: int
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    center_x: float
    center_y: float
    width: int
    height: int
    area: int
    
    # 距离信息
    distance_meters: Optional[float] = None
    distance_confidence: Optional[float] = None
    depth_quality: Optional[float] = None
    
    # 速度信息
    speed_mps: Optional[float] = None
    speed_kmh: Optional[float] = None
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    acceleration: Optional[float] = None
    
    # 帧元数据
    frame_width: int = 0
    frame_height: int = 0
    processing_time_ms: float = 0.0

@dataclass
class FrameAnalysis:
    """完整帧分析信息"""
    frame_number: int
    timestamp: float
    frame_width: int
    frame_height: int
    total_objects: int
    processing_time_ms: float
    objects: List[FrameObjectInfo]

class FrameAnalyzer:
    """视频帧分析器"""
    
    def __init__(self, 
                 log_dir: str = "logs/frame_analysis",
                 enable_json_log: bool = True,
                 enable_csv_log: bool = True,
                 log_level: str = "INFO"):
        """
        初始化帧分析器
        
        Args:
            log_dir: 日志目录
            enable_json_log: 启用JSON格式日志
            enable_csv_log: 启用CSV格式日志
            log_level: 日志级别
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_json_log = enable_json_log
        self.enable_csv_log = enable_csv_log

        # 生成会话ID
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()
        
        # 存储分析数据
        self.frames_data = []
        # 会话统计：使用 float 初始值以免类型检查警告
        self.session_stats: Dict[str, float] = {
            "frames_processed": 0,
            "total_objects_detected": 0,
            "session_duration_seconds": 0.0,
            "average_objects_per_frame": 0.0,
            "processing_fps": 0.0
        }
        
        # 设置日志
        self.logger = logging.getLogger(f"FrameAnalyzer_{self.session_id}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 不再创建逐帧详细日志文件处理器
        
        # 初始化CSV文件
        if self.enable_csv_log:
            self.csv_file_path = self.log_dir / f"frame_analysis_{self.session_id}.csv"
            self._init_csv_file()
        
        # 不再写逐帧详细日志
        
    def _init_csv_file(self):
        """初始化CSV文件头"""
        csv_headers = [
            "frame_number", "timestamp", "object_id", "class_name", "class_id", "confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "center_x", "center_y", 
            "width", "height", "area",
            "distance_meters", "distance_confidence", "depth_quality",
            "speed_mps", "speed_kmh", "velocity_x", "velocity_y", "acceleration",
            "frame_width", "frame_height", "processing_time_ms"
        ]
        
        with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    
    def analyze_frame(self, 
                     frame_number: int,
                     detections: List,
                     frame_width: int,
                     frame_height: int,
                     processing_time_ms: float,
                     timestamp: Optional[float] = None) -> FrameAnalysis:
        """
        分析单帧数据
        
        Args:
            frame_number: 帧号
            detections: 检测结果列表
            frame_width: 帧宽度
            frame_height: 帧高度
            processing_time_ms: 处理时间（毫秒）
            timestamp: 时间戳
            
        Returns:
            FrameAnalysis: 帧分析结果
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 创建物体信息列表
        objects = []
        
        for detection in detections:
            # 提取基本检测信息
            bbox = detection.bbox
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            # 提取距离信息
            distance_meters = None
            distance_confidence = None
            depth_quality = None

            # 首先尝试从midas_distance获取（如果存在）
            if hasattr(detection, 'midas_distance') and detection.midas_distance:
                distance_meters = detection.midas_distance.distance_meters
                distance_confidence = detection.midas_distance.confidence
                depth_quality = detection.midas_distance.depth_quality
            # 如果没有midas_distance，尝试从distance属性获取
            elif hasattr(detection, 'distance') and detection.distance is not None:
                distance_meters = detection.distance
                distance_confidence = 1.0  # 默认置信度
                depth_quality = 1.0  # 默认深度质量
            
            # 提取速度信息
            speed_mps = None
            speed_kmh = None
            velocity_x = None
            velocity_y = None
            acceleration = None
            object_id = None
            
            if hasattr(detection, 'speed_info') and detection.speed_info:
                speed_info = detection.speed_info
                speed_mps = speed_info.speed_mps
                speed_kmh = speed_info.speed_kmh
                velocity_x = getattr(speed_info, 'velocity_x', None)
                velocity_y = getattr(speed_info, 'velocity_y', None)
                acceleration = getattr(speed_info, 'acceleration', None)
                object_id = getattr(speed_info, 'object_id', None)
            
            # 创建物体信息
            obj_info = FrameObjectInfo(
                frame_number=frame_number,
                timestamp=timestamp,
                object_id=object_id,
                class_name=detection.class_name,
                class_id=detection.class_id,
                confidence=detection.confidence,
                bbox_x1=int(bbox[0]),
                bbox_y1=int(bbox[1]),
                bbox_x2=int(bbox[2]),
                bbox_y2=int(bbox[3]),
                center_x=center_x,
                center_y=center_y,
                width=int(width),
                height=int(height),
                area=int(area),
                distance_meters=distance_meters,
                distance_confidence=distance_confidence,
                depth_quality=depth_quality,
                speed_mps=speed_mps,
                speed_kmh=speed_kmh,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                acceleration=acceleration,
                frame_width=frame_width,
                frame_height=frame_height,
                processing_time_ms=processing_time_ms
            )
            
            objects.append(obj_info)
        
        # 创建帧分析结果
        frame_analysis = FrameAnalysis(
            frame_number=frame_number,
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
            total_objects=len(objects),
            processing_time_ms=processing_time_ms,
            objects=objects
        )
        
        # 存储分析数据
        self.frames_data.append(frame_analysis)
        
        # 更新统计信息
        self.session_stats["frames_processed"] += 1
        self.session_stats["total_objects_detected"] += len(objects)
        
        # 记录到CSV
        if self.enable_csv_log:
            self._write_to_csv(objects)
        
        # 不再写逐帧详细日志
        
        return frame_analysis
    
    def _write_to_csv(self, objects: List[FrameObjectInfo]):
        """写入CSV文件"""
        if not self.enable_csv_log:
            return
        
        with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for obj in objects:
                row = [
                    obj.frame_number, obj.timestamp, obj.object_id, obj.class_name, obj.class_id, obj.confidence,
                    obj.bbox_x1, obj.bbox_y1, obj.bbox_x2, obj.bbox_y2, obj.center_x, obj.center_y,
                    obj.width, obj.height, obj.area,
                    obj.distance_meters, obj.distance_confidence, obj.depth_quality,
                    obj.speed_mps, obj.speed_kmh, obj.velocity_x, obj.velocity_y, obj.acceleration,
                    obj.frame_width, obj.frame_height, obj.processing_time_ms
                ]
                writer.writerow(row)
    
    def finalize_session(self):
        """完成分析会话，生成最终报告"""
        session_end_time = time.time()
        session_duration = session_end_time - self.session_start_time
        
        # 更新会话统计
        self.session_stats["session_duration_seconds"] = session_duration
        
        if self.session_stats["frames_processed"] > 0:
            self.session_stats["average_objects_per_frame"] = (
                self.session_stats["total_objects_detected"] / self.session_stats["frames_processed"]
            )
            self.session_stats["processing_fps"] = (
                self.session_stats["frames_processed"] / session_duration
            )
        
        # 生成JSON报告
        if self.enable_json_log:
            self._generate_json_report(session_end_time)
        
        self.logger.info(f"Session {self.session_id} completed: "
                        f"{self.session_stats['frames_processed']} frames, "
                        f"{self.session_stats['total_objects_detected']} objects, "
                        f"{session_duration:.2f}s")
    
    def _generate_json_report(self, session_end_time: float):
        """生成JSON格式的完整报告"""
        report = {
            "session_id": self.session_id,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.session_start_time)),
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(session_end_time)),
            "frames": [
                {
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp,
                    "frame_width": frame.frame_width,
                    "frame_height": frame.frame_height,
                    "total_objects": frame.total_objects,
                    "processing_time_ms": frame.processing_time_ms,
                    "objects": [asdict(obj) for obj in frame.objects]
                }
                for frame in self.frames_data
            ],
            "session_summary": {
                "session_id": self.session_id,
                **self.session_stats
            }
        }
        
        json_file_path = self.log_dir / f"frame_analysis_{self.session_id}.json"
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON report saved: {json_file_path}")
