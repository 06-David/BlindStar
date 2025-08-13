"""
视频帧分析模块
记录和分析视频处理过程中每一帧的详细信息，包括物体检测、距离测量和速度信息
支持基础分析和高级分析（包含物体跟踪）
"""

import os
import json
import csv
import logging
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import cv2

logger = logging.getLogger(__name__)

@dataclass
class FrameObjectInfo:
    """单个物体的帧信息 - 兼容基础和高级分析"""
    frame_number: int
    timestamp: float
    object_id: Optional[int]
    class_name: str
    class_id: int
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    center_x: float
    center_y: float
    width: float
    height: float
    area: float

    # 距离信息
    distance_meters: Optional[float] = None
    distance_confidence: Optional[float] = None
    depth_quality: Optional[float] = None
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None
    median_depth: Optional[float] = None

    # 速度信息
    speed_mps: Optional[float] = None
    speed_kmh: Optional[float] = None
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    acceleration: Optional[float] = None
    speed_pixels_per_second: Optional[float] = None

    # 运动信息
    movement_direction: Optional[float] = None  # 运动方向角度 (度)
    movement_status: Optional[str] = None  # 运动状态: "静止", "移动", "加速", "减速"

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
    error_message: Optional[str] = None


@dataclass
class VideoAnalysisStats:
    """视频分析统计信息"""
    video_name: str
    total_frames: int
    processed_frames: int
    successful_frames: int
    failed_frames: int
    total_detections: int
    unique_classes: List[str]
    processing_time_seconds: float
    average_fps: float
    analysis_start_time: str
    analysis_end_time: str
    output_directory: str

class FrameAnalyzer:
    """视频帧分析器 - 支持基础和高级分析"""

    def __init__(self,
                 log_dir: str = "logs/frame_analysis",
                 enable_json_log: bool = True,
                 enable_csv_log: bool = True,
                 log_level: str = "INFO",
                 enable_tracking: bool = False,
                 fps: float = 30.0):
        """
        初始化帧分析器

        Args:
            log_dir: 日志目录
            enable_json_log: 启用JSON格式日志
            enable_csv_log: 启用CSV格式日志
            log_level: 日志级别
            enable_tracking: 是否启用物体跟踪（高级分析）
            fps: 视频帧率（用于速度计算）
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.enable_json_log = enable_json_log
        self.enable_csv_log = enable_csv_log
        self.enable_tracking = enable_tracking
        self.fps = fps

        # 生成会话ID
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()

        # 存储分析数据
        self.frames_data = []
        self.all_detections: List[FrameObjectInfo] = []  # 用于高级分析

        # 会话统计：使用 float 初始值以免类型检查警告
        self.session_stats: Dict[str, float] = {
            "frames_processed": 0,
            "total_objects_detected": 0,
            "session_duration_seconds": 0.0,
            "average_objects_per_frame": 0.0,
            "processing_fps": 0.0
        }

        # 物体跟踪器（高级分析）
        self.tracker = None
        if enable_tracking:
            try:
                from .object_tracker import SimpleObjectTracker
                self.tracker = SimpleObjectTracker(
                    max_distance_threshold=100.0,
                    max_missing_frames=30,
                    min_tracking_frames=3,
                    fps=fps
                )
                logger.info("物体跟踪器已启用")
            except ImportError:
                logger.warning("物体跟踪器模块未找到，跳过跟踪功能")
                self.enable_tracking = False
        
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
        # 定义所有字段（包含高级分析字段）
        fieldnames = [
            'frame_number', 'timestamp', 'object_id',
            'class_id', 'class_name', 'confidence',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'center_x', 'center_y', 'width', 'height', 'area',
            'distance_meters', 'distance_confidence', 'depth_quality',
            'min_depth', 'max_depth', 'median_depth',
            'speed_mps', 'speed_kmh', 'velocity_x', 'velocity_y', 'acceleration',
            'speed_pixels_per_second', 'movement_direction', 'movement_status',
            'frame_width', 'frame_height', 'processing_time_ms'
        ]

        # 中文表头映射
        chinese_headers = {
            'frame_number': '帧编号',
            'timestamp': '时间戳',
            'object_id': '物体ID',
            'class_id': '类别ID',
            'class_name': '物体类别',
            'confidence': '检测置信度',
            'bbox_x1': '边界框X1',
            'bbox_y1': '边界框Y1',
            'bbox_x2': '边界框X2',
            'bbox_y2': '边界框Y2',
            'center_x': '中心点X',
            'center_y': '中心点Y',
            'width': '宽度',
            'height': '高度',
            'area': '面积',
            'distance_meters': '距离(米)',
            'distance_confidence': '距离置信度',
            'depth_quality': '深度质量',
            'min_depth': '最小深度(米)',
            'max_depth': '最大深度(米)',
            'median_depth': '中位数深度(米)',
            'speed_mps': '速度(米/秒)',
            'speed_kmh': '速度(公里/小时)',
            'velocity_x': 'X方向速度(像素/秒)',
            'velocity_y': 'Y方向速度(像素/秒)',
            'acceleration': '加速度(米/秒²)',
            'speed_pixels_per_second': '速度(像素/秒)',
            'movement_direction': '运动方向(度)',
            'movement_status': '运动状态',
            'frame_width': '帧宽度',
            'frame_height': '帧高度',
            'processing_time_ms': '处理时间(毫秒)'
        }

        with open(self.csv_file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # 写入中文表头
            writer.writerow(chinese_headers)
    
    def analyze_frame(self,
                     frame_number: int,
                     detections: List,
                     frame_width: int,
                     frame_height: int,
                     processing_time_ms: float,
                     timestamp: Optional[float] = None,
                     frame: Optional[np.ndarray] = None) -> FrameAnalysis:
        """
        分析单帧数据

        Args:
            frame_number: 帧号
            detections: 检测结果列表
            frame_width: 帧宽度
            frame_height: 帧高度
            processing_time_ms: 处理时间（毫秒）
            timestamp: 时间戳
            frame: 图像帧（用于高级分析）

        Returns:
            FrameAnalysis: 帧分析结果
        """
        if timestamp is None:
            timestamp = time.time()

        error_message = None
        objects = []

        try:
            # 如果启用了跟踪，先进行跟踪处理
            if self.enable_tracking and self.tracker:
                # 转换检测结果为字典格式（用于跟踪器）
                detection_dicts = []
                for detection in detections:
                    detection_dict = self._convert_detection_to_dict(detection)
                    if detection_dict:
                        detection_dicts.append(detection_dict)

                # 使用物体跟踪器更新跟踪信息
                enhanced_detections = self.tracker.update(
                    detection_dicts, frame_number, timestamp
                )

                # 处理增强后的检测结果
                for enhanced_detection in enhanced_detections:
                    obj_info = self._extract_enhanced_detection_info(
                        enhanced_detection, frame_number, timestamp,
                        frame_width, frame_height, processing_time_ms
                    )
                    if obj_info:
                        objects.append(obj_info)
                        self.all_detections.append(obj_info)
            else:
                # 基础分析模式
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
                        bbox_x1=float(bbox[0]),
                        bbox_y1=float(bbox[1]),
                        bbox_x2=float(bbox[2]),
                        bbox_y2=float(bbox[3]),
                        center_x=center_x,
                        center_y=center_y,
                        width=float(width),
                        height=float(height),
                        area=float(area),
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
                    self.all_detections.append(obj_info)

        except Exception as e:
            error_message = str(e)
            logger.error(f"分析帧 {frame_number} 时出错: {e}")

        # 创建帧分析结果
        frame_analysis = FrameAnalysis(
            frame_number=frame_number,
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
            total_objects=len(objects),
            processing_time_ms=processing_time_ms,
            objects=objects,
            error_message=error_message
        )

        # 存储分析数据
        self.frames_data.append(frame_analysis)

        # 更新统计信息
        self.session_stats["frames_processed"] += 1
        self.session_stats["total_objects_detected"] += len(objects)

        # 记录到CSV
        if self.enable_csv_log:
            self._write_to_csv(objects)

        # tqdm风格进度条已替代定期日志输出
        # 原有每100帧日志输出已移除，进度由tqdm显示

        return frame_analysis
    
    def _write_to_csv(self, objects: List[FrameObjectInfo]):
        """写入CSV文件"""
        if not self.enable_csv_log:
            return

        # 定义字段顺序
        fieldnames = [
            'frame_number', 'timestamp', 'object_id',
            'class_id', 'class_name', 'confidence',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'center_x', 'center_y', 'width', 'height', 'area',
            'distance_meters', 'distance_confidence', 'depth_quality',
            'min_depth', 'max_depth', 'median_depth',
            'speed_mps', 'speed_kmh', 'velocity_x', 'velocity_y', 'acceleration',
            'speed_pixels_per_second', 'movement_direction', 'movement_status',
            'frame_width', 'frame_height', 'processing_time_ms'
        ]

        with open(self.csv_file_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            for obj in objects:
                obj_dict = asdict(obj)
                # 确保所有字段都存在，缺失的用None填充
                row_data = {}
                for field in fieldnames:
                    row_data[field] = obj_dict.get(field, None)
                writer.writerow(row_data)
    
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

    def _convert_detection_to_dict(self, detection: Any) -> Optional[Dict]:
        """将检测结果转换为字典格式"""
        try:
            if isinstance(detection, dict):
                return detection
            else:
                # 处理对象格式的检测结果
                bbox = getattr(detection, 'bbox', [0, 0, 0, 0])
                center_x = (bbox[0] + bbox[2]) / 2 if len(bbox) >= 4 else 0
                center_y = (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0

                return {
                    'class_id': getattr(detection, 'class_id', -1),
                    'class_name': getattr(detection, 'class_name', 'unknown'),
                    'confidence': float(getattr(detection, 'confidence', 0.0)),
                    'bbox': bbox,
                    'center': [center_x, center_y],
                    'area': float(getattr(detection, 'area', (bbox[2]-bbox[0])*(bbox[3]-bbox[1]))),
                    'distance_meters': getattr(detection, 'distance', None),
                    'depth_confidence': getattr(detection, 'depth_confidence', None)
                }
        except Exception as e:
            logger.warning(f"转换检测结果失败: {e}")
            return None

    def _extract_enhanced_detection_info(self,
                                       detection_dict: Dict,
                                       frame_number: int,
                                       timestamp: float,
                                       frame_width: int,
                                       frame_height: int,
                                       processing_time_ms: float) -> Optional[FrameObjectInfo]:
        """从增强的检测字典中提取完整信息"""
        try:
            bbox = detection_dict.get('bbox', [0, 0, 0, 0])
            center = detection_dict.get('center', [0, 0])

            return FrameObjectInfo(
                frame_number=frame_number,
                timestamp=timestamp,
                object_id=detection_dict.get('object_id'),
                class_id=detection_dict.get('class_id', -1),
                class_name=detection_dict.get('class_name', 'unknown'),
                confidence=float(detection_dict.get('confidence', 0.0)),
                bbox_x1=float(bbox[0]),
                bbox_y1=float(bbox[1]),
                bbox_x2=float(bbox[2]),
                bbox_y2=float(bbox[3]),
                center_x=float(center[0]),
                center_y=float(center[1]),
                width=float(bbox[2] - bbox[0]),
                height=float(bbox[3] - bbox[1]),
                area=float(detection_dict.get('area', 0)),
                distance_meters=detection_dict.get('distance_meters'),
                distance_confidence=detection_dict.get('depth_confidence'),
                depth_quality=detection_dict.get('depth_quality'),
                min_depth=detection_dict.get('min_depth'),
                max_depth=detection_dict.get('max_depth'),
                median_depth=detection_dict.get('median_depth'),
                velocity_x=detection_dict.get('velocity_x'),
                velocity_y=detection_dict.get('velocity_y'),
                speed_pixels_per_second=detection_dict.get('speed_pixels_per_second'),
                speed_mps=detection_dict.get('speed_meters_per_second'),
                speed_kmh=detection_dict.get('speed_kmh'),
                acceleration=detection_dict.get('acceleration'),
                movement_direction=detection_dict.get('movement_direction'),
                movement_status=detection_dict.get('movement_status'),
                frame_width=frame_width,
                frame_height=frame_height,
                processing_time_ms=processing_time_ms
            )
        except Exception as e:
            logger.warning(f"提取增强检测信息失败: {e}")
            return None

    def finalize_video_analysis(self, video_name: str, total_frames: int) -> VideoAnalysisStats:
        """
        完成视频分析并生成统计信息（高级分析功能）

        Args:
            video_name: 视频文件名
            total_frames: 总帧数

        Returns:
            视频分析统计信息
        """
        end_time = time.time()
        processing_time = end_time - self.session_start_time

        # 计算统计信息
        processed_frames = len(self.frames_data)
        successful_frames = len([r for r in self.frames_data if r.error_message is None])
        failed_frames = processed_frames - successful_frames
        total_detections = len(self.all_detections)
        unique_classes = list(set(d.class_name for d in self.all_detections))

        # 创建统计信息
        stats = VideoAnalysisStats(
            video_name=video_name,
            total_frames=total_frames,
            processed_frames=processed_frames,
            successful_frames=successful_frames,
            failed_frames=failed_frames,
            total_detections=total_detections,
            unique_classes=unique_classes,
            processing_time_seconds=processing_time,
            average_fps=processed_frames / processing_time if processing_time > 0 else 0,
            analysis_start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.session_start_time)),
            analysis_end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            output_directory=str(self.log_dir)
        )

        # 保存高级分析结果
        self._save_advanced_results(stats)

        logger.info(f"视频分析完成: {video_name}")
        logger.info(f"  处理帧数: {processed_frames}/{total_frames}")
        logger.info(f"  成功帧数: {successful_frames}")
        logger.info(f"  失败帧数: {failed_frames}")
        logger.info(f"  总检测数: {total_detections}")
        logger.info(f"  检测类别: {', '.join(unique_classes) if unique_classes else '无'}")
        logger.info(f"  处理时间: {processing_time:.2f}s")
        logger.info(f"  平均FPS: {stats.average_fps:.2f}")

        return stats

    def _save_advanced_results(self, stats: VideoAnalysisStats):
        """保存高级分析结果到文件"""
        try:
            # 保存JSON格式的详细结果
            json_file = self.log_dir / f"advanced_analysis_{self.session_id}.json"
            json_data = {
                "session_id": self.session_id,
                "stats": asdict(stats),
                "frame_results": [asdict(result) for result in self.frames_data]
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

            # 保存统计摘要
            summary_file = self.log_dir / f"advanced_summary_{self.session_id}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"高级视频分析摘要\n")
                f.write(f"{'='*50}\n")
                f.write(f"视频文件: {stats.video_name}\n")
                f.write(f"分析时间: {stats.analysis_start_time} - {stats.analysis_end_time}\n")
                f.write(f"总帧数: {stats.total_frames}\n")
                f.write(f"处理帧数: {stats.processed_frames}\n")
                f.write(f"成功帧数: {stats.successful_frames}\n")
                f.write(f"失败帧数: {stats.failed_frames}\n")
                f.write(f"完成率: {(stats.processed_frames/stats.total_frames)*100:.1f}%\n")
                f.write(f"成功率: {(stats.successful_frames/stats.processed_frames)*100:.1f}%\n")
                f.write(f"总检测数: {stats.total_detections}\n")
                f.write(f"检测类别: {', '.join(stats.unique_classes) if stats.unique_classes else '无'}\n")
                f.write(f"处理时间: {stats.processing_time_seconds:.2f}秒\n")
                f.write(f"平均FPS: {stats.average_fps:.2f}\n")

            logger.info(f"高级分析结果已保存到: {self.log_dir}")
            logger.info(f"  JSON详细结果: {json_file.name}")
            logger.info(f"  文本摘要: {summary_file.name}")

        except Exception as e:
            logger.error(f"保存高级分析结果时出错: {e}")

    def get_progress_info(self) -> Dict[str, Any]:
        """获取当前进度信息"""
        processed_frames = len(self.frames_data)
        total_detections = len(self.all_detections)
        current_time = time.time()
        elapsed_time = current_time - self.session_start_time
        current_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0

        return {
            "processed_frames": processed_frames,
            "total_detections": total_detections,
            "elapsed_time": elapsed_time,
            "current_fps": current_fps,
            "session_id": self.session_id
        }

    def generate_session_report(self):
        """生成会话报告（向后兼容方法）"""
        logger.info("生成会话报告...")

        # 更新会话统计
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        frames_processed = len(self.frames_data)
        total_objects = len(self.all_detections)

        self.session_stats.update({
            "session_duration_seconds": session_duration,
            "average_objects_per_frame": total_objects / frames_processed if frames_processed > 0 else 0,
            "processing_fps": frames_processed / session_duration if session_duration > 0 else 0
        })

        # 保存会话报告
        report_file = self.log_dir / f"session_report_{self.session_id}.json"
        report_data = {
            "session_id": self.session_id,
            "session_stats": self.session_stats,
            "frames_processed": frames_processed,
            "total_detections": total_objects,
            "unique_classes": list(set(d.class_name for d in self.all_detections)),
            "session_start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.session_start_time)),
            "session_end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
            "log_directory": str(self.log_dir)
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"会话报告已保存: {report_file}")
        logger.info(f"处理帧数: {frames_processed}")
        logger.info(f"总检测数: {total_objects}")
        logger.info(f"会话时长: {session_duration:.2f}秒")
        logger.info(f"平均FPS: {self.session_stats['processing_fps']:.2f}")

    def draw_enhanced_detections(self, frame: np.ndarray, detections: List[FrameObjectInfo]) -> np.ndarray:
        """
        在帧上绘制增强的检测信息

        Args:
            frame: 输入帧
            detections: 检测结果列表

        Returns:
            标注后的帧
        """
        annotated_frame = frame.copy()

        # 定义颜色 (BGR格式)
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红色
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
        ]

        for detection in detections:
            # 获取颜色
            color_idx = detection.object_id % len(colors) if detection.object_id else 0
            color = colors[color_idx]

            # 绘制边界框
            x1, y1, x2, y2 = int(detection.bbox_x1), int(detection.bbox_y1), int(detection.bbox_x2), int(detection.bbox_y2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # 准备标签信息
            label_lines = []

            # 基本信息 - 使用英文避免编码问题
            confidence_text = f"{detection.confidence:.2f}"
            if detection.object_id:
                label_lines.append(f"ID:{detection.object_id} {detection.class_name} {confidence_text}")
            else:
                label_lines.append(f"{detection.class_name} {confidence_text}")

            # 距离信息
            if detection.distance_meters is not None:
                label_lines.append(f"Dist: {detection.distance_meters:.2f}m")

            # 速度信息
            if detection.speed_mps is not None:
                label_lines.append(f"Speed: {detection.speed_mps:.2f}m/s")

            # 绘制标签背景和文字 - 使用更robust的字体设置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # 计算标签尺寸
            max_width = 0
            total_height = 0
            line_heights = []

            for line in label_lines:
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                max_width = max(max_width, text_width)
                line_height = text_height + baseline + 5
                line_heights.append(line_height)
                total_height += line_height

            # 绘制标签背景 - 使用半透明黑色背景确保文字清晰
            label_x = x1
            label_y = y1 - total_height - 10
            if label_y < 0:
                label_y = y2 + 10

            # 绘制黑色背景
            cv2.rectangle(annotated_frame,
                         (label_x, label_y),
                         (label_x + max_width + 15, label_y + total_height + 10),
                         (0, 0, 0), -1)

            # 绘制彩色边框
            cv2.rectangle(annotated_frame,
                         (label_x, label_y),
                         (label_x + max_width + 15, label_y + total_height + 10),
                         color, 2)

            # 绘制标签文字
            current_y = label_y + 15
            for i, line in enumerate(label_lines):
                cv2.putText(annotated_frame, line,
                           (label_x + 5, current_y),
                           font, font_scale, (255, 255, 255), thickness)
                current_y += line_heights[i]

            # 绘制运动方向箭头
            if (detection.movement_direction is not None and
                detection.speed_pixels_per_second is not None and
                detection.speed_pixels_per_second > 5.0):  # 只有在明显移动时才绘制箭头

                center_x = int(detection.center_x)
                center_y = int(detection.center_y)

                # 计算箭头长度（基于速度）
                arrow_length = min(max(detection.speed_pixels_per_second * 0.5, 30), 80)

                # 转换角度（OpenCV使用的是数学坐标系）
                angle_rad = math.radians(detection.movement_direction)
                end_x = int(center_x + arrow_length * math.cos(angle_rad))
                end_y = int(center_y + arrow_length * math.sin(angle_rad))

                # 绘制箭头主线
                cv2.arrowedLine(annotated_frame,
                               (center_x, center_y),
                               (end_x, end_y),
                               color, 3, tipLength=0.3)

                # 在箭头旁边显示方向角度
                angle_text = f"{detection.movement_direction:.0f}°"
                text_x = end_x + 10
                text_y = end_y - 10
                cv2.putText(annotated_frame, angle_text,
                           (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated_frame


# 为了向后兼容，保留原有的类名别名
AdvancedFrameAnalyzer = FrameAnalyzer


class RobustVideoAnalyzer:
    """Robust视频分析器 - 使用合并后的FrameAnalyzer"""

    def __init__(self,
                 yolo_model: str = "small",
                 midas_model: str = "MiDaS_small",
                 device: str = "cuda",
                 confidence: float = 0.6,
                 input_size: Tuple[int, int] = (384, 512)):
        """
        初始化Robust视频分析器

        Args:
            yolo_model: YOLO模型变体
            midas_model: MiDaS模型变体
            device: 设备
            confidence: 置信度阈值
            input_size: 深度模型输入尺寸 (width, height)
        """
        self.yolo_model = yolo_model
        self.midas_model = midas_model
        self.device = device
        self.confidence = confidence
        self.input_size = input_size

        # 延迟初始化BlindStar（避免重复初始化）
        self.blindstar = None

        logger.info(f"Robust视频分析器创建完成")
        logger.info(f"  YOLO模型: {yolo_model}")
        logger.info(f"  深度模型: {midas_model}")
        logger.info(f"  设备: {device}")
        logger.info(f"  置信度: {confidence}")
        logger.info(f"  输入尺寸: {input_size}")

    def analyze_video(self,
                     video_path: str,
                     max_frames: Optional[int] = None,
                     output_video: bool = True,
                     output_video_path: Optional[str] = None) -> VideoAnalysisStats:
        """
        分析视频文件并生成标注视频

        Args:
            video_path: 视频文件路径
            max_frames: 最大处理帧数（None表示处理所有帧）
            output_video: 是否生成输出视频
            output_video_path: 输出视频路径（None则自动生成）

        Returns:
            视频分析统计信息
        """
        video_path_obj = Path(video_path)
        logger.info(f"开始分析视频: {video_path_obj.name}")

        # 初始化BlindStar
        if not self._initialize_blindstar():
            raise RuntimeError("BlindStar初始化失败")

        # 创建高级帧分析器
        output_dir = f"logs/robust_analysis_{video_path_obj.stem}_{int(time.time())}"
        analyzer = FrameAnalyzer(
            log_dir=output_dir,
            enable_tracking=True,  # 启用跟踪功能
            fps=30.0
        )

        # 打开视频
        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path_obj}")

        # 初始化视频写入器
        video_writer = None

        try:
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 确定实际处理的帧数
            frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

            logger.info(f"视频信息:")
            logger.info(f"  总帧数: {total_frames}")
            logger.info(f"  处理帧数: {frames_to_process}")
            logger.info(f"  帧率: {fps:.2f} FPS")
            logger.info(f"  分辨率: {width}x{height}")

            # 设置输出视频
            if output_video:
                if output_video_path is None:
                    output_video_path = str(Path(output_dir) / f"annotated_{video_path_obj.stem}.mp4")

                # 创建视频写入器
                fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # type: ignore
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                if not video_writer.isOpened():
                    logger.warning(f"无法创建输出视频: {output_video_path}")
                    video_writer = None
                else:
                    logger.info(f"  输出视频: {output_video_path}")

            # 逐帧处理
            frame_number = 0
            processed_count = 0

            while processed_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"无法读取帧 {frame_number}，视频可能已结束")
                    break

                try:
                    # 分析帧
                    frame_start_time = time.time()
                    result = self.blindstar.analyze_frame(frame)
                    processing_time_ms = (time.time() - frame_start_time) * 1000

                    # 记录分析结果
                    frame_result = analyzer.analyze_frame(
                        frame_number=frame_number,
                        detections=result['detections'],
                        frame_width=width,
                        frame_height=height,
                        processing_time_ms=processing_time_ms,
                        timestamp=time.time(),
                        frame=frame
                    )

                    # 生成标注帧并写入视频
                    if output_video and video_writer is not None:
                        annotated_frame = analyzer.draw_enhanced_detections(frame, frame_result.objects)
                        video_writer.write(annotated_frame)

                    processed_count += 1

                    # 进度报告
                    if processed_count % 200 == 0:
                        progress = analyzer.get_progress_info()
                        completion = (processed_count / frames_to_process) * 100
                        remaining_frames = frames_to_process - processed_count
                        eta = remaining_frames / progress['current_fps'] if progress['current_fps'] > 0 else 0

                        logger.info(f"进度: {completion:.1f}% ({processed_count}/{frames_to_process})")
                        logger.info(f"  当前FPS: {progress['current_fps']:.1f}")
                        logger.info(f"  已检测: {progress['total_detections']} 个物体")
                        logger.info(f"  预计剩余: {eta:.1f}秒")

                except Exception as e:
                    logger.error(f"处理帧 {frame_number} 时出错: {e}")
                    # 继续处理下一帧，不中断整个分析
                    processed_count += 1

                frame_number += 1

            # 完成分析
            stats = analyzer.finalize_video_analysis(video_path_obj.name, total_frames)

            return stats

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                logger.info(f"输出视频已保存: {output_video_path}")
            if self.blindstar:
                self.blindstar.cleanup()
                self.blindstar = None

    def _initialize_blindstar(self) -> bool:
        """初始化BlindStar"""
        try:
            from . import BlindStar

            self.blindstar = BlindStar(
                yolo_model=self.yolo_model,
                midas_model=self.midas_model,
                confidence_threshold=self.confidence,
                enable_distance=True,
                device=self.device
            )

            return self.blindstar.initialize()

        except Exception as e:
            logger.error(f"BlindStar初始化失败: {e}")
            return False
