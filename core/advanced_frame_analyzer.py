#!/usr/bin/env python3
"""
高级帧分析器
重新设计的robust帧分析系统，具备完整性、准确性和可靠性
"""

import os
import json
import csv
import time
import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import cv2
import numpy as np

from .object_tracker import SimpleObjectTracker

logger = logging.getLogger(__name__)


@dataclass
class FrameDetection:
    """单个检测结果的数据结构 - 包含完整的物体信息"""
    # 基本信息
    frame_number: int
    timestamp: float
    object_id: Optional[int] = None  # 物体跟踪ID

    # 检测信息
    class_id: int = -1
    class_name: str = "unknown"
    confidence: float = 0.0

    # 位置信息
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    area: float = 0.0

    # 距离信息 (MiDaS)
    distance_meters: Optional[float] = None
    depth_confidence: Optional[float] = None
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None
    median_depth: Optional[float] = None

    # 速度信息
    velocity_x: Optional[float] = None  # X方向速度 (像素/秒)
    velocity_y: Optional[float] = None  # Y方向速度 (像素/秒)
    speed_pixels_per_second: Optional[float] = None  # 总速度 (像素/秒)
    speed_meters_per_second: Optional[float] = None  # 实际速度 (米/秒)
    speed_kmh: Optional[float] = None  # 速度 (公里/小时)
    acceleration: Optional[float] = None  # 加速度 (米/秒²)

    # 运动方向
    movement_direction: Optional[float] = None  # 运动方向角度 (度)
    movement_status: Optional[str] = None  # 运动状态: "静止", "移动", "加速", "减速"


@dataclass
class FrameAnalysisResult:
    """单帧分析结果"""
    frame_number: int
    timestamp: float
    frame_width: int
    frame_height: int
    processing_time_ms: float
    detections: List[FrameDetection]
    total_objects: int
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


class AdvancedFrameAnalyzer:
    """高级帧分析器 - 支持物体跟踪和速度计算"""

    def __init__(self,
                 output_dir: str = "logs/advanced_analysis",
                 enable_tracking: bool = True,
                 fps: float = 30.0):
        """
        初始化高级帧分析器

        Args:
            output_dir: 输出目录
            enable_tracking: 是否启用物体跟踪
            fps: 视频帧率（用于速度计算）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 分析数据存储
        self.frame_results: List[FrameAnalysisResult] = []
        self.all_detections: List[FrameDetection] = []
        self.stats: Optional[VideoAnalysisStats] = None

        # 物体跟踪器
        self.enable_tracking = enable_tracking
        self.fps = fps
        if enable_tracking:
            self.tracker = SimpleObjectTracker(
                max_distance_threshold=100.0,
                max_missing_frames=30,
                min_tracking_frames=3,
                fps=fps
            )
        else:
            self.tracker = None

        # 会话信息
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()

        logger.info(f"高级帧分析器初始化完成，会话ID: {self.session_id}")
        logger.info(f"  物体跟踪: {'启用' if enable_tracking else '禁用'}")
        logger.info(f"  视频帧率: {fps} FPS")
    
    def analyze_frame(self,
                     frame_number: int,
                     frame: np.ndarray,
                     detections: List[Any],
                     processing_time_ms: float,
                     timestamp: Optional[float] = None) -> FrameAnalysisResult:
        """
        分析单帧 - 包含物体跟踪和速度计算

        Args:
            frame_number: 帧编号
            frame: 图像帧
            detections: 检测结果列表
            processing_time_ms: 处理时间（毫秒）
            timestamp: 时间戳

        Returns:
            帧分析结果
        """
        if timestamp is None:
            timestamp = time.time()

        frame_height, frame_width = frame.shape[:2]
        frame_detections = []
        error_message = None

        try:
            # 转换检测结果为字典格式（用于跟踪器）
            detection_dicts = []
            for detection in detections:
                detection_dict = self._convert_detection_to_dict(detection)
                if detection_dict:
                    detection_dicts.append(detection_dict)

            # 使用物体跟踪器更新跟踪信息
            if self.enable_tracking and self.tracker:
                enhanced_detections = self.tracker.update(
                    detection_dicts, frame_number, timestamp
                )
            else:
                enhanced_detections = detection_dicts

            # 处理增强后的检测结果
            for enhanced_detection in enhanced_detections:
                try:
                    # 提取完整的检测信息（包含跟踪信息）
                    frame_detection = self._extract_enhanced_detection_info(
                        enhanced_detection, frame_number, timestamp
                    )
                    if frame_detection:
                        frame_detections.append(frame_detection)
                        self.all_detections.append(frame_detection)

                except Exception as e:
                    logger.warning(f"处理增强检测结果时出错 (帧 {frame_number}): {e}")
                    continue

        except Exception as e:
            error_message = str(e)
            logger.error(f"分析帧 {frame_number} 时出错: {e}")

        # 创建帧分析结果
        result = FrameAnalysisResult(
            frame_number=frame_number,
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
            processing_time_ms=processing_time_ms,
            detections=frame_detections,
            total_objects=len(frame_detections),
            error_message=error_message
        )

        self.frame_results.append(result)

        # 定期日志输出（包含跟踪统计）
        if frame_number % 100 == 0:
            tracking_info = ""
            if self.tracker:
                stats = self.tracker.get_tracking_stats()
                tracking_info = f", 跟踪物体: {stats['active_objects']}/{stats['total_tracked_objects']}"

            logger.info(f"已分析帧 {frame_number}, 检测到 {len(frame_detections)} 个物体{tracking_info}")

        return result

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
                    'distance_meters': getattr(detection, 'distance_meters', None),
                    'depth_confidence': getattr(detection, 'depth_confidence', None)
                }
        except Exception as e:
            logger.warning(f"转换检测结果失败: {e}")
            return None

    def _extract_enhanced_detection_info(self,
                                       detection_dict: Dict,
                                       frame_number: int,
                                       timestamp: float) -> Optional[FrameDetection]:
        """从增强的检测字典中提取完整信息"""
        try:
            bbox = detection_dict.get('bbox', [0, 0, 0, 0])
            center = detection_dict.get('center', [0, 0])

            return FrameDetection(
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
                depth_confidence=detection_dict.get('depth_confidence'),
                min_depth=detection_dict.get('min_depth'),
                max_depth=detection_dict.get('max_depth'),
                median_depth=detection_dict.get('median_depth'),
                velocity_x=detection_dict.get('velocity_x'),
                velocity_y=detection_dict.get('velocity_y'),
                speed_pixels_per_second=detection_dict.get('speed_pixels_per_second'),
                speed_meters_per_second=detection_dict.get('speed_meters_per_second'),
                speed_kmh=detection_dict.get('speed_kmh'),
                acceleration=detection_dict.get('acceleration'),
                movement_direction=detection_dict.get('movement_direction'),
                movement_status=detection_dict.get('movement_status')
            )
        except Exception as e:
            logger.warning(f"提取增强检测信息失败: {e}")
            return None

    def _extract_detection_info(self,
                               detection: Any,
                               frame_number: int,
                               timestamp: float) -> Optional[FrameDetection]:
        """
        安全地提取检测信息
        
        Args:
            detection: 检测结果对象
            frame_number: 帧编号
            timestamp: 时间戳
            
        Returns:
            检测信息或None
        """
        try:
            # 处理字典格式的检测结果
            if isinstance(detection, dict):
                bbox = detection.get('bbox', [0, 0, 0, 0])
                center = detection.get('center', [0, 0])
                
                return FrameDetection(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    class_id=detection.get('class_id', -1),
                    class_name=detection.get('class_name', 'unknown'),
                    confidence=float(detection.get('confidence', 0.0)),
                    bbox_x1=float(bbox[0]),
                    bbox_y1=float(bbox[1]),
                    bbox_x2=float(bbox[2]),
                    bbox_y2=float(bbox[3]),
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    width=float(bbox[2] - bbox[0]),
                    height=float(bbox[3] - bbox[1]),
                    area=float(detection.get('area', 0)),
                    distance_meters=detection.get('distance_meters'),
                    depth_confidence=detection.get('depth_confidence'),
                    min_depth=detection.get('min_depth'),
                    max_depth=detection.get('max_depth'),
                    median_depth=detection.get('median_depth')
                )
            
            # 处理对象格式的检测结果
            else:
                bbox = getattr(detection, 'bbox', [0, 0, 0, 0])
                center_x = (bbox[0] + bbox[2]) / 2 if len(bbox) >= 4 else 0
                center_y = (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0
                width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
                height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
                
                return FrameDetection(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    class_id=getattr(detection, 'class_id', -1),
                    class_name=getattr(detection, 'class_name', 'unknown'),
                    confidence=float(getattr(detection, 'confidence', 0.0)),
                    bbox_x1=float(bbox[0]),
                    bbox_y1=float(bbox[1]),
                    bbox_x2=float(bbox[2]),
                    bbox_y2=float(bbox[3]),
                    center_x=float(center_x),
                    center_y=float(center_y),
                    width=float(width),
                    height=float(height),
                    area=float(getattr(detection, 'area', width * height)),
                    distance_meters=getattr(detection, 'distance_meters', None),
                    depth_confidence=getattr(detection, 'depth_confidence', None),
                    min_depth=getattr(detection, 'min_depth', None),
                    max_depth=getattr(detection, 'max_depth', None),
                    median_depth=getattr(detection, 'median_depth', None)
                )
        
        except Exception as e:
            logger.warning(f"提取检测信息失败: {e}")
            return None
    
    def finalize_analysis(self, video_name: str, total_frames: int) -> VideoAnalysisStats:
        """
        完成分析并生成统计信息
        
        Args:
            video_name: 视频文件名
            total_frames: 总帧数
            
        Returns:
            视频分析统计信息
        """
        end_time = time.time()
        processing_time = end_time - self.start_time
        
        # 计算统计信息
        processed_frames = len(self.frame_results)
        successful_frames = len([r for r in self.frame_results if r.error_message is None])
        failed_frames = processed_frames - successful_frames
        total_detections = len(self.all_detections)
        unique_classes = list(set(d.class_name for d in self.all_detections))
        
        # 创建统计信息
        self.stats = VideoAnalysisStats(
            video_name=video_name,
            total_frames=total_frames,
            processed_frames=processed_frames,
            successful_frames=successful_frames,
            failed_frames=failed_frames,
            total_detections=total_detections,
            unique_classes=unique_classes,
            processing_time_seconds=processing_time,
            average_fps=processed_frames / processing_time if processing_time > 0 else 0,
            analysis_start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            analysis_end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            output_directory=str(self.output_dir)
        )
        
        # 保存分析结果
        self._save_results()
        
        logger.info(f"分析完成: {video_name}")
        logger.info(f"  处理帧数: {processed_frames}/{total_frames}")
        logger.info(f"  成功帧数: {successful_frames}")
        logger.info(f"  失败帧数: {failed_frames}")
        logger.info(f"  总检测数: {total_detections}")
        logger.info(f"  检测类别: {', '.join(unique_classes) if unique_classes else '无'}")
        logger.info(f"  处理时间: {processing_time:.2f}s")
        logger.info(f"  平均FPS: {self.stats.average_fps:.2f}")
        
        return self.stats
    
    def _save_results(self):
        """保存分析结果到文件"""
        try:
            # 保存JSON格式的详细结果
            json_file = self.output_dir / f"analysis_{self.session_id}.json"
            json_data = {
                "session_id": self.session_id,
                "stats": asdict(self.stats) if self.stats else {},
                "frame_results": [asdict(result) for result in self.frame_results]
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存CSV格式的检测数据
            csv_file = self.output_dir / f"detections_{self.session_id}.csv"
            if self.all_detections:
                # 定义CSV字段和中文表头的映射
                fieldnames = [
                    'frame_number', 'timestamp', 'object_id',
                    'class_id', 'class_name', 'confidence',
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                    'center_x', 'center_y', 'width', 'height', 'area',
                    'distance_meters', 'depth_confidence', 'min_depth', 'max_depth', 'median_depth',
                    'velocity_x', 'velocity_y', 'speed_pixels_per_second',
                    'speed_meters_per_second', 'speed_kmh', 'acceleration',
                    'movement_direction', 'movement_status'
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
                    'depth_confidence': '深度置信度',
                    'min_depth': '最小深度(米)',
                    'max_depth': '最大深度(米)',
                    'median_depth': '中位数深度(米)',
                    'velocity_x': 'X方向速度(像素/秒)',
                    'velocity_y': 'Y方向速度(像素/秒)',
                    'speed_pixels_per_second': '速度(像素/秒)',
                    'speed_meters_per_second': '速度(米/秒)',
                    'speed_kmh': '速度(公里/小时)',
                    'acceleration': '加速度(米/秒²)',
                    'movement_direction': '运动方向(度)',
                    'movement_status': '运动状态'
                }

                with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)

                    # 写入中文表头
                    writer.writerow(chinese_headers)

                    # 写入数据
                    for detection in self.all_detections:
                        detection_dict = asdict(detection)
                        # 确保所有字段都存在，缺失的用None填充
                        row_data = {}
                        for field in fieldnames:
                            row_data[field] = detection_dict.get(field, None)
                        writer.writerow(row_data)
            
            # 保存统计摘要
            summary_file = self.output_dir / f"summary_{self.session_id}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                if self.stats:
                    f.write(f"视频分析摘要\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"视频文件: {self.stats.video_name}\n")
                    f.write(f"分析时间: {self.stats.analysis_start_time} - {self.stats.analysis_end_time}\n")
                    f.write(f"总帧数: {self.stats.total_frames}\n")
                    f.write(f"处理帧数: {self.stats.processed_frames}\n")
                    f.write(f"成功帧数: {self.stats.successful_frames}\n")
                    f.write(f"失败帧数: {self.stats.failed_frames}\n")
                    f.write(f"完成率: {(self.stats.processed_frames/self.stats.total_frames)*100:.1f}%\n")
                    f.write(f"成功率: {(self.stats.successful_frames/self.stats.processed_frames)*100:.1f}%\n")
                    f.write(f"总检测数: {self.stats.total_detections}\n")
                    f.write(f"检测类别: {', '.join(self.stats.unique_classes) if self.stats.unique_classes else '无'}\n")
                    f.write(f"处理时间: {self.stats.processing_time_seconds:.2f}秒\n")
                    f.write(f"平均FPS: {self.stats.average_fps:.2f}\n")
            
            logger.info(f"分析结果已保存到: {self.output_dir}")
            logger.info(f"  JSON详细结果: {json_file.name}")
            logger.info(f"  CSV检测数据: {csv_file.name}")
            logger.info(f"  文本摘要: {summary_file.name}")
        
        except Exception as e:
            logger.error(f"保存分析结果时出错: {e}")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """获取当前进度信息"""
        processed_frames = len(self.frame_results)
        total_detections = len(self.all_detections)
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        current_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "processed_frames": processed_frames,
            "total_detections": total_detections,
            "elapsed_time": elapsed_time,
            "current_fps": current_fps,
            "session_id": self.session_id
        }

    def draw_enhanced_detections(self, frame: np.ndarray, detections: List[FrameDetection]) -> np.ndarray:
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
            if detection.speed_meters_per_second is not None:
                label_lines.append(f"Speed: {detection.speed_meters_per_second:.2f}m/s")

            # 运动状态 - 暂时不显示
            # if detection.movement_status:
            #     status_map = {
            #         "静止": "Still",
            #         "匀速": "Uniform",
            #         "加速": "Accel",
            #         "减速": "Decel"
            #     }
            #     status = status_map.get(detection.movement_status, detection.movement_status)
            #     label_lines.append(f"Status: {status}")

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


class RobustVideoAnalyzer:
    """Robust视频分析器"""

    def __init__(self,
                 yolo_model: str = "small",
                 midas_model: str = "MiDaS_small",
                 device: str = "cuda",
                 confidence: float = 0.6):
        """
        初始化Robust视频分析器

        Args:
            yolo_model: YOLO模型变体
            midas_model: MiDaS模型变体
            device: 设备
            confidence: 置信度阈值
        """
        self.yolo_model = yolo_model
        self.midas_model = midas_model
        self.device = device
        self.confidence = confidence

        # 延迟初始化BlindStar（避免重复初始化）
        self.blindstar = None

        logger.info(f"Robust视频分析器创建完成")
        logger.info(f"  YOLO模型: {yolo_model}")
        logger.info(f"  MiDaS模型: {midas_model}")
        logger.info(f"  设备: {device}")
        logger.info(f"  置信度: {confidence}")

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
        video_path = Path(video_path)
        logger.info(f"开始分析视频: {video_path.name}")

        # 初始化BlindStar
        if not self._initialize_blindstar():
            raise RuntimeError("BlindStar初始化失败")

        # 创建高级帧分析器
        output_dir = f"logs/robust_analysis_{video_path.stem}_{int(time.time())}"
        analyzer = AdvancedFrameAnalyzer(output_dir)

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

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
                    output_video_path = str(Path(output_dir) / f"annotated_{video_path.stem}.mp4")

                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                        frame=frame,
                        detections=result['detections'],
                        processing_time_ms=processing_time_ms,
                        timestamp=time.time()
                    )

                    # 生成标注帧并写入视频
                    if output_video and video_writer is not None:
                        annotated_frame = analyzer.draw_enhanced_detections(frame, frame_result.detections)
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
            stats = analyzer.finalize_analysis(video_path.name, total_frames)

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
            from core import BlindStar

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
