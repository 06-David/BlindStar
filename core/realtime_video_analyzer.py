#!/usr/bin/env python3
"""
实时视频决策分析器 - BlindStar核心组件
集成YOLO检测、深度估计和决策系统，提供实时视频流分析和语音播报
"""

import cv2
import numpy as np
import time
import threading
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import queue
from collections import deque

from .detector import YOLOv8Detector, DetectionResult
from .distance import ZoeDepthDistanceMeasurement
from .decision_engine import DecisionEngine, create_decision_context
from .tts_engine import TTSEngine
from .navigation_context import NavigationContextManager, NavigationMode, GPSLocation

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysisResult:
    """单帧分析结果"""
    frame_id: int
    timestamp: float
    detections: List[Dict[str, Any]]
    depth_map: Optional[np.ndarray]
    decision_output: Optional[Dict[str, Any]]
    navigation_output: Optional[Dict[str, Any]]
    processing_time: float
    fps: float


@dataclass
class VideoAnalysisStats:
    """视频分析统计信息"""
    total_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    average_fps: float = 0.0
    average_processing_time: float = 0.0
    detection_count: int = 0
    hazard_count: int = 0
    decision_count: int = 0


class RealtimeVideoAnalyzer:
    """实时视频决策分析器"""
    
    def __init__(self,
                 yolo_model: str = "small",
                 depth_model: str = "MiDaS_small",
                 confidence_threshold: float = 0.6,
                 enable_depth: bool = True,
                 enable_tts: bool = True,
                 max_fps: float = 15.0,
                 buffer_size: int = 5,
                 enable_navigation: bool = False,
                 nav_mode: str = "assist"):
        """
        初始化实时视频分析器
        
        Args:
            yolo_model: YOLO模型大小
            depth_model: 深度估计模型
            confidence_threshold: 检测置信度阈值
            enable_depth: 是否启用深度估计
            enable_tts: 是否启用语音播报
            max_fps: 最大处理帧率
        enable_navigation: 是否启用导航
        nav_mode: 导航模式（assist/guide/full）
            buffer_size: 帧缓冲区大小
            enable_navigation: 是否启用导航集成
            nav_mode: 导航模式（assist/guide/full）
        """
        self.max_fps = max_fps
        self.frame_interval = 1.0 / max_fps
        self.buffer_size = buffer_size
        self.enable_depth = enable_depth
        self.enable_tts = enable_tts
        self.enable_navigation = enable_navigation
        self.nav_mode = nav_mode
        
        # 初始化组件
        self.detector = None
        self.depth_estimator = None
        self.decision_engine = None
        self.tts_engine = None
        self.nav_context = None
        
        # 存储初始化参数
        self.yolo_model = yolo_model
        self.depth_model = depth_model
        self.confidence_threshold = confidence_threshold
        
        # 处理状态
        self.is_running = False
        self.is_initialized = False
        
        # 帧缓冲和处理队列
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)
        
        # 统计信息
        self.stats = VideoAnalysisStats()
        self.processing_times = deque(maxlen=100)  # 保存最近100帧的处理时间
        
        # 线程控制
        self.processing_thread = None
        self.display_thread = None
        
        # 导航语音节流
        self._last_nav_tts_time = 0.0
        self._nav_tts_interval = 5.0
        
        logger.info("✅ 实时视频分析器创建完成")
    
    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            logger.info("🚀 初始化实时视频分析器组件...")
            
            # 初始化YOLO检测器
            logger.info("初始化YOLO检测器...")
            self.detector = YOLOv8Detector(
                model_variant=self.yolo_model,
                confidence_threshold=self.confidence_threshold
            )
            
            # 初始化深度估计器
            if self.enable_depth:
                logger.info("初始化深度估计器...")
                self.depth_estimator = ZoeDepthDistanceMeasurement(
                    model_type=self.depth_model
                )
                self.depth_estimator.ensure_model_loaded()
            
            # 初始化决策引擎
            logger.info("初始化决策引擎...")
            self.decision_engine = DecisionEngine(enable_logging=False)
            
            # 初始化语音引擎
            if self.enable_tts:
                try:
                    logger.info("初始化语音引擎...")
                    self.tts_engine = TTSEngine()
                except Exception as e:
                    logger.warning(f"语音引擎初始化失败: {e}")
                    self.enable_tts = False
            
            # 初始化导航上下文
            if self.enable_navigation:
                try:
                    mode_map = {
                        "assist": NavigationMode.ASSIST,
                        "guide": NavigationMode.GUIDE,
                        "full": NavigationMode.FULL,
                    }
                    mode_enum = mode_map.get((self.nav_mode or "assist").lower(), NavigationMode.ASSIST)
                    self.nav_context = NavigationContextManager(mode=mode_enum)
                    logger.info(f"初始化导航上下文: 模式 {mode_enum.value}")
                except Exception as e:
                    logger.warning(f"导航上下文初始化失败: {e}")
                    self.enable_navigation = False
            
            self.is_initialized = True
            logger.info("✅ 所有组件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            return False
    
    def _process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> FrameAnalysisResult:
        """处理单帧"""
        start_time = time.time()
        
        # YOLO检测
        detections = self.detector.detect(frame)
        detection_data = []
        
        for detection in detections:
            det_dict = {
                'class_id': detection.class_id,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'bbox': detection.bbox,
                'center': detection.center,
                'area': detection.area
            }
            detection_data.append(det_dict)
        
        # 深度估计
        depth_map = None
        if self.enable_depth and self.depth_estimator:
            try:
                depth_map = self.depth_estimator.predict_depth(frame)
                
                # 为检测物体添加距离信息
                if detections and depth_map is not None:
                    depth_infos = self.depth_estimator.calculate_distances_batch(frame, detections)
                    for i, (detection, depth_info) in enumerate(zip(detections, depth_infos)):
                        if i < len(detection_data):
                            detection_data[i]['distance_m'] = depth_info.distance_meters
                            detection_data[i]['depth_confidence'] = depth_info.depth_confidence
                            
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
        
        # 决策分析
        decision_output = None
        nav_output = None
        try:
            context = create_decision_context(
                frame_id=frame_id,
                detections=detection_data,
                frame_info={
                    'frame_width': frame.shape[1],
                    'frame_height': frame.shape[0],
                    'depth_map': depth_map,
                    'detections': detection_data
                }
            )
            
            decision_result = self.decision_engine.make_decision(context)
            decision_output = decision_result.to_dict()
            
            # 语音播报（异步）
            if (self.enable_tts and self.tts_engine and 
                decision_result.priority >= 3 and  # 只播报重要决策
                decision_result.speech):
                
                # 异步播报，避免阻塞处理
                threading.Thread(
                    target=self._async_speak,
                    args=(decision_result.speech,),
                    daemon=True
                ).start()
            
            # 导航信息融合与播报（仅在启用导航时）
            if self.enable_navigation and self.nav_context:
                try:
                    nav_instruction = self.nav_context.get_current_instruction()
                    nav_ctx = self.nav_context.get_context()
                    nav_output = {
                        'mode': nav_ctx.navigation_mode.value,
                        'state': nav_ctx.navigation_state.value,
                        'instruction': nav_instruction,
                        'distance_to_destination': nav_ctx.distance_to_destination,
                    }
                    # 若无高危风险，适度播报导航指令（节流）
                    hazard_cnt = 0
                    try:
                        hazard_cnt = (decision_output or {}).get('statistics', {}).get('high_risk_objects', 0)
                    except Exception:
                        hazard_cnt = 0
                    if (self.enable_tts and self.tts_engine and nav_instruction and 
                        hazard_cnt == 0 and
                        (time.time() - self._last_nav_tts_time) >= self._nav_tts_interval):
                        threading.Thread(
                            target=self._async_speak,
                            args=(nav_instruction,),
                            daemon=True
                        ).start()
                        self._last_nav_tts_time = time.time()
                except Exception as e:
                    logger.debug(f"导航信息获取失败: {e}")
                
        except Exception as e:
            logger.error(f"决策分析失败: {e}")
        
        processing_time = time.time() - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        
        return FrameAnalysisResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detection_data,
            depth_map=depth_map,
            decision_output=decision_output,
            navigation_output=nav_output,
            processing_time=processing_time,
            fps=current_fps
        )
    
    def _async_speak(self, text: str):
        """异步语音播报"""
        try:
            if self.tts_engine:
                self.tts_engine.speak(text)
        except Exception as e:
            logger.warning(f"语音播报失败: {e}")
    
    def _processing_worker(self):
        """处理线程工作函数"""
        frame_id = 0
        last_process_time = 0
        
        while self.is_running:
            try:
                # 控制处理帧率
                current_time = time.time()
                if current_time - last_process_time < self.frame_interval:
                    time.sleep(0.001)  # 短暂休眠
                    continue
                
                # 获取帧
                if self.frame_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame, timestamp = self.frame_queue.get(timeout=0.1)
                
                # 处理帧
                result = self._process_frame(frame, frame_id, timestamp)
                
                # 更新统计
                self.stats.processed_frames += 1
                self.stats.detection_count += len(result.detections)
                if result.decision_output:
                    self.stats.decision_count += 1
                    if result.decision_output.get('high_risk_objects', 0) > 0:
                        self.stats.hazard_count += 1
                
                self.processing_times.append(result.processing_time)
                
                # 将结果放入结果队列
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                frame_id += 1
                last_process_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理线程错误: {e}")
                continue
    
    def _update_stats(self):
        """更新统计信息"""
        if self.processing_times:
            self.stats.average_processing_time = sum(self.processing_times) / len(self.processing_times)
            self.stats.average_fps = 1.0 / self.stats.average_processing_time if self.stats.average_processing_time > 0 else 0
    
    def start_analysis(self, source: str = "0", display_results: bool = True) -> bool:
        """
        开始实时分析
        
        Args:
            source: 视频源（摄像头ID或视频文件路径）
            display_results: 是否显示结果窗口
            
        Returns:
            是否成功启动
        """
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        try:
            # 打开视频源
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
            else:
                cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                logger.error(f"无法打开视频源: {source}")
                return False
            
            # 设置摄像头参数（如果是摄像头）
            if source.isdigit():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"✅ 视频源已打开: {source}")
            
            # 启动处理线程
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.processing_thread.start()
            
            logger.info("🚀 开始实时分析...")
            
            # 主循环：读取帧并显示结果
            last_display_time = 0
            display_interval = 1.0 / 30.0  # 30fps显示
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("无法读取帧，可能是视频结束")
                    break
                
                self.stats.total_frames += 1
                
                # 将帧放入处理队列
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), time.time()))
                else:
                    self.stats.dropped_frames += 1
                
                # 显示结果
                if display_results:
                    current_time = time.time()
                    if current_time - last_display_time >= display_interval:
                        self._display_results(frame)
                        last_display_time = current_time
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("用户请求退出")
                    break
            
            # 清理资源
            cap.release()
            if display_results:
                cv2.destroyAllWindows()
            
            self.stop_analysis()
            return True
            
        except Exception as e:
            logger.error(f"分析过程中出错: {e}")
            self.stop_analysis()
            return False
    
    def _display_results(self, frame: np.ndarray):
        """显示分析结果"""
        try:
            # 获取最新的分析结果
            if not self.result_queue.empty():
                result = self.result_queue.get()
                
                # 在帧上绘制检测结果
                display_frame = frame.copy()
                
                # 绘制检测框
                for detection in result.detections:
                    bbox = detection['bbox']
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    # 绘制边界框
                    cv2.rectangle(display_frame, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"{class_name}: {confidence:.2f}"
                    if 'distance_m' in detection and detection['distance_m'] is not None:
                        label += f" ({detection['distance_m']:.1f}m)"
                    
                    cv2.putText(display_frame, label,
                              (int(bbox[0]), int(bbox[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 显示决策信息
                if result.decision_output:
                    decision_info = [
                        f"Action: {result.decision_output.get('action', 'N/A')}",
                        f"Priority: {result.decision_output.get('priority', 0)}",
                        f"Objects: {result.decision_output.get('statistics', {}).get('total_objects', 0)}",
                        f"Hazards: {result.decision_output.get('statistics', {}).get('high_risk_objects', 0)}"
                    ]
                    
                    for i, info in enumerate(decision_info):
                        cv2.putText(display_frame, info,
                                  (10, 30 + i * 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示导航信息
                if getattr(result, 'navigation_output', None):
                    nav = result.navigation_output
                    nav_lines = []
                    if nav.get('instruction'):
                        nav_lines.append(f"Nav: {nav['instruction']}")
                    nav_lines.append(f"NavMode: {nav.get('mode', '')}, State: {nav.get('state', '')}")
                    dist = nav.get('distance_to_destination')
                    if dist is not None:
                        dist_str = f"{dist/1000:.1f}km" if dist > 1000 else f"{dist:.0f}m"
                        nav_lines.append(f"ToDest: {dist_str}")
                    for i, info in enumerate(nav_lines):
                        cv2.putText(display_frame, info,
                                  (10, display_frame.shape[0] - 10 - i * 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                
                # 显示性能信息
                self._update_stats()
                perf_info = [
                    f"FPS: {self.stats.average_fps:.1f}",
                    f"Processing: {self.stats.average_processing_time*1000:.1f}ms",
                    f"Frames: {self.stats.processed_frames}/{self.stats.total_frames}",
                    f"Dropped: {self.stats.dropped_frames}"
                ]
                
                for i, info in enumerate(perf_info):
                    cv2.putText(display_frame, info,
                              (display_frame.shape[1] - 200, 30 + i * 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow('BlindStar Real-time Analysis', display_frame)
            else:
                # 没有结果时显示原始帧
                cv2.imshow('BlindStar Real-time Analysis', frame)
                
        except Exception as e:
            logger.warning(f"显示结果时出错: {e}")
    
    def stop_analysis(self):
        """停止分析"""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("✅ 实时分析已停止")
    
    def update_gps_location(self, latitude: float, longitude: float,
                            altitude: Optional[float] = None,
                            accuracy: Optional[float] = None) -> bool:
        """更新GPS位置(导航启用时可用)"""
        if not (self.enable_navigation and self.nav_context):
            return False
        location = GPSLocation(latitude=latitude, longitude=longitude,
                               altitude=altitude, accuracy=accuracy)
        return self.nav_context.update_location(location)
    
    def set_navigation_destination(self, latitude: float, longitude: float,
                                   altitude: Optional[float] = None,
                                   accuracy: Optional[float] = None) -> bool:
        """设置导航目的地(导航启用时可用)"""
        if not (self.enable_navigation and self.nav_context):
            return False
        destination = GPSLocation(latitude=latitude, longitude=longitude,
                                  altitude=altitude, accuracy=accuracy)
        return self.nav_context.set_destination(destination)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        self._update_stats()
        return {
            "total_frames": self.stats.total_frames,
            "processed_frames": self.stats.processed_frames,
            "dropped_frames": self.stats.dropped_frames,
            "drop_rate": self.stats.dropped_frames / max(1, self.stats.total_frames),
            "average_fps": self.stats.average_fps,
            "average_processing_time": self.stats.average_processing_time,
            "detection_count": self.stats.detection_count,
            "hazard_count": self.stats.hazard_count,
            "decision_count": self.stats.decision_count
        }
    
    def update_parameters(self, **kwargs):
        """更新分析参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"参数已更新: {key} = {value}")
            
            # 更新子组件参数
            if key == "confidence_threshold" and self.detector:
                self.detector.confidence_threshold = value
            elif key.startswith("depth_") and self.decision_engine:
                self.decision_engine.update_risk_parameters(**{key: value})
            elif key == "enable_navigation":
                # 启用或关闭导航上下文
                if bool(value) and not self.nav_context:
                    try:
                        mode_map = {
                            "assist": NavigationMode.ASSIST,
                            "guide": NavigationMode.GUIDE,
                            "full": NavigationMode.FULL,
                        }
                        mode_enum = mode_map.get((self.nav_mode or "assist").lower(), NavigationMode.ASSIST)
                        self.nav_context = NavigationContextManager(mode=mode_enum)
                        logger.info(f"导航上下文已启用: 模式 {mode_enum.value}")
                    except Exception as e:
                        logger.warning(f"导航上下文启用失败: {e}")
                        self.enable_navigation = False
                elif not bool(value):
                    self.nav_context = None
                    logger.info("导航上下文已关闭")
            elif key == "nav_mode":
                # 动态切换导航模式
                try:
                    mode_map = {
                        "assist": NavigationMode.ASSIST,
                        "guide": NavigationMode.GUIDE,
                        "full": NavigationMode.FULL,
                    }
                    mode_enum = mode_map.get((str(value) or "assist").lower(), NavigationMode.ASSIST)
                    if self.nav_context:
                        self.nav_context.context.navigation_mode = mode_enum
                        logger.info(f"导航模式已切换为: {mode_enum.value}")
                except Exception as e:
                    logger.debug(f"切换导航模式失败: {e}")
            elif key == "nav_tts_interval":
                try:
                    self._nav_tts_interval = float(value)
                    logger.info(f"导航TTS播报间隔已更新为: {self._nav_tts_interval}s")
                except Exception:
                    logger.warning("无效的 nav_tts_interval 值")


# 便捷函数
def analyze_video_realtime(source: str = "0", 
                          yolo_model: str = "small",
                          enable_depth: bool = True,
                          enable_tts: bool = True,
                          max_fps: float = 15.0,
                          enable_navigation: bool = False,
                          nav_mode: str = "assist") -> bool:
    """
    便捷函数：启动实时视频分析
    
    Args:
        source: 视频源
        yolo_model: YOLO模型大小
        enable_depth: 是否启用深度估计
        enable_tts: 是否启用语音播报
        max_fps: 最大处理帧率
        
    Returns:
        是否成功完成分析
    """
    analyzer = RealtimeVideoAnalyzer(
        yolo_model=yolo_model,
        enable_depth=enable_depth,
        enable_tts=enable_tts,
        max_fps=max_fps,
        enable_navigation=enable_navigation,
        nav_mode=nav_mode
    )
    
    try:
        return analyzer.start_analysis(source, display_results=True)
    finally:
        stats = analyzer.get_statistics()
        logger.info(f"分析完成统计: {stats}")
        return True
