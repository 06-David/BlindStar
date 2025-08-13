#!/usr/bin/env python3
"""
Video Processing Module for YOLOv8 Object Detection System
Handles video file processing, frame extraction, batch detection, and result video generation
"""

import cv2
import numpy as np
import time
import threading
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm  # 用于进度条显示

# 添加项目根目录到路径，支持直接运行脚本
sys.path.append(str(Path(__file__).parent.parent))

# 现在可以安全地导入相对模块
try:
    from .detector import YOLOv8Detector, DetectionResult, draw_detections
    from .distance import DistanceMeasurement
    from config import VideoConfig
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from core.detector import YOLOv8Detector, DetectionResult, draw_detections
    from core.distance import DistanceMeasurement
    from config import VideoConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoProcessingStats:
    """Statistics for video processing"""
    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    processing_time: float = 0.0
    average_fps: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class VideoProcessor:
    """
    Video processing class for object detection
    Handles video file reading, frame processing, and result video generation
    """
    
    def __init__(self,
                 detector: YOLOv8Detector,
                 distance_calculator: Optional[DistanceMeasurement] = None,
                 speed_calculator: Optional[Any] = None,
                 frame_skip: int = 1,
                 output_fps: int = 30,
                 adaptive_frame_skip: bool = True,
                 batch_size: int = 1):
        """
        Initialize video processor

        Args:
            detector: YOLOv8 detector instance
            distance_calculator: Optional distance measurement instance
            speed_calculator: Optional speed measurement instance
            frame_skip: Skip frames (1 = process all, 2 = every other frame, etc.)
            output_fps: Output video FPS
            adaptive_frame_skip: Enable adaptive frame skipping based on processing speed
            batch_size: Batch size for processing multiple frames at once
        """
        self.detector = detector
        self.distance_calculator = distance_calculator
        self.speed_calculator = speed_calculator
        self.frame_skip = max(1, frame_skip)
        self.output_fps = output_fps
        self.adaptive_frame_skip = adaptive_frame_skip
        self.batch_size = max(1, batch_size)

        # Processing state
        self.is_processing = False
        self.processing_stats = VideoProcessingStats()
        self.progress_callback: Optional[Callable[[float, Dict], None]] = None
        self.cancel_requested = False

        # Performance tracking
        self.frame_times = []
        self.current_frame_skip = self.frame_skip

        # Thread safety
        self._lock = threading.Lock()
    
    def set_progress_callback(self, callback: Callable[[float, Dict], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def cancel_processing(self):
        """Cancel current processing operation"""
        with self._lock:
            self.cancel_requested = True
    
    def process_video_file(self,
                           input_path: str | Path,
                           output_path: str | Path,
                           max_duration: Optional[float] = None) -> VideoProcessingStats:
        """
        Process video file with object detection
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            max_duration: Maximum duration to process (seconds)
            
        Returns:
            VideoProcessingStats object with processing statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reset processing state
        with self._lock:
            self.is_processing = True
            self.cancel_requested = False
            self.processing_stats = VideoProcessingStats()
            self.processing_stats.start_time = time.time()
        
        cap = None
        out = None
        
        try:
            # Open input video
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate frames to process
            if max_duration:
                max_frames = int(max_duration * fps)
                total_frames = min(total_frames, max_frames)
            
            self.processing_stats.total_frames = total_frames
            
            logger.info(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Initialize video writer
            fourcc = getattr(cv2, 'VideoWriter_fourcc')(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.output_fps, (width, height))
            
            frame_count = 0
            processed_count = 0
            frames_buffer = []

            # 用tqdm显示主循环进度条，unit为帧，desc为进度说明
            with tqdm(total=total_frames, desc="视频处理进度", unit="帧", ncols=80) as pbar:
                while True:
                    # 检查是否取消
                    with self._lock:
                        if self.cancel_requested:
                            logger.info("Video processing cancelled by user")
                            break
                    
                    ret, frame = cap.read()
                    if not ret or frame_count >= total_frames:
                        break
                    
                    # 自适应跳帧
                    if self.adaptive_frame_skip and len(self.frame_times) > 10:
                        avg_frame_time = np.mean(self.frame_times[-10:])
                        target_fps = 5.0  # 目标处理5FPS
                        if avg_frame_time > 1.0 / target_fps:
                            self.current_frame_skip = min(self.current_frame_skip + 1, 10)
                        elif avg_frame_time < 1.0 / (target_fps * 2):
                            self.current_frame_skip = max(self.current_frame_skip - 1, 1)
                    
                    # 需要处理的帧
                    if frame_count % self.current_frame_skip == 0:
                        frames_buffer.append((frame_count, frame))
                        # 缓冲区满或结束时处理批次
                        if len(frames_buffer) >= self.batch_size or not ret:
                            batch_start_time = time.time()
                            # 处理批次
                            for batch_idx, (batch_frame_count, batch_frame) in enumerate(frames_buffer):
                                detections = self.detector.detect_frame(batch_frame)
                                # 计算距离
                                if self.distance_calculator and detections:
                                    depth_infos = self.distance_calculator.calculate_distances_batch(
                                        batch_frame,
                                        detections
                                    )
                                    for det, info in zip(detections, depth_infos):
                                        det.distance = info.distance_meters  # type: ignore[attr-defined]
                                # 计算速度
                                if self.speed_calculator and detections:
                                    detections = self.speed_calculator.update_tracking(batch_frame, detections)
                                # 更新统计
                                self.processing_stats.total_detections += len(detections)
                                processed_count += 1
                                # 绘制检测结果并写入输出
                                result_frame = draw_detections(
                                    batch_frame,
                                    detections,
                                    show_distance=self.distance_calculator is not None,
                                    show_speed=self.speed_calculator is not None
                                )
                                out.write(result_frame)
                                # tqdm进度+1
                                pbar.update(1)
                            # 更新性能统计
                            batch_time = time.time() - batch_start_time
                            self.frame_times.append(batch_time / len(frames_buffer))
                            # 清空缓冲区
                            frames_buffer.clear()
                    else:
                        # 跳帧也写入原始帧并进度+1
                        out.write(frame)
                        pbar.update(1)
                    # 进度回调
                    progress = (frame_count + 1) / total_frames
                    if self.progress_callback:
                        progress_data = {
                            'frame': frame_count + 1,
                            'total_frames': total_frames,
                            'detections': len(detections) if 'detections' in locals() else 0,
                            'total_detections': self.processing_stats.total_detections,
                            'processed_frames': processed_count,
                            'current_fps': 1.0 / np.mean(self.frame_times[-10:]) if self.frame_times else 0,
                            'frame_skip': self.current_frame_skip
                        }
                        self.progress_callback(progress, progress_data)
                    frame_count += 1
                    # tqdm已替代原有进度日志输出
            
            # Update final statistics
            self.processing_stats.processed_frames = processed_count
            self.processing_stats.end_time = time.time()
            self.processing_stats.processing_time = (
                self.processing_stats.end_time - self.processing_stats.start_time
            )
            
            if self.processing_stats.processing_time > 0:
                self.processing_stats.average_fps = processed_count / self.processing_stats.processing_time
            
            logger.info(f"Video processing completed: {processed_count} frames processed, "
                       f"{self.processing_stats.total_detections} total detections")

            return self.processing_stats
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
            
        finally:
            # Cleanup
            if cap:
                cap.release()
            if out:
                out.release()
            
            with self._lock:
                self.is_processing = False

    def process_video_file_with_frame_analysis(self,
                          input_path: str | Path,
                          output_path: str | Path,
                          max_duration: Optional[float] = None,
                          frame_analysis_config: Optional[Dict[str, Any]] = None) -> VideoProcessingStats:
        """
        Process video file with detailed frame analysis (for local file analysis only)

        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            max_duration: Maximum duration to process (seconds)
            frame_analysis_config: Configuration for frame analysis

        Returns:
            VideoProcessingStats object with processing statistics
        """
        # Initialize frame analyzer for this session
        try:
            from .frame_analyzer import FrameAnalyzer
        except ImportError:
            from core.frame_analyzer import FrameAnalyzer

        # Default frame analysis config
        default_config = {
            "log_dir": "logs/local_video_analysis",
            "enable_json_log": True,
            "enable_csv_log": True,
            "log_level": "INFO"
        }

        if frame_analysis_config:
            default_config.update(frame_analysis_config)

        frame_analyzer = FrameAnalyzer(**default_config)
        logger.info("Frame analysis enabled for local video processing")

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reset processing state
        with self._lock:
            self.is_processing = True
            self.cancel_requested = False
            self.processing_stats = VideoProcessingStats()
            self.processing_stats.start_time = time.time()

        cap = None
        out = None

        try:
            # Open input video
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate frames to process
            if max_duration:
                max_frames = int(max_duration * fps)
                total_frames = min(total_frames, max_frames)

            self.processing_stats.total_frames = total_frames

            logger.info(f"Processing video with frame analysis: {width}x{height}, {fps} FPS, {total_frames} frames")

            # Initialize video writer
            fourcc = getattr(cv2, 'VideoWriter_fourcc')(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.output_fps, (width, height))

            frame_count = 0
            processed_count = 0

            # 使用tqdm显示帧分析进度
            with tqdm(total=total_frames, desc="帧分析进度", unit="帧", ncols=80) as pbar:
                while True:
                    # Check for cancellation
                    with self._lock:
                        if self.cancel_requested:
                            logger.info("Video processing cancelled by user")
                            break

                    ret, frame = cap.read()
                    if not ret or frame_count >= total_frames:
                        break

                    # Skip frames if needed
                    if frame_count % self.frame_skip == 0:
                        frame_start_time = time.time()

                        # Process frame
                        detections = self.detector.detect_frame(frame)

                        # Calculate distances
                        if self.distance_calculator and detections:
                            depth_infos = self.distance_calculator.calculate_distances_batch(
                                frame,
                                detections
                            )
                            for det, info in zip(detections, depth_infos):
                                det.distance = info.distance_meters  # type: ignore[attr-defined]

                        # Calculate speeds
                        if self.speed_calculator and detections:
                            detections = self.speed_calculator.update_tracking(frame, detections)

                        # Frame analysis (only for local file analysis)
                        frame_processing_time = (time.time() - frame_start_time) * 1000  # Convert to ms
                        current_timestamp = time.time()

                        frame_analyzer.analyze_frame(
                            frame_number=frame_count,
                            detections=detections,
                            frame_width=width,
                            frame_height=height,
                            processing_time_ms=frame_processing_time,
                            timestamp=current_timestamp
                        )

                        # Update statistics
                        self.processing_stats.total_detections += len(detections)
                        processed_count += 1

                        # Draw detections on frame
                        result_frame = draw_detections(
                            frame,
                            detections,
                            show_distance=self.distance_calculator is not None,
                            show_speed=self.speed_calculator is not None
                        )

                        # Write frame to output video
                        out.write(result_frame)

                        # tqdm 进度 +1
                        pbar.update(1)

                    else:
                        # Write original frame without processing
                        out.write(frame)
                        # tqdm 进度 +1（尽管未分析，但也推进总帧进度）
                        pbar.update(1)

                    # Update progress callback
                    progress = (frame_count + 1) / total_frames
                    if self.progress_callback:
                        progress_data = {
                            'frame': frame_count + 1,
                            'total_frames': total_frames,
                            'detections': len(detections) if 'detections' in locals() else 0,
                            'total_detections': self.processing_stats.total_detections,
                            'processed_frames': processed_count
                        }
                        self.progress_callback(progress, progress_data)

                    frame_count += 1
            
            # Update final statistics
            self.processing_stats.processed_frames = processed_count
            self.processing_stats.end_time = time.time()
            self.processing_stats.processing_time = (
                self.processing_stats.end_time - self.processing_stats.start_time
            )

            if self.processing_stats.processing_time > 0:
                self.processing_stats.average_fps = processed_count / self.processing_stats.processing_time

            # Finalize frame analysis
            frame_analyzer.finalize_session()
            logger.info("Frame analysis completed and saved")

            logger.info(f"Video processing with frame analysis completed: {processed_count} frames processed, "
                       f"{self.processing_stats.total_detections} total detections")

            return self.processing_stats

        except Exception as e:
            logger.error(f"Error processing video with frame analysis: {e}")
            raise

        finally:
            # Cleanup
            if cap:
                cap.release()
            if out:
                out.release()

            with self._lock:
                self.is_processing = False

    def extract_frames(self,
                      video_path: str | Path, 
                      output_dir: str | Path,
                      interval: int = 30,
                      max_frames: Optional[int] = None) -> List[str]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            interval: Frame interval (extract every N frames)
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frame file paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        extracted_frames = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    # Save frame
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = output_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            logger.info(f"Extracted {extracted_count} frames from video")
            return extracted_frames
            
        finally:
            cap.release()
    
    def get_video_info(self, video_path: str | Path) -> Dict[str, Any]:
        """
        Get video file information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            frame_count_val = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            info: Dict[str, Any] = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': fps_val,
                'frame_count': frame_count_val,
                'duration': 0.0,
                'codec': 'Unknown'
            }

            # Calculate duration
            if fps_val > 0:
                info['duration'] = frame_count_val / fps_val

            # Get codec information (if available)
            fourcc_val = cap.get(cv2.CAP_PROP_FOURCC)
            if fourcc_val:
                codec_bytes = int(fourcc_val).to_bytes(4, byteorder='little')
                try:
                    info['codec'] = codec_bytes.decode('ascii').strip('\x00')
                except UnicodeDecodeError:
                    info['codec'] = 'Unknown'

            return info

        finally:
            cap.release()
    
    def is_video_file(self, file_path: str | Path) -> bool:
        """
        Check if file is a supported video format
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is a supported video format
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower().lstrip('.')
        return extension in VideoConfig.SUPPORTED_FORMATS


def process_video_async(processor: VideoProcessor,
                       input_path: str | Path,
                       output_path: str | Path,
                       callback: Optional[Callable[[Optional[VideoProcessingStats]], None]] = None,
                       max_duration: Optional[float] = None) -> threading.Thread:
    """
    Process video asynchronously in a separate thread
    
    Args:
        processor: VideoProcessor instance
        input_path: Path to input video
        output_path: Path to output video
        callback: Optional callback when processing completes
        max_duration: Maximum duration to process
        
    Returns:
        Thread object
    """
    def _process():
        try:
            stats: Optional[VideoProcessingStats] = processor.process_video_file(input_path, output_path, max_duration)
            if callback:
                callback(stats)
        except Exception as e:
            logger.error(f"Async video processing failed: {e}")
            if callback:
                callback(None)
    
    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    return thread


def main():
    """主函数 - 处理命令行参数并执行视频处理"""
    parser = argparse.ArgumentParser(
        description="BlindStar 视频处理器 - 使用YOLOv8进行视频对象检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python video_processor.py --input data/1.mp4 --output output/processed.mp4
  python video_processor.py --input data/1.mp4 --model large --conf 0.2
  python video_processor.py --input data/1.mp4 --data obstacle.yaml --weights runs/train/test_train1/weights/best.pt --conf 0.2
  python video_processor.py --input data/1.mp4 --max-duration 60 --device cuda
        """
    )

    # 输入/输出参数
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入视频文件路径'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出视频文件路径 (默认: logs/input_processed_timestamp.mp4)'
    )

    # 模型配置
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='small',
        choices=['nano', 'small', 'medium', 'large', 'xlarge'],
        help='YOLOv8模型变体 (默认: small)'
    )

    parser.add_argument(
        '--weights',
        type=str,
        help='自定义模型权重文件路径'
    )

    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.6,
        help='检测置信度阈值 (默认: 0.6)'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='数据集配置文件路径 (YAML格式)'
    )

    # 处理选项
    parser.add_argument(
        '--max-duration',
        type=float,
        help='最大处理时长（秒）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='处理设备 (默认: auto)'
    )

    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='跳帧数 (默认: 1, 即处理所有帧)'
    )

    parser.add_argument(
        '--output-fps',
        type=int,
        default=30,
        help='输出视频FPS (默认: 30)'
    )

    # 日志选项
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志'
    )

    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 验证输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return False

    # 设置输出路径 - 创建类似 /1_20250804_083829 的文件夹结构
    if args.output:
        output_path = Path(args.output)
        output_dir = output_path.parent
    else:
        # 生成默认输出路径
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        source_name = input_path.stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = logs_dir / f"{source_name}_{timestamp}"
        output_path = output_dir / "yolo.mp4"  # 主输出视频文件

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始处理视频: {input_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"模型: {args.model}, 置信度: {args.conf}, 设备: {args.device}")

    try:
        # 初始化检测器
        # 如果指定了自定义权重，使用权重路径作为模型变体
        model_variant = args.weights if args.weights else args.model
        
        detector = YOLOv8Detector(
            model_variant=model_variant,
            confidence_threshold=args.conf,
            device=args.device,
            data_yaml=args.data
        )

        # 初始化距离计算器
        distance_calculator = DistanceMeasurement()

        # 初始化视频处理器
        processor = VideoProcessor(
            detector=detector,
            distance_calculator=distance_calculator,
            frame_skip=args.frame_skip,
            output_fps=args.output_fps
        )

        # 设置帧分析配置
        frame_analysis_config = {
            "log_dir": str(output_dir),
            "enable_json_log": True,
            "enable_csv_log": True,
            "log_level": "INFO"
        }

        # 处理视频并生成多文件输出
        stats = processor.process_video_file_with_frame_analysis(
            input_path=str(input_path),
            output_path=str(output_path),
            max_duration=args.max_duration,
            frame_analysis_config=frame_analysis_config
        )

        # 生成深度统计文件和深度视频
        depth_stats_path = output_dir / "depth_stats.csv"
        depth_video_path = output_dir / "depth.mp4"
        
        try:
            # 导入深度可视化模块
            from core.depth_visualizer import DepthVisualizer
            import csv
            
            logger.info("开始生成深度统计和深度视频...")
            
            # 初始化深度可视化器
            depth_vis = DepthVisualizer(
                model_type="ZoeD_M12_NK",
                device=args.device,
                max_depth=10.0,
                min_depth=0.1
            )
            
            # 打开输入视频
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError("无法打开输入视频")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 限制处理帧数
            if args.max_duration:
                max_frames = int(args.max_duration * fps)
                total_frames = min(total_frames, max_frames)
            
            # 初始化视频写入器
            fourcc = getattr(cv2, 'VideoWriter_fourcc')(*'mp4v')
            writer = cv2.VideoWriter(str(depth_video_path), fourcc, fps, (width, height))
            
            stats_rows = []
            
            # 处理每一帧
            with tqdm(total=total_frames, desc="生成深度视频", unit="帧") as pbar:
                for frame_idx in range(1, total_frames + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 计算深度图
                    depth_map = depth_vis.get_depth_map(frame)
                    
                    # 生成伪彩色深度图
                    depth_colored = depth_vis.get_colormap(frame)
                    
                    # 写入视频
                    writer.write(depth_colored)
                    
                    # 记录统计信息
                    if depth_map.size > 0:
                        depth_stats = {
                            "frame": frame_idx,
                            "timestamp": frame_idx / fps,
                            "min": float(np.min(depth_map)),
                            "max": float(np.max(depth_map)),
                            "mean": float(np.mean(depth_map)),
                            "median": float(np.median(depth_map))
                        }
                    else:
                        depth_stats = {
                            "frame": frame_idx,
                            "timestamp": frame_idx / fps,
                            "min": 0.0,
                            "max": 0.0,
                            "mean": 0.0,
                            "median": 0.0
                        }
                    stats_rows.append(depth_stats)
                    pbar.update(1)
            
            # 保存统计日志
            with open(depth_stats_path, "w", newline="", encoding="utf-8") as f:
                if stats_rows:
                    writer_csv = csv.DictWriter(f, fieldnames=stats_rows[0].keys())
                    writer_csv.writeheader()
                    writer_csv.writerows(stats_rows)
            
            # 清理资源
            cap.release()
            writer.release()
            depth_vis.cleanup()
            
            logger.info(f"深度统计已保存: {depth_stats_path}")
            logger.info(f"深度视频已保存: {depth_video_path}")
            
        except Exception as e:
            logger.error(f"生成深度统计失败: {e}")
            # 如果失败，至少创建一个空的统计文件
            with open(depth_stats_path, "w", newline="", encoding="utf-8") as f:
                f.write("frame,timestamp,min,max,mean,median\n")
            logger.info(f"创建了空的深度统计文件: {depth_stats_path}")

        # 打印结果
        logger.info("视频处理完成!")
        logger.info(f"  总帧数: {stats.total_frames}")
        logger.info(f"  处理帧数: {stats.processed_frames}")
        logger.info(f"  总检测数: {stats.total_detections}")
        logger.info(f"  平均FPS: {stats.average_fps:.2f}")
        logger.info(f"  处理时间: {stats.processing_time:.2f}秒")
        logger.info(f"  输出文件: {output_path}")

        return True

    except Exception as e:
        logger.error(f"视频处理失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
