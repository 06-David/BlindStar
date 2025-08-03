#!/usr/bin/env python3
"""
Video Processing Module for YOLOv8 Object Detection System
Handles video file processing, frame extraction, batch detection, and result video generation
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple
import logging
from dataclasses import dataclass

from .detector import YOLOv8Detector, DetectionResult, draw_detections
from .distance import DistanceMeasurement
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
                 output_fps: int = 30):
        """
        Initialize video processor

        Args:
            detector: YOLOv8 detector instance
            distance_calculator: Optional distance measurement instance
            speed_calculator: Optional speed measurement instance
            frame_skip: Skip frames (1 = process all, 2 = every other frame, etc.)
            output_fps: Output video FPS
        """
        self.detector = detector
        self.distance_calculator = distance_calculator
        self.speed_calculator = speed_calculator
        self.frame_skip = max(1, frame_skip)
        self.output_fps = output_fps

        # Processing state
        self.is_processing = False
        self.processing_stats = VideoProcessingStats()
        self.progress_callback: Optional[Callable[[float, Dict], None]] = None
        self.cancel_requested = False

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
                        # 计算每个检测框的距离并写回 detection 对象，便于后续可视化
                        depth_infos = self.distance_calculator.calculate_distances_batch(
                            frame,
                            detections
                        )
                        for det, info in zip(detections, depth_infos):
                            det.distance = info.distance_meters  # type: ignore[attr-defined]

                    # Calculate speeds
                    if self.speed_calculator and detections:
                        detections = self.speed_calculator.update_tracking(frame, detections)

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
                    
                    # Update progress
                    progress = (frame_count + 1) / total_frames
                    if self.progress_callback:
                        progress_data = {
                            'frame': frame_count + 1,
                            'total_frames': total_frames,
                            'detections': len(detections),
                            'total_detections': self.processing_stats.total_detections,
                            'processed_frames': processed_count
                        }
                        self.progress_callback(progress, progress_data)
                
                else:
                    # Write original frame without processing
                    out.write(frame)
                
                frame_count += 1
                
                # Log progress periodically
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
            
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
        from .frame_analyzer import FrameAnalyzer

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

                    # Debug logging every 100 frames
                    # if frame_count % 100 == 0:
                    #     logger.info(f"Frame {frame_count}: Analyzing {len(detections)} detections")


                    frame_analyzer.analyze_frame(
                        frame_number=frame_count,
                        detections=detections,
                        frame_width=width,
                        frame_height=height,
                        processing_time_ms=frame_processing_time,
                        timestamp=current_timestamp
                    )

                    # if frame_count % 100 == 0:
                    #     logger.info(f"Frame {frame_count}: Analysis completed")

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

                    # Update progress
                    progress = (frame_count + 1) / total_frames
                    if self.progress_callback:
                        progress_data = {
                            'frame': frame_count + 1,
                            'total_frames': total_frames,
                            'detections': len(detections),
                            'total_detections': self.processing_stats.total_detections,
                            'processed_frames': processed_count
                        }
                        self.progress_callback(progress, progress_data)

                else:
                    # Write original frame without processing
                    out.write(frame)

                frame_count += 1

                # Log progress periodically
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")

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
