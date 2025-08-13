#!/usr/bin/env python3
"""
BlindStar Core Module
Integrates MiDaS depth estimation with YOLO object detection
"""

import logging
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .detector import YOLOv8Detector, DetectionResult, draw_detections
from .distance import ZoeDepthDistanceMeasurement, DepthInfo
from .video_processor import VideoProcessor, VideoProcessingStats
from .camera import CameraHandler

logger = logging.getLogger(__name__)


class BlindStar:
    """
    BlindStar Core Class with MiDaS Distance Measurement
    
    Integrates YOLO object detection with MiDaS depth estimation
    for accurate distance measurement and visual assistance.
    """
    
    def __init__(self,
                 yolo_model: str = "small",
                 midas_model: str = "MiDaS_small",
                 confidence_threshold: float = 0.6,
                 enable_distance: bool = True,
                 device: str = "auto",
                 distance_unit: str = "meters",
                 max_depth: float = 10.0,
                 min_depth: float = 0.1):
        """
        Initialize BlindStar with MiDaS distance measurement
        
        Args:
            yolo_model: YOLO model variant ('nano', 'small', 'medium', 'large', 'xlarge')
            midas_model: MiDaS model variant ('MiDaS_small', 'MiDaS', 'DPT_Large', etc.)
            confidence_threshold: Detection confidence threshold (0.0-1.0)
            enable_distance: Enable MiDaS distance measurement
            device: Device to use ('auto', 'cpu', 'cuda')
            distance_unit: Distance unit ('meters', 'cm', 'inches')
            max_depth: Maximum depth in meters
            min_depth: Minimum depth in meters
        """
        self.yolo_model = yolo_model
        self.midas_model = midas_model
        self.confidence_threshold = confidence_threshold
        self.enable_distance = enable_distance
        self.device = device
        self.distance_unit = distance_unit
        self.max_depth = max_depth
        self.min_depth = min_depth
        
        # Components
        self.detector = None
        self.distance_calc = None
        self.video_processor = None
        self.camera_handler = None
        
        # State
        self.initialized = False
        self.processing_stats = {}
        
        logger.info(f"BlindStar initialized with YOLO: {yolo_model}, MiDaS: {midas_model}")
    
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing BlindStar components...")
            
            # Initialize YOLO detector
            logger.info(f"Loading {self.yolo_model} YOLO model...")
            self.detector = YOLOv8Detector(
                model_variant=self.yolo_model,
                confidence_threshold=self.confidence_threshold,
                device=self.device
            )
            
            # Initialize MiDaS distance measurement
            if self.enable_distance:
                logger.info(f"Initializing MiDaS distance measurement...")
                self.distance_calc = ZoeDepthDistanceMeasurement(
                    model_type=self.midas_model,
                    device=self.device,
                    max_depth=self.max_depth,
                    min_depth=self.min_depth,
                    distance_unit=self.distance_unit,
                    model_path=r"E:\BlindStar\models\ZoeD_M12_NK.pt"
                )
            
            # Initialize video processor
            self.video_processor = VideoProcessor(
                detector=self.detector,
                distance_calculator=self.distance_calc,
                frame_skip=1,
                output_fps=30
            )
            
            self.initialized = True
            logger.info("✅ BlindStar initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ BlindStar initialization failed: {e}")
            return False
    
    def detect_image(self, image_path: str, save_result: bool = False, output_path: Optional[str] = None) -> Dict:
        """
        Detect objects and measure distances in an image
        
        Args:
            image_path: Path to input image
            save_result: Whether to save annotated result
            output_path: Output path for annotated image
            
        Returns:
            Dictionary with detection results and processing info
        """
        if not self.initialized:
            raise RuntimeError("BlindStar not initialized. Call initialize() first.")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        start_time = time.time()
        
        # Detect objects
        assert self.detector is not None
        detections = self.detector.detect(image)
        
        # Calculate distances
        results = []
        if self.enable_distance and self.distance_calc and detections:
            depth_infos = self.distance_calc.calculate_distances_batch(image, detections)
            
            for detection, depth_info in zip(detections, depth_infos):
                result = {
                    'class_id': detection.class_id,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'center': detection.center,
                    'area': detection.area,
                    'distance_meters': depth_info.distance_meters,
                    'depth_confidence': depth_info.depth_confidence,
                    'min_depth': depth_info.min_depth,
                    'max_depth': depth_info.max_depth
                }
                results.append(result)
        else:
            for detection in detections:
                result = {
                    'class_id': detection.class_id,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'center': detection.center,
                    'area': detection.area
                }
                results.append(result)
        
        processing_time = time.time() - start_time
        
        # Save annotated result if requested
        if save_result:
            if output_path is None:
                output_path = f"detected_{Path(image_path).stem}.jpg"
            
            annotated_image = draw_detections(image, detections)
            cv2.imwrite(output_path, annotated_image)
        
        return {
            'detections': results,
            'processing_time': processing_time,
            'image_shape': image.shape,
            'total_objects': len(results)
        }
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame with detection and distance measurement
        
        Args:
            frame: Input frame
            
        Returns:
            Analysis results
        """
        if not self.initialized:
            raise RuntimeError("BlindStar not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # Detect objects
        assert self.detector is not None
        detections = self.detector.detect(frame)
        
        # Calculate distances
        results = []
        if self.enable_distance and self.distance_calc and detections:
            depth_infos = self.distance_calc.calculate_distances_batch(frame, detections)
            
            for detection, depth_info in zip(detections, depth_infos):
                result = {
                    'class_id': detection.class_id,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'center': detection.center,
                    'area': detection.area,
                    'distance_meters': depth_info.distance_meters,
                    'depth_confidence': depth_info.depth_confidence
                }
                results.append(result)
        else:
            for detection in detections:
                result = {
                    'class_id': detection.class_id,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'center': detection.center,
                    'area': detection.area
                }
                results.append(result)
        
        processing_time = time.time() - start_time
        
        return {
            'detections': results,
            'processing_time': processing_time,
            'frame_shape': frame.shape,
            'total_objects': len(results)
        }
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            'initialized': self.initialized,
            'yolo_model': self.yolo_model,
            'midas_model': self.midas_model,
            'confidence_threshold': self.confidence_threshold,
            'components': {
                'detector': self.detector is not None,
                'distance_calc': self.distance_calc is not None,
                'video_processor': self.video_processor is not None,
                'camera_handler': self.camera_handler is not None
            },
            'device': self.device,
            'distance_unit': self.distance_unit,
            'depth_range': f"{self.min_depth}-{self.max_depth}",
            'processing_stats': self.processing_stats
        }
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up BlindStar resources...")
        
        if self.camera_handler:
            self.camera_handler.stop()
        
        if self.distance_calc:
            self.distance_calc.cleanup()
        
        self.initialized = False
        logger.info("BlindStar cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

    # ---------------------------------------------------------------------
    # Convenience wrappers for legacy callers (e.g. CLI scripts)
    # ---------------------------------------------------------------------

    def start_camera(self, camera_id: int = 0, **kwargs) -> bool:  # noqa: D401
        """Start a camera stream using internal :class:`CameraHandler`.

        This method is kept for backward-compatibility with older CLI examples
        that expected :pyclass:`BlindStar` to expose simple camera helpers.
        Internally we lazily create a ``CameraHandler`` instance and delegate
        the start call. Additional keyword arguments are forwarded to
        :class:`CameraHandler` for future extensibility, but are currently
        ignored.
        """

        if self.camera_handler is None:
            from .camera import CameraHandler  # Local import to avoid cycles
            self.camera_handler = CameraHandler(source=camera_id)

        return self.camera_handler.start()

    def get_camera_frame(self):
        """Retrieve a single frame from the active camera handler.

        Returns ``None`` if the camera has not been started or yielded no
        frame.
        """

        if self.camera_handler is None:
            return None

        return self.camera_handler.read_frame()

    def stop_camera(self):  # noqa: D401
        """Stop the current camera session (if any)."""

        if self.camera_handler is not None:
            self.camera_handler.stop()
            self.camera_handler = None

    # ------------------------------------------------------------------
    # Video processing helper
    # ------------------------------------------------------------------

    def process_video(self,
                      video_path: str,
                      output_path: str | None = None,
                      max_duration: float | None = None):
        """Run offline video processing via :class:`VideoProcessor`.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        output_path:
            Where to save the annotated result video. If *None*, a file named
            ``<stem>_detected.mp4`` will be written next to the input.
        max_duration:
            Optional cap on processing duration in seconds (handy for quick
            tests).
        """

        if not self.initialized:
            raise RuntimeError("BlindStar not initialized. Call initialize() first.")

        if output_path is None:
            from pathlib import Path as _Path
            in_path = _Path(video_path)
            output_path = str(in_path.with_name(f"{in_path.stem}_detected.mp4"))

        assert self.video_processor is not None, "VideoProcessor not initialized"

        return self.video_processor.process_video_file(
            input_path=video_path,
            output_path=output_path,
            max_duration=max_duration,
        )
