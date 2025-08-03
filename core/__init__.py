"""
BlindStar Core Modules

Modular computer vision system for intelligent visual assistance.
Provides object detection, distance measurement, speed analysis, and frame analysis.
"""

from .blindstar import BlindStar
from .detector import YOLOv8Detector, DetectionResult, draw_detections
from .distance import MiDaSDistanceMeasurement, DepthInfo
from .camera import CameraHandler
from .video_processor import VideoProcessor, VideoProcessingStats, process_video_async
from .advanced_frame_analyzer import AdvancedFrameAnalyzer, RobustVideoAnalyzer
from .batch_processor import (
    BatchVideoProcessor,
    BatchImageProcessor,
    BatchProcessingConfig,
    BatchProcessingResult
)

# Optional imports with fallback
try:
    from .speed_measurement import OpticalFlowSpeedMeasurement
except ImportError:
    OpticalFlowSpeedMeasurement = None

try:
    from .frame_analyzer import FrameAnalyzer
except ImportError:
    FrameAnalyzer = None

try:
    from .audio import AudioFeedback
except ImportError:
    AudioFeedback = None

__all__ = [
    'BlindStar',
    'YOLOv8Detector',
    'DetectionResult',
    'draw_detections',
    'MiDaSDistanceMeasurement',
    'DepthInfo',
    'CameraHandler',
    'VideoProcessor',
    'VideoProcessingStats',
    'process_video_async',
    'OpticalFlowSpeedMeasurement',
    'FrameAnalyzer',
    'AdvancedFrameAnalyzer',
    'RobustVideoAnalyzer',
    'BatchVideoProcessor',
    'BatchImageProcessor',
    'BatchProcessingConfig',
    'BatchProcessingResult',
    'AudioFeedback'
]
