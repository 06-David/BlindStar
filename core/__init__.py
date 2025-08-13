"""
BlindStar Core Modules

Modular computer vision system for intelligent visual assistance.
Provides object detection, distance measurement, speed analysis, and frame analysis.
"""

from .blindstar import BlindStar
from .detector import YOLOv8Detector, DetectionResult, draw_detections
from .distance import ZoeDepthDistanceMeasurement, DepthInfo
from .camera import CameraHandler
from .video_processor import VideoProcessor, VideoProcessingStats, process_video_async
from .frame_analyzer import FrameAnalyzer, AdvancedFrameAnalyzer, VideoAnalysisStats, RobustVideoAnalyzer
from .batch_processor import (
    BatchVideoProcessor,
    BatchImageProcessor,
    BatchProcessingConfig,
    BatchProcessingResult
)

# 决策系统模块
from .decision_engine import DecisionEngine, DecisionContext, DecisionOutput, create_decision_context, make_quick_decision
from .risk_assessor import RiskAssessor, ObjectRisk, RiskLevel
from .action_planner import ActionPlanner, ActionPlan, ActionType
from .depth_hazard_detector import DepthHazardDetector, DepthHazard, DepthHazardType
from .realtime_video_analyzer import RealtimeVideoAnalyzer, analyze_video_realtime
from .navigation_context import NavigationContextManager, NavigationMode, NavigationState, GPSLocation
from .navigation_fusion import NavigationDecisionFusion, NavigationDecision

# Optional imports with fallback
try:
    from .speed_measurement import OpticalFlowSpeedMeasurement
except ImportError:
    OpticalFlowSpeedMeasurement = None

# FrameAnalyzer已经在上面导入了

try:
    from .audio import AudioFeedback
except ImportError:
    AudioFeedback = None

__all__ = [
    'BlindStar',
    'YOLOv8Detector',
    'DetectionResult',
    'draw_detections',
    'ZoeDepthDistanceMeasurement',
    'DepthInfo',
    'CameraHandler',
    'VideoProcessor',
    'VideoProcessingStats',
    'process_video_async',
    'OpticalFlowSpeedMeasurement',
    'FrameAnalyzer',
    'AdvancedFrameAnalyzer',
    'VideoAnalysisStats',
    'RobustVideoAnalyzer',
    'BatchVideoProcessor',
    'BatchImageProcessor',
    'BatchProcessingConfig',
    'BatchProcessingResult',
    'AudioFeedback',
    # 决策系统模块
    'DecisionEngine',
    'DecisionContext', 
    'DecisionOutput',
    'create_decision_context',
    'make_quick_decision',
    'RiskAssessor',
    'ObjectRisk',
    'RiskLevel',
    'ActionPlanner',
    'ActionPlan',
    'ActionType',
    'DepthHazardDetector',
    'DepthHazard',
    'DepthHazardType',
    'RealtimeVideoAnalyzer',
    'analyze_video_realtime',
    # 导航模块
    'NavigationContextManager',
    'NavigationMode',
    'NavigationState',
    'GPSLocation',
    'NavigationDecisionFusion',
    'NavigationDecision'
]
