"""
Configuration file for YOLOv8 Object Detection and Distance Measurement System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
class ModelConfig:
    # YOLOv8 model variants
    YOLO_VARIANTS = {
        'nano': 'yolov8n.pt',      # Fastest, smallest
        'small': 'yolov8s.pt',     # Good balance
        'medium': 'yolov8m.pt',    # Better accuracy
        'large': 'yolov8l.pt',     # High accuracy
        'xlarge': 'yolov8x.pt'     # Best accuracy
    }

    # MiDaS depth estimation model variants
    MIDAS_VARIANTS = {
        'small': 'MiDaS_small',           # Fastest, good for real-time
        'default': 'MiDaS',               # Standard model
        'large': 'DPT_Large',             # High accuracy
        'hybrid': 'DPT_Hybrid',           # Balanced accuracy/speed
        'swin': 'DPT_SwinV2_L_384'        # Best accuracy, slowest
    }

    # Depth estimation method options
    DEPTH_METHODS = {
        'geometric': 'geometric',         # Geometric calculation (fast)
        'midas': 'midas'                  # MiDaS depth estimation (accurate)
    }

    # Default models for different use cases
    DEFAULT_YOLO_MODEL = 'small'          # Good balance for most applications
    DEFAULT_MIDAS_MODEL = 'small'         # Fast depth estimation
    DEFAULT_DEPTH_METHOD = 'geometric'    # Default to geometric for compatibility
    MOBILE_YOLO_MODEL = 'nano'            # Optimized for mobile deployment
    MOBILE_MIDAS_MODEL = 'small'          # Fast depth for mobile

    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.6
    NMS_THRESHOLD = 0.45
    MAX_DETECTIONS = 100

    # Input image size (must be multiple of 32)
    INPUT_SIZE = 640

    # Backward compatibility
    MODEL_VARIANTS = YOLO_VARIANTS  # For backward compatibility
    DEFAULT_MODEL = DEFAULT_YOLO_MODEL

# Camera configuration
class CameraConfig:
    # Default camera settings
    DEFAULT_CAMERA_ID = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30
    
    # Video processing
    BUFFER_SIZE = 1
    THREADING_ENABLED = True

# MiDaS Distance measurement configuration
class MiDaSConfig:
    # MiDaS model variants
    MODEL_VARIANTS = {
        'small': 'MiDaS_small',           # Fastest, good for real-time
        'default': 'MiDaS',               # Standard model
        'large': 'DPT_Large',             # High accuracy
        'hybrid': 'DPT_Hybrid',           # Balanced accuracy/speed
        'swin': 'DPT_SwinV2_L_384'        # Best accuracy, slowest
    }

    # Default settings
    DEFAULT_MODEL = 'small'               # Good balance for most applications
    DEFAULT_DEVICE = 'auto'               # Auto-detect best device

    # Depth range settings
    MAX_DEPTH = 10.0                      # Maximum depth in meters
    MIN_DEPTH = 0.1                       # Minimum depth in meters

    # Units
    DISTANCE_UNIT = "meters"              # or "cm", "inches"

    # Performance settings
    BATCH_SIZE = 1                        # Batch size for processing
    USE_HALF_PRECISION = False            # Use FP16 for faster inference (if supported)

# Video processing configuration
class VideoConfig:
    # Video processing settings
    MAX_VIDEO_DURATION = 300  # 5 minutes max
    FRAME_SKIP = 1  # Process every frame (1 = no skip, 2 = every other frame, etc.)
    OUTPUT_FPS = 30  # Output video FPS

    # Supported video formats
    SUPPORTED_FORMATS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'}

    # Output settings
    OUTPUT_DIR = RESULTS_DIR / "videos"
    OUTPUT_DIR.mkdir(exist_ok=True)


# Image processing configuration
class ImageConfig:
    """图像处理配置"""
    # Supported image formats
    SUPPORTED_FORMATS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}

    # Output settings
    DEFAULT_OUTPUT_FORMAT = 'jpg'
    DEFAULT_QUALITY = 95
    OUTPUT_DIR = RESULTS_DIR / "images"
    OUTPUT_DIR.mkdir(exist_ok=True)

# Export configuration for mobile deployment
class ExportConfig:
    # Supported export formats
    EXPORT_FORMATS = {
        'onnx': {'extension': '.onnx', 'mobile_friendly': True},
        'tflite': {'extension': '.tflite', 'mobile_friendly': True},
        'coreml': {'extension': '.mlmodel', 'mobile_friendly': True},
        'torchscript': {'extension': '.torchscript', 'mobile_friendly': True},
        'tensorrt': {'extension': '.engine', 'mobile_friendly': False}
    }
    
    # Optimization settings
    QUANTIZATION = True
    HALF_PRECISION = True
    DYNAMIC_BATCH = False

# Logging configuration
class LogConfig:
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
    
    # Create logs directory
    LOG_FILE.parent.mkdir(exist_ok=True)

# Performance configuration
class PerformanceConfig:
    # GPU settings
    USE_GPU = True
    GPU_MEMORY_FRACTION = 0.8
    GPU_MEMORY_LIMIT = 8192  # MB

    # Threading
    MAX_WORKERS = 4

    # Caching
    ENABLE_MODEL_CACHE = True
    CACHE_SIZE = 100

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
