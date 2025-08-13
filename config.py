"""
Configuration file for YOLOv8 Object Detection and Distance Measurement System
Enhanced with navigation configuration support
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, LOGS_DIR]:
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




# Image processing configuration
class ImageConfig:
    """图像处理配置"""
    # Supported image formats
    SUPPORTED_FORMATS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}

    # Output settings
    DEFAULT_OUTPUT_FORMAT = 'jpg'
    DEFAULT_QUALITY = 95

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

# 高德地图API配置
class AmapConfig:
    """高德地图API配置"""

    # API密钥（请替换为您自己的密钥）
    API_KEY = "717d9a827c0ac3521932d3ae59aebbfe"
    API_SECRET = ""  # 如果需要的话

    # API基础URL
    BASE_URL = "https://restapi.amap.com/v3"

    # 默认搜索参数
    DEFAULT_RADIUS = 1000  # 默认搜索半径（米）
    DEFAULT_LOCATION = "116.397428,39.99923"  # 默认位置（北京天安门）
    MAX_RESULTS = 20  # 最大返回结果数

    # 导航配置
    NAVIGATION_UPDATE_INTERVAL = 5  # 导航状态更新间隔（秒）
    DEFAULT_TRAVEL_MODE = "walking"  # 默认出行方式

    # 安全配置
    REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
    RETRY_TIMES = 3  # 重试次数


# 导航配置
class NavigationConfig:
    """导航系统配置"""

    # 基础设置
    ENABLED = True
    DEFAULT_MODE = "assist"  # assist, guide, full

    # GPS设置
    GPS_UPDATE_INTERVAL = 1.0  # GPS更新间隔(秒)
    GPS_ACCURACY_THRESHOLD = 10.0  # 精度阈值(米)
    LOCATION_UPDATE_THRESHOLD = 5.0  # 位置更新阈值(米)
    ARRIVAL_THRESHOLD = 10.0  # 到达阈值(米)
    WAYPOINT_THRESHOLD = 20.0  # 路径点阈值(米)
    MAX_HISTORY_SIZE = 100  # 历史记录最大长度

    # 路径规划
    ROUTING_MODE = "walking"  # 出行方式
    AVOID_STAIRS = True  # 避开楼梯
    AVOID_CONSTRUCTION = True  # 避开施工区域
    PREFER_SIDEWALKS = True  # 优先人行道
    PREFER_ACCESSIBLE_ROUTES = True  # 优先无障碍路线
    MAX_DETOUR_RATIO = 1.5  # 最大绕行比例

    # 语音设置
    NAVIGATION_VOICE = True  # 启用导航语音
    INSTRUCTION_INTERVAL = 50  # 指令播报间隔(米)
    HAZARD_PRIORITY = True  # 危险提醒优先级
    VOICE_RATE = 150  # 语速
    VOICE_VOLUME = 1.0  # 音量
    EMERGENCY_OVERRIDE = True  # 紧急情况覆盖

    # 语音播报优先级设置
    PRIORITY_COOLDOWNS = {
        "emergency": 0.0,    # 紧急情况立即播报
        "high": 0.2,         # 高优先级短冷却
        "normal": 0.5,       # 正常冷却
        "low": 2.0           # 低优先级长冷却
    }

    # 决策融合
    VISION_WEIGHT = 0.7  # 视觉决策权重
    NAVIGATION_WEIGHT = 0.3  # 导航决策权重
    FUSION_INSTRUCTION_INTERVAL = 5.0  # 指令播报间隔(秒)
    MAX_DECISION_HISTORY = 50  # 决策历史最大长度

    # POI查询设置
    POI_DEFAULT_RADIUS = 1000  # 默认搜索半径(米)
    POI_MAX_RESULTS = 20  # 最大返回结果数
    POI_MAX_ANNOUNCE_RESULTS = 3  # 最大播报结果数
    POI_ENABLE_PINYIN_MATCHING = True  # 启用拼音匹配

    # 安全设置
    ENABLE_HAZARD_DETECTION = True  # 启用危险检测
    EMERGENCY_STOP_DISTANCE = 2.0  # 紧急停止距离(米)
    WARNING_DISTANCE = 5.0  # 警告距离(米)
    SAFE_DISTANCE = 10.0  # 安全距离(米)

    # 性能设置
    MAX_PROCESSING_FPS = 15.0  # 最大处理帧率
    ENABLE_MULTITHREADING = True  # 启用多线程
    CACHE_SIZE = 10  # 缓存大小
    ENABLE_GPU_ACCELERATION = True  # 启用GPU加速

    # 调试设置
    ENABLE_LOGGING = True  # 启用日志
    LOG_LEVEL = "INFO"  # 日志级别
    ENABLE_PERFORMANCE_MONITORING = True  # 启用性能监控
    SAVE_DEBUG_DATA = False  # 保存调试数据


class ConfigLoader:
    """配置文件加载器"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or PROJECT_ROOT / "config"
        self.logger = logging.getLogger(__name__)
        self._loaded_configs = {}

    def load_navigation_config(self, config_file: str = "navigation_config.yaml") -> Dict[str, Any]:
        """
        加载导航配置文件

        Args:
            config_file: 配置文件名

        Returns:
            配置字典
        """
        config_path = self.config_dir / config_file

        if config_path in self._loaded_configs:
            return self._loaded_configs[config_path]

        try:
            if not config_path.exists():
                self.logger.warning(f"配置文件不存在: {config_path}")
                return self._get_default_navigation_config()

            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 处理环境变量
            config = self._process_environment_variables(config)

            # 缓存配置
            self._loaded_configs[config_path] = config

            self.logger.info(f"导航配置加载成功: {config_path}")
            return config

        except Exception as e:
            self.logger.error(f"加载导航配置失败: {e}")
            return self._get_default_navigation_config()

    def _process_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置中的环境变量"""
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                default_value = config.get('defaults', {}).get(env_var, "")
                return os.getenv(env_var, default_value)
            else:
                return obj

        return replace_env_vars(config)

    def _get_default_navigation_config(self) -> Dict[str, Any]:
        """获取默认导航配置"""
        return {
            'navigation': {
                'enabled': NavigationConfig.ENABLED,
                'mode': NavigationConfig.DEFAULT_MODE,
                'gps': {
                    'update_interval': NavigationConfig.GPS_UPDATE_INTERVAL,
                    'accuracy_threshold': NavigationConfig.GPS_ACCURACY_THRESHOLD,
                    'location_update_threshold': NavigationConfig.LOCATION_UPDATE_THRESHOLD,
                    'arrival_threshold': NavigationConfig.ARRIVAL_THRESHOLD,
                    'waypoint_threshold': NavigationConfig.WAYPOINT_THRESHOLD,
                    'max_history_size': NavigationConfig.MAX_HISTORY_SIZE
                },
                'map_service': {
                    'provider': 'amap',
                    'api_key': AmapConfig.API_KEY,
                    'api_secret': AmapConfig.API_SECRET,
                    'cache_enabled': True,
                    'offline_maps': True,
                    'request_timeout': AmapConfig.REQUEST_TIMEOUT,
                    'retry_times': AmapConfig.RETRY_TIMES
                },
                'routing': {
                    'mode': NavigationConfig.ROUTING_MODE,
                    'avoid_stairs': NavigationConfig.AVOID_STAIRS,
                    'avoid_construction': NavigationConfig.AVOID_CONSTRUCTION,
                    'prefer_sidewalks': NavigationConfig.PREFER_SIDEWALKS,
                    'prefer_accessible_routes': NavigationConfig.PREFER_ACCESSIBLE_ROUTES,
                    'max_detour_ratio': NavigationConfig.MAX_DETOUR_RATIO
                },
                'voice': {
                    'navigation_voice': NavigationConfig.NAVIGATION_VOICE,
                    'instruction_interval': NavigationConfig.INSTRUCTION_INTERVAL,
                    'hazard_priority': NavigationConfig.HAZARD_PRIORITY,
                    'voice_rate': NavigationConfig.VOICE_RATE,
                    'voice_volume': NavigationConfig.VOICE_VOLUME,
                    'emergency_override': NavigationConfig.EMERGENCY_OVERRIDE,
                    'priority_settings': NavigationConfig.PRIORITY_COOLDOWNS
                },
                'fusion': {
                    'vision_weight': NavigationConfig.VISION_WEIGHT,
                    'navigation_weight': NavigationConfig.NAVIGATION_WEIGHT,
                    'emergency_override': NavigationConfig.EMERGENCY_OVERRIDE,
                    'instruction_interval': NavigationConfig.FUSION_INSTRUCTION_INTERVAL,
                    'hazard_priority': NavigationConfig.HAZARD_PRIORITY,
                    'max_history_size': NavigationConfig.MAX_DECISION_HISTORY
                },
                'poi': {
                    'default_radius': NavigationConfig.POI_DEFAULT_RADIUS,
                    'max_results': NavigationConfig.POI_MAX_RESULTS,
                    'max_announce_results': NavigationConfig.POI_MAX_ANNOUNCE_RESULTS,
                    'enable_pinyin_matching': NavigationConfig.POI_ENABLE_PINYIN_MATCHING
                },
                'safety': {
                    'enable_hazard_detection': NavigationConfig.ENABLE_HAZARD_DETECTION,
                    'emergency_stop_distance': NavigationConfig.EMERGENCY_STOP_DISTANCE,
                    'warning_distance': NavigationConfig.WARNING_DISTANCE,
                    'safe_distance': NavigationConfig.SAFE_DISTANCE
                },
                'performance': {
                    'max_processing_fps': NavigationConfig.MAX_PROCESSING_FPS,
                    'enable_multithreading': NavigationConfig.ENABLE_MULTITHREADING,
                    'cache_size': NavigationConfig.CACHE_SIZE,
                    'enable_gpu_acceleration': NavigationConfig.ENABLE_GPU_ACCELERATION
                },
                'debug': {
                    'enable_logging': NavigationConfig.ENABLE_LOGGING,
                    'log_level': NavigationConfig.LOG_LEVEL,
                    'enable_performance_monitoring': NavigationConfig.ENABLE_PERFORMANCE_MONITORING,
                    'save_debug_data': NavigationConfig.SAVE_DEBUG_DATA
                }
            }
        }

    def get_navigation_setting(self, key_path: str, default=None):
        """
        获取导航配置项

        Args:
            key_path: 配置项路径，如 'navigation.gps.update_interval'
            default: 默认值

        Returns:
            配置值
        """
        config = self.load_navigation_config()

        keys = key_path.split('.')
        value = config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


# 全局配置加载器实例
config_loader = ConfigLoader()


# POI类型映射 - 为盲人优化的常用地点分类
POI_TYPES = {
    # 餐饮服务
    "餐厅": "050000", "餐馆": "050000", "饭店": "050000", "中餐厅": "050000",
    "咖啡厅": "050100", "咖啡馆": "050100", "茶馆": "050200", "酒吧": "050400",
    "快餐": "050500", "肯德基": "050501", "麦当劳": "050502", "必胜客": "050503",

    # 购物服务
    "超市": "060100", "便利店": "060101", "商场": "060200", "购物中心": "060200",
    "市场": "060300", "菜市场": "060301", "专卖店": "060400", "药店": "061000",

    # 生活服务 - 盲人日常需要的重要服务
    "银行": "160300", "ATM": "160301", "医院": "090100", "诊所": "090200",
    "药店": "090700", "邮局": "070000", "洗衣店": "070200", "美容美发": "070300",
    "维修": "070400",

    # 交通设施 - 盲人出行的关键设施
    "地铁": "150500", "地铁站": "150500", "公交": "150700", "公交站": "150700",
    "火车站": "150200", "机场": "150100", "停车场": "150900",

    # 休闲娱乐
    "公园": "110100", "景区": "110200", "博物馆": "140400", "图书馆": "140500",
    "电影院": "130100", "KTV": "130200", "健身房": "130300", "体育馆": "130400",

    # 其他重要设施
    "酒店": "100000", "宾馆": "100000", "厕所": "200300", "无障碍厕所": "200301"
}
