"""
BlindStar - 智能视觉辅助系统
Enhanced Navigation Module
"""

__version__ = "2.0.0"
__author__ = "BlindStar Team"

# 导入核心模块
try:
    from . import config
    from . import core
    from . import navigation
    from . import utils
    from . import voice
    from . import accessibility

    # 导入主要服务
    from .navigation.enhanced_location import get_location_service
    from .navigation.enhanced_poi import get_poi_service
    from .navigation.enhanced_routing import get_routing_service
    from .navigation.traffic_service import get_traffic_service
    from .core.cache_manager import get_cache_manager
    from .core.fallback_manager import get_fallback_manager
    from .core.amap_client import get_amap_client
    from .config.amap_config import get_amap_config
    from .config.poi_codes import get_poi_manager

    __all__ = [
        'config', 'core', 'navigation', 'utils', 'voice', 'accessibility',
        'get_location_service', 'get_poi_service', 'get_routing_service',
        'get_traffic_service', 'get_cache_manager', 'get_fallback_manager',
        'get_amap_client', 'get_amap_config', 'get_poi_manager'
    ]

except ImportError as e:
    # 如果导入失败，提供基础功能
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []
