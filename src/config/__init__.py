"""
BlindStar Configuration Module
配置管理模块
"""

try:
    from .amap_config import AmapAPIConfig, get_amap_config, init_amap_config
    from .poi_codes import POICodeManager, get_poi_manager
    
    __all__ = [
        'AmapAPIConfig', 'get_amap_config', 'init_amap_config',
        'POICodeManager', 'get_poi_manager'
    ]
    
except ImportError as e:
    print(f"Warning: Config module import failed: {e}")
    __all__ = []
