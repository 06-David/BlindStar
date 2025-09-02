"""
BlindStar Navigation Module
导航功能模块
"""

try:
    from .enhanced_location import EnhancedLocationService, LocationInfo, get_location_service
    from .enhanced_poi import EnhancedPOIService, POIInfo, SearchResult, get_poi_service
    from .enhanced_routing import EnhancedRoutingService, RouteInfo, RouteStep, RouteStrategy, get_routing_service
    from .traffic_service import TrafficService, TrafficInfo, TrafficReport, get_traffic_service

    # 兼容旧版本
    try:
        from .amap_navigation import AmapNavigation
        from .poi_query import POIQuery
        from .navigation_context import NavigationContext
        from .navigation_fusion import NavigationFusion
    except ImportError:
        pass

    __all__ = [
        'EnhancedLocationService', 'LocationInfo', 'get_location_service',
        'EnhancedPOIService', 'POIInfo', 'SearchResult', 'get_poi_service',
        'EnhancedRoutingService', 'RouteInfo', 'RouteStep', 'RouteStrategy', 'get_routing_service',
        'TrafficService', 'TrafficInfo', 'TrafficReport', 'get_traffic_service'
    ]

except ImportError as e:
    print(f"Warning: Navigation module import failed: {e}")
    __all__ = []
