#!/usr/bin/env python3
"""
高德地图API配置管理模块
统一管理API密钥、调用限制、缓存策略等配置
"""

import os
import json
import time
import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class APILimits:
    """API调用限制配置"""
    daily_limit: int = 100000  # 每日调用限制
    per_second_limit: int = 200  # 每秒调用限制
    per_minute_limit: int = 6000  # 每分钟调用限制
    concurrent_limit: int = 50  # 并发调用限制


@dataclass
class APIEndpoint:
    """API端点配置"""
    name: str
    url: str
    method: str = "GET"
    timeout: int = 30
    retry_times: int = 3
    cache_ttl: int = 300  # 缓存时间(秒)
    limits: Optional[APILimits] = None


class AmapAPIConfig:
    """高德地图API配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化API配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.config_file = config_file or "config/amap_api_config.json"
        self.config_dir = Path(self.config_file).parent
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程锁
        self._lock = threading.RLock()
        
        # API调用统计
        self.call_stats = defaultdict(lambda: {
            'total_calls': 0,
            'daily_calls': 0,
            'last_reset': time.time(),
            'recent_calls': deque(maxlen=1000),  # 最近1000次调用时间
            'error_count': 0,
            'success_count': 0
        })
        
        # 加载配置
        self._load_config()
        
        # 启动统计重置定时器
        self._start_stats_reset_timer()
    
    def _load_config(self):
        """加载配置"""
        try:
            # 从环境变量获取API密钥 - BlindStar_POI应用
            self.api_key = os.getenv('AMAP_API_KEY', '请填写你的key')
            self.api_secret = os.getenv('AMAP_API_SECRET', '')
            self.app_name = "BlindStar_POI"  # 应用名称
            
            # 确保即使环境变量未设置也使用正确的默认API密钥
            if self.api_key == '' or self.api_key is None:
                self.api_key = '请填写你的key'
            
            # 基础配置
            self.base_url = "https://restapi.amap.com/v3"
            self.base_url_v4 = "https://restapi.amap.com/v4"
            self.base_url_v5 = "https://restapi.amap.com/v5"
            
            # 默认限制
            default_limits = APILimits()
            
            # API端点配置
            self.endpoints = {
                # 基础服务
                'ip_location': APIEndpoint(
                    name='IP定位',
                    url=f'{self.base_url}/ip',
                    cache_ttl=3600,  # IP定位结果缓存1小时
                    limits=APILimits(daily_limit=50000, per_second_limit=100)
                ),
                'geocoding': APIEndpoint(
                    name='地理编码',
                    url=f'{self.base_url}/geocode/geo',
                    cache_ttl=86400,  # 地理编码结果缓存24小时
                    limits=APILimits(daily_limit=50000, per_second_limit=100)
                ),
                'reverse_geocoding': APIEndpoint(
                    name='逆地理编码',
                    url=f'{self.base_url}/geocode/regeo',
                    cache_ttl=3600,
                    limits=APILimits(daily_limit=50000, per_second_limit=100)
                ),
                
                # 搜索服务
                'poi_search': APIEndpoint(
                    name='POI搜索',
                    url=f'{self.base_url}/place/around',  # 使用周边搜索，更适合盲人用户
                    cache_ttl=1800,  # POI搜索结果缓存30分钟
                    limits=APILimits(daily_limit=30000, per_second_limit=50)
                ),
                'poi_search_v2': APIEndpoint(
                    name='POI搜索V2',
                    url=f'{self.base_url_v5}/place/text',
                    cache_ttl=1800,
                    limits=APILimits(daily_limit=30000, per_second_limit=50)
                ),
                'around_search': APIEndpoint(
                    name='周边搜索',
                    url=f'{self.base_url}/place/around',
                    cache_ttl=1800,
                    limits=APILimits(daily_limit=30000, per_second_limit=50)
                ),
                'polygon_search': APIEndpoint(
                    name='多边形搜索',
                    url=f'{self.base_url}/place/polygon',
                    cache_ttl=1800,
                    limits=APILimits(daily_limit=10000, per_second_limit=20)
                ),
                'input_tips': APIEndpoint(
                    name='输入提示',
                    url=f'{self.base_url}/place/inputtips',
                    cache_ttl=300,
                    limits=APILimits(daily_limit=100000, per_second_limit=200)
                ),
                
                # 路径规划服务
                'walking_route': APIEndpoint(
                    name='步行路径规划',
                    url=f'{self.base_url}/direction/walking',
                    cache_ttl=1800,
                    limits=APILimits(daily_limit=100000, per_second_limit=200)
                ),
                'walking_route_v2': APIEndpoint(
                    name='步行路径规划V2',
                    url=f'{self.base_url_v5}/direction/walking',
                    cache_ttl=1800,
                    limits=APILimits(daily_limit=100000, per_second_limit=200)
                ),
                'driving_route': APIEndpoint(
                    name='驾车路径规划',
                    url=f'{self.base_url}/direction/driving',
                    cache_ttl=1800,
                    limits=APILimits(daily_limit=100000, per_second_limit=200)
                ),
                'transit_route': APIEndpoint(
                    name='公交路径规划',
                    url=f'{self.base_url}/direction/transit/integrated',
                    cache_ttl=1800,
                    limits=APILimits(daily_limit=50000, per_second_limit=100)
                ),
                
                # 距离测量服务
                'distance_measure': APIEndpoint(
                    name='距离测量',
                    url=f'{self.base_url}/distance',
                    cache_ttl=3600,
                    limits=APILimits(daily_limit=100000, per_second_limit=200)
                ),
                
                # 坐标转换服务
                'coordinate_convert': APIEndpoint(
                    name='坐标转换',
                    url=f'{self.base_url}/assistant/coordinate/convert',
                    cache_ttl=86400,
                    limits=APILimits(daily_limit=100000, per_second_limit=200)
                ),
                
                # 天气服务
                'weather_info': APIEndpoint(
                    name='天气信息',
                    url=f'{self.base_url}/weather/weatherInfo',
                    cache_ttl=1800,
                    limits=APILimits(daily_limit=10000, per_second_limit=10)
                ),
                
                # 交通态势服务
                'traffic_status': APIEndpoint(
                    name='交通态势',
                    url=f'{self.base_url}/traffic/status/rectangle',
                    cache_ttl=300,
                    limits=APILimits(daily_limit=10000, per_second_limit=20)
                ),
                'traffic_circle': APIEndpoint(
                    name='圆形区域交通态势',
                    url=f'{self.base_url}/traffic/status/circle',
                    cache_ttl=300,
                    limits=APILimits(daily_limit=10000, per_second_limit=20)
                ),
                'traffic_road': APIEndpoint(
                    name='道路交通态势',
                    url=f'{self.base_url}/traffic/status/road',
                    cache_ttl=300,
                    limits=APILimits(daily_limit=10000, per_second_limit=20)
                ),
                
                # 行政区域服务
                'district_info': APIEndpoint(
                    name='行政区域查询',
                    url=f'{self.base_url}/config/district',
                    cache_ttl=86400,
                    limits=APILimits(daily_limit=10000, per_second_limit=10)
                ),
                
                # 道路服务
                'road_info': APIEndpoint(
                    name='道路信息',
                    url=f'{self.base_url}/road/status',
                    cache_ttl=300,
                    limits=APILimits(daily_limit=10000, per_second_limit=20)
                ),
            }
            
            logger.info(f"✅ 高德地图API配置加载完成，API Key: {self.api_key[:6]}...{self.api_key[-4:]}")
            
        except Exception as e:
            logger.error(f"❌ 高德地图API配置加载失败: {e}")
            # 使用默认配置
            self.api_key = '请填写你的key'
            self.api_secret = ''
            self.app_name = "BlindStar_POI"
            self.base_url = "https://restapi.amap.com/v3"
            self.base_url_v4 = "https://restapi.amap.com/v4"
            self.base_url_v5 = "https://restapi.amap.com/v5"
            
            # 默认端点配置
            default_limits = APILimits()
            self.endpoints = {
                'ip_location': APIEndpoint(name='IP定位', url=f'{self.base_url}/ip'),
                'geocoding': APIEndpoint(name='地理编码', url=f'{self.base_url}/geocode/geo'),
                'reverse_geocoding': APIEndpoint(name='逆地理编码', url=f'{self.base_url}/geocode/regeo'),
                'poi_search': APIEndpoint(name='POI搜索', url=f'{self.base_url}/place/around'),
                'around_search': APIEndpoint(name='周边搜索', url=f'{self.base_url}/place/around'),
                'walking_route': APIEndpoint(name='步行路径规划', url=f'{self.base_url}/direction/walking'),
            }
            
            logger.info("✅ 使用默认高德地图API配置")
    
    def _start_stats_reset_timer(self):
        """启动统计重置定时器"""
        def reset_daily_stats():
            while True:
                time.sleep(86400)  # 每24小时
                with self._lock:
                    for stats in self.call_stats.values():
                        stats['daily_calls'] = 0
                        stats['last_reset'] = time.time()
        
        timer_thread = threading.Thread(target=reset_daily_stats, daemon=True)
        timer_thread.start()
    
    def get_endpoint(self, endpoint_name: str) -> Optional[APIEndpoint]:
        """
        获取API端点配置
        
        Args:
            endpoint_name: 端点名称
            
        Returns:
            APIEndpoint: 端点配置对象
        """
        return self.endpoints.get(endpoint_name)
    
    def get_api_key(self) -> str:
        """
        获取API密钥
        
        Returns:
            str: API密钥
        """
        return self.api_key
    
    def get_base_url(self, version: str = "v3") -> str:
        """
        获取基础URL
        
        Args:
            version: API版本 (v3, v4, v5)
            
        Returns:
            str: 基础URL
        """
        if version == "v3":
            return self.base_url
        elif version == "v4":
            return self.base_url_v4
        elif version == "v5":
            return self.base_url_v5
        else:
            return self.base_url
    
    def record_call(self, endpoint_name: str, success: bool = True):
        """
        记录API调用
        
        Args:
            endpoint_name: 端点名称
            success: 是否成功
        """
        with self._lock:
            stats = self.call_stats[endpoint_name]
            stats['total_calls'] += 1
            stats['daily_calls'] += 1
            stats['recent_calls'].append(time.time())
            
            if success:
                stats['success_count'] += 1
            else:
                stats['error_count'] += 1
    
    def can_make_request(self, endpoint_name: str) -> Tuple[bool, str]:
        """
        检查是否可以发起API请求（限流检查）
        
        Args:
            endpoint_name: 端点名称
            
        Returns:
            (是否可以请求, 原因)
        """
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            return True, "端点配置不存在"
        
        limits = endpoint.limits
        if not limits:
            return True, "无限制"
        
        with self._lock:
            stats = self.call_stats[endpoint_name]
            current_time = time.time()
            
            # 检查每日限制
            if stats['daily_calls'] >= limits.daily_limit:
                return False, "超过每日调用限制"
            
            # 检查并发限制
            recent_calls = [t for t in stats['recent_calls'] if current_time - t < 1]
            if len(recent_calls) >= limits.concurrent_limit:
                return False, "超过并发调用限制"
            
            # 检查每秒限制
            recent_calls = [t for t in stats['recent_calls'] if current_time - t < 1]
            if len(recent_calls) >= limits.per_second_limit:
                return False, "超过每秒调用限制"
            
            # 检查每分钟限制
            recent_calls = [t for t in stats['recent_calls'] if current_time - t < 60]
            if len(recent_calls) >= limits.per_minute_limit:
                return False, "超过每分钟调用限制"
        
        return True, "允许调用"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取API调用统计
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            return {
                endpoint: dict(stats) 
                for endpoint, stats in self.call_stats.items()
            }
    
    def get_api_stats(self) -> Dict[str, Any]:
        """
        获取API统计信息（用于健康检查）
        
        Returns:
            Dict: API统计信息
        """
        with self._lock:
            stats = {}
            current_time = time.time()
            
            for endpoint_name, endpoint_stats in self.call_stats.items():
                # 计算最近1分钟、5分钟、1小时的调用次数
                calls_1min = 0
                calls_5min = 0
                calls_1hour = 0
                
                for call_time in endpoint_stats['recent_calls']:
                    if current_time - call_time <= 60:
                        calls_1min += 1
                    if current_time - call_time <= 300:
                        calls_5min += 1
                    if current_time - call_time <= 3600:
                        calls_1hour += 1
                
                stats[endpoint_name] = {
                    'total_calls': endpoint_stats['total_calls'],
                    'daily_calls': endpoint_stats['daily_calls'],
                    'success_count': endpoint_stats['success_count'],
                    'error_count': endpoint_stats['error_count'],
                    'calls_per_minute': calls_1min,
                    'calls_per_5minute': calls_5min,
                    'calls_per_hour': calls_1hour,
                    'success_rate': (
                        endpoint_stats['success_count'] / endpoint_stats['total_calls'] 
                        if endpoint_stats['total_calls'] > 0 else 0
                    )
                }
            
            return stats

# 全局配置实例
_amap_config = None

def get_amap_config() -> AmapAPIConfig:
    """
    获取高德地图API配置单例
    
    Returns:
        AmapAPIConfig: 高德地图API配置实例
    """
    global _amap_config
    if _amap_config is None:
        _amap_config = AmapAPIConfig()
    return _amap_config

def init_amap_config(config_file: Optional[str] = None) -> AmapAPIConfig:
    """
    初始化高德地图API配置
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        AmapAPIConfig: 高德地图API配置实例
    """
    global _amap_config
    _amap_config = AmapAPIConfig(config_file)
    return _amap_config
