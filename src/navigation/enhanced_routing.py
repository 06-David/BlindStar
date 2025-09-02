#!/usr/bin/env python3
"""
增强路径规划服务
基于高德地图路径规划2.0 API，提供盲人友好的路径规划功能
"""

import time
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import threading
import urllib3
from enum import Enum

try:
    from ..core.amap_client import get_amap_client
    from .enhanced_location import LocationInfo, get_location_service
except ImportError:
    try:
        # 尝试绝对导入
        from src.core.amap_client import get_amap_client
        from src.navigation.enhanced_location import LocationInfo, get_location_service
    except ImportError:
        # 兼容性处理
        get_amap_client = None
        LocationInfo = None
        get_location_service = None

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class RouteStrategy(Enum):
    """路径规划策略"""
    FASTEST = "0"           # 最快路径
    SHORTEST = "1"          # 最短路径
    AVOID_CONGESTION = "2"  # 避开拥堵
    AVOID_HIGHWAY = "3"     # 避开高速
    AVOID_TOLL = "4"        # 避开收费
    AVOID_FERRY = "5"       # 避开轮渡
    SAFE_PRIORITY = "6"     # 安全优先（盲人友好）
    ACCESSIBLE = "7"        # 无障碍优先
    SIDEWALK_PRIORITY = "8" # 人行道优先


@dataclass
class RouteStep:
    """路径步骤"""
    instruction: str        # 导航指令
    distance: float        # 距离（米）
    duration: int          # 时间（秒）
    polyline: str          # 路径坐标串
    action: str            # 动作类型
    orientation: str       # 方向
    road_name: str         # 道路名称
    assistant_action: str  # 辅助动作
    
    # 盲人友好信息
    has_sidewalk: bool = False      # 是否有人行道
    has_blind_path: bool = False    # 是否有盲道
    traffic_light: bool = False     # 是否有红绿灯
    crosswalk: bool = False         # 是否有人行横道
    hazard_level: int = 0           # 危险等级 0-3
    accessibility_note: str = ""    # 无障碍说明


@dataclass
class RouteInfo:
    """路径信息"""
    origin: LocationInfo
    destination: LocationInfo
    distance: float         # 总距离（米）
    duration: int          # 总时间（秒）
    steps: List[RouteStep] = field(default_factory=list)
    strategy: RouteStrategy = RouteStrategy.SAFE_PRIORITY
    polyline: str = ""     # 完整路径坐标串
    
    # 盲人友好统计
    sidewalk_coverage: float = 0.0    # 人行道覆盖率
    blind_path_coverage: float = 0.0  # 盲道覆盖率
    traffic_lights_count: int = 0     # 红绿灯数量
    crosswalks_count: int = 0         # 人行横道数量
    overall_safety_score: float = 0.0 # 整体安全评分
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'origin': self.origin.to_dict() if self.origin else None,
            'destination': self.destination.to_dict() if self.destination else None,
            'distance': self.distance,
            'duration': self.duration,
            'steps': [step.__dict__ for step in self.steps],
            'strategy': self.strategy.value,
            'polyline': self.polyline,
            'sidewalk_coverage': self.sidewalk_coverage,
            'blind_path_coverage': self.blind_path_coverage,
            'traffic_lights_count': self.traffic_lights_count,
            'crosswalks_count': self.crosswalks_count,
            'overall_safety_score': self.overall_safety_score
        }


class EnhancedRoutingService:
    """增强路径规划服务"""
    
    def __init__(self, api_key: str = "请填写你的key"):
        """
        初始化增强路径规划服务
        
        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3"
        self.base_url_v4 = "https://restapi.amap.com/v4"
        
        # 获取API客户端和定位服务
        if get_amap_client:
            try:
                self.client = get_amap_client()
            except Exception as e:
                logger.warning(f"无法初始化API客户端: {e}")
                self.client = None
        else:
            self.client = None
        
        if get_location_service:
            try:
                self.location_service = get_location_service()
            except Exception as e:
                logger.warning(f"无法初始化定位服务: {e}")
                self.location_service = None
        else:
            self.location_service = None
        
        # 路径缓存
        self._route_cache = {}
        self._cache_lock = threading.RLock()
        self.cache_ttl = 600  # 10分钟缓存
        
        # 盲人友好的避让区域和道路
        self.avoid_areas = []  # 避让区域
        self.avoid_roads = []  # 避让道路
        
        # 盲人友好的路径权重配置
        self.blind_friendly_weights = {
            'sidewalk_bonus': 0.3,      # 人行道加分
            'blind_path_bonus': 0.5,    # 盲道加分
            'traffic_light_penalty': 0.2, # 红绿灯扣分
            'busy_road_penalty': 0.4,   # 繁忙道路扣分
            'construction_penalty': 0.8, # 施工区域扣分
            'accessibility_bonus': 0.6   # 无障碍设施加分
        }
        
        logger.info("✅ 增强路径规划服务初始化完成")
    
    def plan_route(self, 
                   origin: str, 
                   destination: str,
                   strategy: RouteStrategy = RouteStrategy.SAFE_PRIORITY,
                   avoid_areas: List[str] = None,
                   avoid_roads: List[str] = None) -> Optional[RouteInfo]:
        """
        规划路径
        
        Args:
            origin: 起点（地址或坐标）
            destination: 终点（地址或坐标）
            strategy: 路径策略
            avoid_areas: 避让区域列表
            avoid_roads: 避让道路列表
            
        Returns:
            RouteInfo: 路径信息
        """
        try:
            # 解析起点和终点坐标
            origin_location = self._parse_location(origin)
            destination_location = self._parse_location(destination)
            
            if not origin_location or not destination_location:
                logger.error("无法解析起点或终点坐标")
                return None
            
            # 检查缓存
            cache_key = f"{origin}_{destination}_{strategy.value}"
            cached_route = self._get_cached_route(cache_key)
            if cached_route:
                logger.debug("🎯 使用缓存的路径信息")
                return cached_route
            
            # 调用路径规划API
            route_data = self._call_routing_api(
                origin_location, 
                destination_location, 
                strategy,
                avoid_areas or [],
                avoid_roads or []
            )
            
            if not route_data:
                return None
            
            # 解析路径数据
            route_info = self._parse_route_data(
                route_data, 
                origin_location, 
                destination_location, 
                strategy
            )
            
            # 增强盲人友好信息
            if route_info:
                self._enhance_blind_friendly_info(route_info)
                
                # 缓存结果
                self._cache_route(cache_key, route_info)
            
            return route_info
            
        except Exception as e:
            logger.error(f"路径规划失败: {e}")
            return None
    
    def _parse_location(self, location: str) -> Optional[LocationInfo]:
        """解析位置信息"""
        try:
            # 检查是否是坐标格式
            if ',' in location:
                coords = location.split(',')
                if len(coords) >= 2:
                    try:
                        lon, lat = float(coords[0]), float(coords[1])
                        return LocationInfo(
                            longitude=lon,
                            latitude=lat,
                            accuracy=10.0,
                            address=location,
                            city="",
                            district="",
                            province="",
                            location_type="coordinate"
                        )
                    except ValueError:
                        pass
            
            # 如果不是坐标，进行地理编码
            if self.location_service:
                return self.location_service.geocode_address(location)
            else:
                # 简化版地理编码
                url = f"{self.base_url}/geocode/geo"
                params = {
                    "key": self.api_key,
                    "address": location,
                    "output": "json"
                }
                
                response = requests.get(url, params=params, timeout=15, verify=False)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == '1':
                    geocodes = data.get('geocodes', [])
                    if geocodes:
                        geocode = geocodes[0]
                        coords = geocode.get('location', '').split(',')
                        
                        if len(coords) >= 2:
                            return LocationInfo(
                                longitude=float(coords[0]),
                                latitude=float(coords[1]),
                                accuracy=50.0,
                                address=geocode.get('formatted_address', location),
                                city=geocode.get('city', ''),
                                district=geocode.get('district', ''),
                                province=geocode.get('province', ''),
                                location_type="geocoding"
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"解析位置失败: {e}")
            return None
    
    def _call_routing_api(self, 
                         origin: LocationInfo, 
                         destination: LocationInfo,
                         strategy: RouteStrategy,
                         avoid_areas: List[str],
                         avoid_roads: List[str]) -> Optional[Dict[str, Any]]:
        """调用路径规划API"""
        try:
            # 构建请求参数 - 根据官方文档
            params = {
                "origin": origin.to_coordinates(),
                "destination": destination.to_coordinates(),
                "output": "json"
            }

            # 添加避让参数（如果支持）
            if avoid_areas:
                params["avoidpolygons"] = "|".join(avoid_areas)

            # 步行路径规划不支持avoidroad参数，跳过

            # 使用标准的步行路径规划API
            if self.client:
                success, data, error = self.client.request(
                    "walking_route",
                    **params
                )
            else:
                # 直接请求官方步行路径规划API
                url = f"{self.base_url}/direction/walking"
                params["key"] = self.api_key

                logger.debug(f"步行路径规划请求: {url}, 参数: {params}")

                response = requests.get(url, params=params, timeout=30, verify=False)
                response.raise_for_status()
                data = response.json()
                success = data.get('status') == '1'

                logger.debug(f"步行路径规划响应: status={data.get('status')}, info={data.get('info')}")
            
            if success and data:
                return data
            else:
                logger.error(f"路径规划API调用失败: {data.get('info', '未知错误') if data else error}")
                return None
                
        except Exception as e:
            logger.error(f"路径规划API调用异常: {e}")
            return None
    
    def _parse_route_data(self, 
                         route_data: Dict[str, Any], 
                         origin: LocationInfo, 
                         destination: LocationInfo,
                         strategy: RouteStrategy) -> Optional[RouteInfo]:
        """解析路径数据"""
        try:
            route = route_data.get('route', {})
            paths = route.get('paths', [])
            
            if not paths:
                logger.error("路径数据为空")
                return None
            
            # 选择第一条路径
            path = paths[0]
            
            # 创建路径信息
            route_info = RouteInfo(
                origin=origin,
                destination=destination,
                distance=float(path.get('distance', 0)),
                duration=int(path.get('duration', 0)),
                strategy=strategy,
                polyline=path.get('polyline', '')
            )
            
            # 解析步骤
            steps = path.get('steps', [])
            for step_data in steps:
                step = RouteStep(
                    instruction=step_data.get('instruction', ''),
                    distance=float(step_data.get('distance', 0)),
                    duration=int(step_data.get('duration', 0)),
                    polyline=step_data.get('polyline', ''),
                    action=step_data.get('action', ''),
                    orientation=step_data.get('orientation', ''),
                    road_name=step_data.get('road', ''),
                    assistant_action=step_data.get('assistant_action', '')
                )
                
                route_info.steps.append(step)
            
            return route_info
            
        except Exception as e:
            logger.error(f"解析路径数据失败: {e}")
            return None
    
    def _enhance_blind_friendly_info(self, route_info: RouteInfo):
        """增强盲人友好信息"""
        try:
            total_distance = route_info.distance
            sidewalk_distance = 0.0
            blind_path_distance = 0.0
            traffic_lights = 0
            crosswalks = 0
            
            for step in route_info.steps:
                # 基于道路名称和指令推断盲人友好信息
                instruction = str(step.instruction).lower() if step.instruction else ""
                road_name = str(step.road_name).lower() if step.road_name else ""
                
                # 判断是否有人行道
                if any(keyword in instruction or keyword in road_name 
                       for keyword in ['人行道', '步行', '行人', '人行横道']):
                    step.has_sidewalk = True
                    sidewalk_distance += step.distance
                
                # 判断是否有盲道
                if any(keyword in instruction or keyword in road_name 
                       for keyword in ['盲道', '导盲', '无障碍']):
                    step.has_blind_path = True
                    blind_path_distance += step.distance
                
                # 判断是否有红绿灯
                if any(keyword in instruction 
                       for keyword in ['红绿灯', '信号灯', '交通灯', '路口']):
                    step.traffic_light = True
                    traffic_lights += 1
                
                # 判断是否有人行横道
                if any(keyword in instruction 
                       for keyword in ['人行横道', '斑马线', '过马路']):
                    step.crosswalk = True
                    crosswalks += 1
                
                # 评估危险等级
                step.hazard_level = self._assess_hazard_level(step)
                
                # 生成无障碍说明
                step.accessibility_note = self._generate_accessibility_note(step)
            
            # 计算统计信息
            if total_distance > 0:
                route_info.sidewalk_coverage = sidewalk_distance / total_distance
                route_info.blind_path_coverage = blind_path_distance / total_distance
            
            route_info.traffic_lights_count = traffic_lights
            route_info.crosswalks_count = crosswalks
            
            # 计算整体安全评分
            route_info.overall_safety_score = self._calculate_safety_score(route_info)
            
        except Exception as e:
            logger.error(f"增强盲人友好信息失败: {e}")
    
    def _assess_hazard_level(self, step: RouteStep) -> int:
        """评估步骤的危险等级"""
        hazard_level = 0
        
        instruction = str(step.instruction).lower() if step.instruction else ""
        road_name = str(step.road_name).lower() if step.road_name else ""
        
        # 基于关键词评估危险等级
        if any(keyword in instruction or keyword in road_name 
               for keyword in ['高速', '快速路', '主干道']):
            hazard_level += 2
        
        if any(keyword in instruction 
               for keyword in ['左转', '右转', '掉头']):
            hazard_level += 1
        
        if any(keyword in instruction 
               for keyword in ['施工', '封闭', '维修']):
            hazard_level += 3
        
        return min(hazard_level, 3)  # 最高等级为3
    
    def _generate_accessibility_note(self, step: RouteStep) -> str:
        """生成无障碍说明"""
        notes = []
        
        if step.has_sidewalk:
            notes.append("有人行道")
        
        if step.has_blind_path:
            notes.append("有盲道")
        
        if step.traffic_light:
            notes.append("注意红绿灯")
        
        if step.crosswalk:
            notes.append("人行横道过街")
        
        if step.hazard_level >= 2:
            notes.append("注意安全")
        
        return "，".join(notes)
    
    def _calculate_safety_score(self, route_info: RouteInfo) -> float:
        """计算整体安全评分"""
        try:
            score = 0.5  # 基础分
            
            # 人行道覆盖率加分
            score += route_info.sidewalk_coverage * 0.3
            
            # 盲道覆盖率加分
            score += route_info.blind_path_coverage * 0.2
            
            # 红绿灯数量适中加分
            if 1 <= route_info.traffic_lights_count <= 3:
                score += 0.1
            elif route_info.traffic_lights_count > 5:
                score -= 0.1
            
            # 危险步骤扣分
            high_risk_steps = sum(1 for step in route_info.steps if step.hazard_level >= 2)
            score -= high_risk_steps * 0.05
            
            return max(0.0, min(1.0, score))  # 限制在0-1范围内
            
        except Exception as e:
            logger.error(f"计算安全评分失败: {e}")
            return 0.5
    
    def _get_cached_route(self, cache_key: str) -> Optional[RouteInfo]:
        """获取缓存的路径信息"""
        with self._cache_lock:
            if cache_key in self._route_cache:
                cached_data = self._route_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['route']
                else:
                    # 缓存过期，删除
                    del self._route_cache[cache_key]
        return None
    
    def _cache_route(self, cache_key: str, route_info: RouteInfo):
        """缓存路径信息"""
        with self._cache_lock:
            self._route_cache[cache_key] = {
                'route': route_info,
                'timestamp': time.time()
            }


# 全局路径规划服务实例
_global_routing_service = None

def get_routing_service() -> EnhancedRoutingService:
    """获取全局路径规划服务实例"""
    global _global_routing_service
    if _global_routing_service is None:
        _global_routing_service = EnhancedRoutingService()
    return _global_routing_service

# 为EnhancedRoutingService类添加新方法
def add_enhanced_walking_methods():
    """为路径规划服务添加增强的步行路径规划方法"""

    def plan_walking_route_enhanced(self, origin, destination,
                                   origin_poi_id: str = None, destination_poi_id: str = None):
        """
        增强步行路径规划 - 基于高德步行路径规划API
        """
        try:
            logger.info(f"🚶 开始增强步行路径规划")
            logger.info(f"起点: ({origin.longitude:.6f}, {origin.latitude:.6f})")
            logger.info(f"终点: ({destination.longitude:.6f}, {destination.latitude:.6f})")

            # 构建API请求参数
            params = {
                'origin': f"{origin.longitude},{origin.latitude}",
                'destination': f"{destination.longitude},{destination.latitude}",
                'output': 'json'
            }

            # 添加POI ID以提升准确性
            if origin_poi_id:
                params['origin_id'] = origin_poi_id
                logger.debug(f"使用起点POI ID: {origin_poi_id}")
            if destination_poi_id:
                params['destination_id'] = destination_poi_id
                logger.debug(f"使用终点POI ID: {destination_poi_id}")

            # 智能API选择：优先v3，如果失败再尝试v5
            route_info = None

            # 首先尝试v3 API（个人账户通常有权限）
            logger.info("🚶 尝试v3步行路径规划API")
            route_info = self._try_v3_walking_api(origin, destination, origin_poi_id, destination_poi_id)

            if route_info:
                logger.info("✅ v3步行路径规划成功")
                return route_info

            # 如果v3失败，尝试v5 API
            logger.info("🔄 v3失败，尝试v5步行路径规划API")
            route_info = self._try_v5_walking_api(origin, destination, origin_poi_id, destination_poi_id)

            if route_info:
                logger.info("✅ v5步行路径规划成功")
                return route_info

            # 如果都失败，使用简单距离估算
            logger.warning("⚠️ 所有API都失败，使用简单距离估算")
            route_info = self._create_simple_route_estimation(origin, destination)

            return route_info

        except Exception as e:
            logger.error(f"❌ 步行路径规划异常: {e}")
            return None

    def _parse_walking_route_data(self, data: dict):
        """解析步行路径规划API返回的数据"""
        try:
            route = data.get('route', {})
            paths = route.get('paths', [])

            if not paths:
                logger.warning("路径数据为空")
                return None

            # 取第一个路径方案
            path = paths[0]

            # 基本信息
            total_distance = int(path.get('distance', 0))
            total_duration = int(path.get('duration', 0))

            # 解析步骤
            steps_data = path.get('steps', [])
            steps = []

            for i, step_data in enumerate(steps_data):
                step = self._parse_walking_step(step_data, i + 1)
                if step:
                    steps.append(step)

            # 创建路径信息
            route_info = RouteInfo(
                origin=None,  # 简化处理
                destination=None,  # 简化处理
                distance=total_distance,
                duration=total_duration,
                steps=steps,
                polyline=""  # 简化处理
            )

            logger.info(f"解析完成: {len(steps)}个步骤, 总距离{total_distance}米")
            return route_info

        except Exception as e:
            logger.error(f"解析步行路径数据失败: {e}")
            return None

    def _parse_walking_step(self, step_data: dict, step_number: int):
        """解析单个步行步骤"""
        try:
            instruction = step_data.get('instruction', '')
            road = step_data.get('road', '')
            distance = int(step_data.get('distance', 0))
            duration = int(step_data.get('duration', 0))
            orientation = step_data.get('orientation', '')
            action = step_data.get('action', '')
            assistant_action = step_data.get('assistant_action', '')
            walk_type = int(step_data.get('walk_type', 0))

            # 处理可能的数组类型字段
            if isinstance(road, list):
                road = road[0] if road else ''
            if isinstance(orientation, list):
                orientation = orientation[0] if orientation else ''
            if isinstance(action, list):
                action = action[0] if action else ''
            if isinstance(assistant_action, list):
                assistant_action = assistant_action[0] if assistant_action else ''

            # 生成盲人友好的指令
            blind_friendly_instruction = self._generate_blind_friendly_instruction(
                instruction, road, action, assistant_action, walk_type, distance, orientation
            )

            step = RouteStep(
                instruction=blind_friendly_instruction,
                distance=distance,
                duration=duration,
                polyline=polyline,
                action=action,
                orientation=orientation,
                road_name=road,
                assistant_action=assistant_action,
                accessibility_note=self._get_walk_type_accessibility_info(walk_type)
            )

            logger.debug(f"步骤{step_number}: {blind_friendly_instruction}")
            return step

        except Exception as e:
            logger.error(f"解析步行步骤失败: {e}")
            return None

    def _parse_walking_route_data_v5(self, data: dict):
        """解析v5步行路径规划API返回的数据"""
        try:
            route = data.get('route', {})
            paths = route.get('paths', [])

            if not paths:
                logger.warning("v5路径数据为空")
                return None

            # 取第一个路径方案
            path = paths[0]

            # 基本信息
            total_distance = int(path.get('distance', 0))

            # 获取用时信息
            cost = path.get('cost', {})
            total_duration = int(cost.get('duration', 0)) if cost else 0

            # 解析步骤
            steps_data = path.get('steps', [])
            steps = []

            for i, step_data in enumerate(steps_data):
                step = self._parse_walking_step_v5(step_data, i + 1)
                if step:
                    steps.append(step)

            # 创建路径信息
            route_info = RouteInfo(
                origin=None,  # 简化处理
                destination=None,  # 简化处理
                distance=total_distance,
                duration=total_duration,
                steps=steps,
                polyline=""  # 简化处理
            )

            logger.info(f"v5解析完成: {len(steps)}个步骤, 总距离{total_distance}米, 用时{total_duration}秒")
            return route_info

        except Exception as e:
            logger.error(f"解析v5步行路径数据失败: {e}")
            return None

    def _parse_walking_step_v5(self, step_data: dict, step_number: int):
        """解析v5单个步行步骤"""
        try:
            instruction = step_data.get('instruction', '')
            road_name = step_data.get('road_name', '')
            step_distance = int(step_data.get('step_distance', 0))
            orientation = step_data.get('orientation', '')

            # v5 API中的导航信息（如果有show_fields=navi）
            action = step_data.get('action', '')
            assistant_action = step_data.get('assistant_action', '')
            walk_type = int(step_data.get('walk_type', 0))
            polyline = step_data.get('polyline', '')

            # 估算用时（按照平均步行速度1.25米/秒）
            step_duration = int(step_distance / 1.25) if step_distance > 0 else 0

            # 生成盲人友好的指令
            blind_friendly_instruction = self._generate_blind_friendly_instruction_v5(
                instruction, road_name, action, assistant_action, walk_type, step_distance, orientation
            )

            step = RouteStep(
                instruction=blind_friendly_instruction,
                distance=step_distance,
                duration=step_duration,
                polyline=polyline,
                action=action,
                orientation=orientation,
                road_name=road_name,
                assistant_action=assistant_action,
                accessibility_note=self._get_walk_type_accessibility_info(walk_type)
            )

            logger.debug(f"v5步骤{step_number}: {blind_friendly_instruction}")
            return step

        except Exception as e:
            logger.error(f"解析v5步行步骤失败: {e}")
            return None

    def _generate_blind_friendly_instruction_v5(self, instruction: str, road_name: str, action: str,
                                               assistant_action: str, walk_type: int, distance: int, orientation: str) -> str:
        """为v5 API生成盲人友好的导航指令"""
        try:
            # 基础指令
            friendly_instruction = instruction

            # 添加距离描述
            if distance > 0:
                if distance < 50:
                    distance_desc = f"短距离{distance}米"
                elif distance < 200:
                    distance_desc = f"中等距离{distance}米"
                else:
                    distance_desc = f"较长距离{distance}米"

                # 替换原始距离描述
                friendly_instruction = friendly_instruction.replace(f"{distance}米", distance_desc)

            # 添加道路信息
            if road_name:
                if road_name not in friendly_instruction:
                    friendly_instruction += f"，沿{road_name}"

            # 添加方向信息
            if orientation:
                direction_map = {
                    '东': '向东',
                    '南': '向南',
                    '西': '向西',
                    '北': '向北',
                    '东南': '向东南',
                    '西南': '向西南',
                    '东北': '向东北',
                    '西北': '向西北'
                }
                if orientation in direction_map and direction_map[orientation] not in friendly_instruction:
                    friendly_instruction += f"，{direction_map[orientation]}"

            # 添加特殊路段信息
            walk_type_info = self._get_walk_type_description(walk_type)
            if walk_type_info:
                friendly_instruction += f"，{walk_type_info}"

            # 添加安全提醒
            safety_reminder = self._get_safety_reminder_v5(action, walk_type, distance)
            if safety_reminder:
                friendly_instruction += f"。{safety_reminder}"

            # 添加到达提醒
            if "到达目的地" in instruction or assistant_action == "到达目的地":
                friendly_instruction += "。恭喜您已到达目的地"

            return friendly_instruction

        except Exception as e:
            logger.error(f"生成v5盲人友好指令失败: {e}")
            return instruction

    def _get_safety_reminder_v5(self, action: str, walk_type: int, distance: int) -> str:
        """获取v5安全提醒"""
        reminders = []

        # 基于动作的提醒
        if action and "转" in action:
            reminders.append("转弯时请注意周围环境")

        # 基于路段类型的提醒
        if walk_type == 1:  # 人行横道
            reminders.append("过马路时请注意车辆，确保安全")
        elif walk_type == 4:  # 过街天桥
            reminders.append("上下天桥时请扶好扶手")
        elif walk_type == 8:  # 扶梯
            reminders.append("乘坐扶梯时请站稳扶好")
        elif walk_type == 20:  # 台阶
            reminders.append("上下台阶时请小心脚下")
        elif walk_type == 21:  # 斜坡
            reminders.append("斜坡路段请注意脚下安全")

        # 基于距离的提醒
        if distance > 500:
            reminders.append("较长路段，请注意休息")

        return "，".join(reminders)

    # 将方法绑定到类
    EnhancedRoutingService.plan_walking_route_enhanced = plan_walking_route_enhanced
    EnhancedRoutingService._parse_walking_route_data = _parse_walking_route_data
    EnhancedRoutingService._parse_walking_step = _parse_walking_step
    EnhancedRoutingService._parse_walking_route_data_v5 = _parse_walking_route_data_v5
    EnhancedRoutingService._parse_walking_step_v5 = _parse_walking_step_v5
    EnhancedRoutingService._generate_blind_friendly_instruction_v5 = _generate_blind_friendly_instruction_v5
    EnhancedRoutingService._get_safety_reminder_v5 = _get_safety_reminder_v5

    def _generate_blind_friendly_instruction(self, instruction: str, road: str, action: str,
                                           assistant_action: str, walk_type: int, distance: int, orientation: str) -> str:
        """生成盲人友好的导航指令"""
        try:
            # 基础指令
            friendly_instruction = instruction

            # 添加距离信息
            if distance > 0:
                if distance < 50:
                    distance_desc = f"短距离{distance}米"
                elif distance < 200:
                    distance_desc = f"中等距离{distance}米"
                else:
                    distance_desc = f"较长距离{distance}米"

                friendly_instruction = friendly_instruction.replace(f"{distance}米", distance_desc)

            # 添加道路信息
            if road:
                friendly_instruction += f"，沿{road}"

            # 添加方向信息
            if orientation:
                direction_map = {
                    '东': '向东',
                    '南': '向南',
                    '西': '向西',
                    '北': '向北',
                    '东南': '向东南',
                    '西南': '向西南',
                    '东北': '向东北',
                    '西北': '向西北'
                }
                if orientation in direction_map:
                    friendly_instruction += f"，{direction_map[orientation]}"

            # 添加特殊路段信息
            walk_type_info = self._get_walk_type_description(walk_type)
            if walk_type_info:
                friendly_instruction += f"，{walk_type_info}"

            # 添加安全提醒
            safety_reminder = self._get_safety_reminder(action, walk_type)
            if safety_reminder:
                friendly_instruction += f"。{safety_reminder}"

            # 添加到达提醒
            if assistant_action == "到达目的地":
                friendly_instruction += "。恭喜您已到达目的地"

            return friendly_instruction

        except Exception as e:
            logger.error(f"生成盲人友好指令失败: {e}")
            return instruction

    def _get_walk_type_description(self, walk_type: int) -> str:
        """获取步行类型描述"""
        walk_type_map = {
            0: "",  # 普通道路
            1: "通过人行横道",
            3: "通过地下通道",
            4: "通过过街天桥",
            5: "通过地铁通道",
            6: "穿过公园",
            7: "穿过广场",
            8: "使用扶梯",
            9: "使用直梯",
            10: "使用索道",
            11: "通过空中通道",
            12: "穿越建筑物",
            13: "使用行人通道",
            14: "乘坐游船",
            15: "乘坐观光车",
            16: "使用滑道",
            18: "走扩路",
            19: "使用道路附属连接线",
            20: "上下阶梯",
            21: "走斜坡",
            22: "过桥",
            23: "穿隧道",
            30: "乘轮渡"
        }
        return walk_type_map.get(walk_type, "")

    def _get_walk_type_accessibility_info(self, walk_type: int) -> str:
        """获取步行类型的无障碍信息"""
        accessibility_map = {
            0: "普通人行道",
            1: "人行横道，注意车辆",
            3: "地下通道，可能有台阶",
            4: "过街天桥，需要上下楼梯",
            5: "地铁通道，通常有无障碍设施",
            6: "公园路径，路面可能不平",
            7: "广场，空间开阔",
            8: "扶梯，注意安全",
            9: "电梯，无障碍友好",
            20: "台阶，需要小心",
            21: "斜坡，注意坡度",
            22: "桥梁，可能有风",
            23: "隧道，注意通风"
        }
        return accessibility_map.get(walk_type, "")

    def _get_safety_reminder(self, action: str, walk_type: int) -> str:
        """获取安全提醒"""
        reminders = []

        # 基于动作的提醒
        if "转" in action:
            reminders.append("转弯时请注意周围环境")

        # 基于路段类型的提醒
        if walk_type == 1:  # 人行横道
            reminders.append("过马路时请注意车辆，确保安全")
        elif walk_type == 4:  # 过街天桥
            reminders.append("上下天桥时请扶好扶手")
        elif walk_type == 8:  # 扶梯
            reminders.append("乘坐扶梯时请站稳扶好")
        elif walk_type == 20:  # 台阶
            reminders.append("上下台阶时请小心脚下")
        elif walk_type == 21:  # 斜坡
            reminders.append("斜坡路段请注意脚下安全")

        return "，".join(reminders)

    def _try_v3_walking_api(self, origin, destination, origin_poi_id=None, destination_poi_id=None):
        """尝试使用v3步行路径规划API"""
        try:
            # 构建v3 API请求参数
            params = {
                'origin': f"{origin.longitude},{origin.latitude}",
                'destination': f"{destination.longitude},{destination.latitude}",
                'output': 'json'
            }

            # 添加POI ID（如果有）
            if origin_poi_id:
                params['origin_id'] = origin_poi_id
            if destination_poi_id:
                params['destination_id'] = destination_poi_id

            # 调用v3步行路径规划API
            url = f"{self.base_url}/direction/walking"
            logger.debug(f"v3 API请求: {url}")
            logger.debug(f"请求参数: {params}")

            import requests
            response = requests.get(url, params={**params, 'key': self.api_key}, timeout=15)
            response.raise_for_status()
            data = response.json()

            logger.debug(f"v3 API响应状态: {data.get('status')}")

            if data.get('status') == '1':
                return self._parse_walking_route_data_v3(data)
            else:
                error_info = data.get('info', '未知错误')
                error_code = data.get('infocode', 'N/A')
                logger.warning(f"v3步行路径规划API失败: {error_info} (错误码: {error_code})")
                return None

        except Exception as e:
            logger.error(f"v3步行路径规划请求失败: {e}")
            return None

    def _try_v5_walking_api(self, origin, destination, origin_poi_id=None, destination_poi_id=None):
        """尝试使用v5步行路径规划API"""
        try:
            # 构建v5 API请求参数
            params = {
                'origin': f"{origin.longitude},{origin.latitude}",
                'destination': f"{destination.longitude},{destination.latitude}",
                'output': 'json',
                'isindoor': 0,
                'alternative_route': 1
            }

            # 添加POI ID（如果有）
            if origin_poi_id:
                params['origin_id'] = origin_poi_id
            if destination_poi_id:
                params['destination_id'] = destination_poi_id

            # 调用v5步行路径规划API
            url = f"{self.base_url}/v5/direction/walking"
            logger.debug(f"v5 API请求: {url}")

            import requests
            response = requests.get(url, params={**params, 'key': self.api_key}, timeout=15)
            response.raise_for_status()
            data = response.json()

            logger.debug(f"v5 API响应状态: {data.get('status')}")

            if data.get('status') == '1':
                return self._parse_walking_route_data_v5(data)
            else:
                error_info = data.get('info', '未知错误')
                error_code = data.get('infocode', 'N/A')
                logger.warning(f"v5步行路径规划API失败: {error_info} (错误码: {error_code})")
                return None

        except Exception as e:
            logger.error(f"v5步行路径规划请求失败: {e}")
            return None

    def _parse_walking_route_data_v3(self, data: dict):
        """解析v3步行路径规划API返回的数据"""
        try:
            route = data.get('route', {})
            paths = route.get('paths', [])

            if not paths:
                logger.warning("v3路径数据为空")
                return None

            # 取第一个路径方案
            path = paths[0]

            # 基本信息
            total_distance = int(path.get('distance', 0))
            total_duration = int(path.get('duration', 0))

            # 解析步骤
            steps_data = path.get('steps', [])
            steps = []

            for i, step_data in enumerate(steps_data):
                step = self._parse_walking_step_v3(step_data, i + 1)
                if step:
                    steps.append(step)

            # 创建路径信息
            route_info = RouteInfo(
                origin=None,
                destination=None,
                distance=total_distance,
                duration=total_duration,
                steps=steps,
                polyline=""
            )

            logger.info(f"v3解析完成: {len(steps)}个步骤, 总距离{total_distance}米, 用时{total_duration}秒")
            return route_info

        except Exception as e:
            logger.error(f"解析v3步行路径数据失败: {e}")
            return None

    def _parse_walking_step_v3(self, step_data: dict, step_number: int):
        """解析v3单个步行步骤"""
        try:
            instruction = step_data.get('instruction', '')
            road = step_data.get('road', '')
            distance = int(step_data.get('distance', 0))
            duration = int(step_data.get('duration', 0))
            orientation = step_data.get('orientation', '')
            action = step_data.get('action', '')
            assistant_action = step_data.get('assistant_action', '')
            walk_type = int(step_data.get('walk_type', 0))
            polyline = step_data.get('polyline', '')

            # 处理可能的数组类型字段
            if isinstance(road, list):
                road = road[0] if road else ''
            if isinstance(orientation, list):
                orientation = orientation[0] if orientation else ''
            if isinstance(action, list):
                action = action[0] if action else ''
            if isinstance(assistant_action, list):
                assistant_action = assistant_action[0] if assistant_action else ''

            # 生成盲人友好的指令
            blind_friendly_instruction = self._generate_blind_friendly_instruction_v3(
                instruction, road, action, assistant_action, walk_type, distance, orientation
            )

            step = RouteStep(
                instruction=blind_friendly_instruction,
                distance=distance,
                duration=duration,
                polyline=polyline,
                action=action,
                orientation=orientation,
                road_name=road,
                assistant_action=assistant_action,
                accessibility_note=self._get_walk_type_accessibility_info(walk_type)
            )

            logger.debug(f"v3步骤{step_number}: {blind_friendly_instruction}")
            return step

        except Exception as e:
            logger.error(f"解析v3步行步骤失败: {e}")
            return None

    def _generate_blind_friendly_instruction_v3(self, instruction: str, road: str, action: str,
                                               assistant_action: str, walk_type: int, distance: int, orientation: str) -> str:
        """为v3 API生成盲人友好的导航指令"""
        try:
            # 基础指令
            friendly_instruction = instruction

            # 添加距离描述
            if distance > 0:
                if distance < 50:
                    distance_desc = f"短距离{distance}米"
                elif distance < 200:
                    distance_desc = f"中等距离{distance}米"
                else:
                    distance_desc = f"较长距离{distance}米"

                # 替换原始距离描述
                friendly_instruction = friendly_instruction.replace(f"{distance}米", distance_desc)

            # 添加道路信息
            if road:
                if road not in friendly_instruction:
                    friendly_instruction += f"，沿{road}"

            # 添加方向信息
            if orientation:
                direction_map = {
                    '东': '向东',
                    '南': '向南',
                    '西': '向西',
                    '北': '向北',
                    '东南': '向东南',
                    '西南': '向西南',
                    '东北': '向东北',
                    '西北': '向西北'
                }
                if orientation in direction_map and direction_map[orientation] not in friendly_instruction:
                    friendly_instruction += f"，{direction_map[orientation]}"

            # 添加特殊路段信息
            walk_type_info = self._get_walk_type_description(walk_type)
            if walk_type_info:
                friendly_instruction += f"，{walk_type_info}"

            # 添加安全提醒
            safety_reminder = self._get_safety_reminder_v3(action, walk_type, distance)
            if safety_reminder:
                friendly_instruction += f"。{safety_reminder}"

            # 添加到达提醒
            if "到达目的地" in instruction or assistant_action == "到达目的地":
                friendly_instruction += "。恭喜您已到达目的地"

            return friendly_instruction

        except Exception as e:
            logger.error(f"生成v3盲人友好指令失败: {e}")
            return instruction

    def _get_safety_reminder_v3(self, action: str, walk_type: int, distance: int) -> str:
        """获取v3安全提醒"""
        reminders = []

        # 基于动作的提醒
        if action and "转" in action:
            reminders.append("转弯时请注意周围环境")

        # 基于路段类型的提醒
        if walk_type == 1:  # 人行横道
            reminders.append("过马路时请注意车辆，确保安全")
        elif walk_type == 4:  # 过街天桥
            reminders.append("上下天桥时请扶好扶手")
        elif walk_type == 8:  # 扶梯
            reminders.append("乘坐扶梯时请站稳扶好")
        elif walk_type == 20:  # 台阶
            reminders.append("上下台阶时请小心脚下")
        elif walk_type == 21:  # 斜坡
            reminders.append("斜坡路段请注意脚下安全")

        # 基于距离的提醒
        if distance > 500:
            reminders.append("较长路段，请注意休息")

        return "，".join(reminders)

    # 将新方法绑定到类
    EnhancedRoutingService._generate_blind_friendly_instruction = _generate_blind_friendly_instruction
    EnhancedRoutingService._get_walk_type_description = _get_walk_type_description
    EnhancedRoutingService._get_walk_type_accessibility_info = _get_walk_type_accessibility_info
    EnhancedRoutingService._get_safety_reminder = _get_safety_reminder
    EnhancedRoutingService._try_v3_walking_api = _try_v3_walking_api
    EnhancedRoutingService._try_v5_walking_api = _try_v5_walking_api
    EnhancedRoutingService._parse_walking_route_data_v3 = _parse_walking_route_data_v3
    EnhancedRoutingService._parse_walking_step_v3 = _parse_walking_step_v3
    EnhancedRoutingService._generate_blind_friendly_instruction_v3 = _generate_blind_friendly_instruction_v3
    EnhancedRoutingService._get_safety_reminder_v3 = _get_safety_reminder_v3

    def _create_simple_route_estimation(self, origin, destination):
        """创建简单的路径估算（当API都失败时的备选方案）"""
        try:
            import math

            # 计算直线距离
            def calculate_distance(lat1, lon1, lat2, lon2):
                """计算两点间距离（米）"""
                R = 6371000  # 地球半径（米）
                lat1_rad = math.radians(lat1)
                lat2_rad = math.radians(lat2)
                delta_lat = math.radians(lat2 - lat1)
                delta_lon = math.radians(lon2 - lon1)

                a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance = R * c
                return distance

            straight_distance = calculate_distance(
                origin.latitude, origin.longitude,
                destination.latitude, destination.longitude
            )

            # 估算步行距离（通常比直线距离长20-30%）
            estimated_distance = int(straight_distance * 1.25)

            # 估算步行时间（按照平均步行速度1.25米/秒）
            estimated_duration = int(estimated_distance / 1.25)

            # 计算大致方向
            lat_diff = destination.latitude - origin.latitude
            lon_diff = destination.longitude - origin.longitude

            if abs(lat_diff) > abs(lon_diff):
                direction = "向北" if lat_diff > 0 else "向南"
            else:
                direction = "向东" if lon_diff > 0 else "向西"

            # 创建简单的步骤
            steps = []

            # 第一步：出发
            start_step = RouteStep(
                instruction=f"从当前位置出发，{direction}行进",
                distance=estimated_distance // 3,
                duration=estimated_duration // 3,
                polyline="",
                action="直行",
                orientation=direction.replace("向", ""),
                road_name="",
                assistant_action="",
                accessibility_note="普通人行道"
            )
            steps.append(start_step)

            # 第二步：继续前进
            continue_step = RouteStep(
                instruction=f"继续{direction}步行{estimated_distance//3}米",
                distance=estimated_distance // 3,
                duration=estimated_duration // 3,
                polyline="",
                action="直行",
                orientation=direction.replace("向", ""),
                road_name="",
                assistant_action="",
                accessibility_note="普通人行道"
            )
            steps.append(continue_step)

            # 第三步：到达
            arrive_step = RouteStep(
                instruction=f"继续前行{estimated_distance - (estimated_distance//3)*2}米到达目的地",
                distance=estimated_distance - (estimated_distance//3)*2,
                duration=estimated_duration - (estimated_duration//3)*2,
                polyline="",
                action="",
                orientation="",
                road_name="",
                assistant_action="到达目的地",
                accessibility_note=""
            )
            steps.append(arrive_step)

            # 创建路径信息
            route_info = RouteInfo(
                origin=None,
                destination=None,
                distance=estimated_distance,
                duration=estimated_duration,
                steps=steps,
                polyline=""
            )

            logger.info(f"简单路径估算完成: 距离{estimated_distance}米, 用时{estimated_duration}秒")
            logger.warning("⚠️ 这是基于直线距离的估算，实际路径可能有所不同")

            return route_info

        except Exception as e:
            logger.error(f"创建简单路径估算失败: {e}")
            return None

    # 绑定新方法
    EnhancedRoutingService._create_simple_route_estimation = _create_simple_route_estimation

# 调用函数添加方法
add_enhanced_walking_methods()
