#!/usr/bin/env python3
"""
BlindStar 导航上下文管理器
管理GPS位置、路径规划和导航状态
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

@dataclass
class GPSLocation:
    """GPS位置信息"""
    latitude: float        # 纬度
    longitude: float       # 经度
    altitude: Optional[float] = None  # 海拔
    accuracy: Optional[float] = None  # 精度(米)
    timestamp: Optional[float] = field(default_factory=time.time)  # 时间戳

    def distance_to(self, other: 'GPSLocation') -> float:
        """计算到另一个位置的距离(米)"""
        import math
        
        # 使用Haversine公式计算距离
        R = 6371000  # 地球半径(米)
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    def bearing_to(self, other: 'GPSLocation') -> float:
        """计算到另一个位置的方位角(度)"""
        import math
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # 转换为0-360度
        return (bearing_deg + 360) % 360

class NavigationMode(Enum):
    """导航模式"""
    ASSIST = "assist"      # 辅助模式：仅提供危险提醒
    GUIDE = "guide"        # 引导模式：提供方向指引
    FULL = "full"          # 完全导航：完整路径规划

class NavigationState(Enum):
    """导航状态"""
    IDLE = "idle"          # 空闲
    PLANNING = "planning"  # 路径规划中
    NAVIGATING = "navigating"  # 导航中
    REROUTING = "rerouting"    # 重新规划
    ARRIVED = "arrived"    # 已到达

@dataclass
class NavigationContext:
    """导航上下文信息"""
    current_location: Optional[GPSLocation] = None
    destination: Optional[GPSLocation] = None
    route_points: List[GPSLocation] = field(default_factory=list)
    current_instruction: Optional[str] = None
    distance_to_destination: Optional[float] = None
    estimated_time: Optional[int] = None
    navigation_mode: NavigationMode = NavigationMode.ASSIST
    navigation_state: NavigationState = NavigationState.IDLE
    
    # 路径跟踪
    current_route_index: int = 0
    next_waypoint: Optional[GPSLocation] = None
    distance_to_next_waypoint: Optional[float] = None
    
    # 导航历史
    location_history: List[GPSLocation] = field(default_factory=list)
    instruction_history: List[str] = field(default_factory=list)

class NavigationContextManager:
    """导航上下文管理器"""

    def __init__(self, mode: NavigationMode = NavigationMode.ASSIST):
        self.context = NavigationContext(navigation_mode=mode)
        self.logger = logging.getLogger(__name__)

        # 配置参数
        self.location_update_threshold = 5.0  # 位置更新阈值(米)
        self.arrival_threshold = 10.0         # 到达阈值(米)
        self.waypoint_threshold = 20.0        # 路径点阈值(米)
        self.max_history_size = 100           # 历史记录最大长度

        # 导航服务集成
        self.map_service = None               # 地图服务实例
        self.poi_service = None               # POI查询服务实例

        # 性能统计
        self.stats = {
            'location_updates': 0,
            'navigation_instructions': 0,
            'route_recalculations': 0,
            'last_update_time': time.time()
        }

        self.logger.info(f"[导航] 初始化导航上下文管理器，模式: {mode.value}")
    
    def update_location(self, location: GPSLocation) -> bool:
        """更新当前位置"""
        try:
            # 检查位置有效性
            if not self._is_valid_location(location):
                self.logger.warning(f"[导航] 无效的GPS位置: {location}")
                return False

            # 检查是否需要更新
            if self.context.current_location:
                distance = location.distance_to(self.context.current_location)
                if distance < self.location_update_threshold:
                    return False  # 位置变化不大，不更新

            # 更新位置
            old_location = self.context.current_location
            self.context.current_location = location

            # 添加到历史记录
            self.context.location_history.append(location)
            if len(self.context.location_history) > self.max_history_size:
                self.context.location_history.pop(0)

            # 更新统计信息
            self.stats['location_updates'] += 1
            self.stats['last_update_time'] = time.time()

            self.logger.info(f"[导航] 位置更新: ({location.latitude:.6f}, {location.longitude:.6f})")

            # 更新导航状态
            self._update_navigation_state()

            # 检查是否需要重新规划路径
            if self._should_recalculate_route(old_location, location):
                self._recalculate_route()

            return True

        except Exception as e:
            self.logger.error(f"[导航] 位置更新失败: {e}")
            return False
    
    def set_destination(self, destination: GPSLocation) -> bool:
        """设置目的地"""
        try:
            if not self._is_valid_location(destination):
                self.logger.warning(f"[导航] 无效的目的地: {destination}")
                return False
            
            self.context.destination = destination
            self.context.navigation_state = NavigationState.PLANNING
            
            # 计算距离
            if self.context.current_location:
                self.context.distance_to_destination = \
                    self.context.current_location.distance_to(destination)
            
            self.logger.info(f"[导航] 设置目的地: ({destination.latitude:.6f}, {destination.longitude:.6f})")
            
            # 如果是完全导航模式，开始路径规划
            if self.context.navigation_mode == NavigationMode.FULL:
                self._plan_route()
            
            return True
            
        except Exception as e:
            self.logger.error(f"[导航] 设置目的地失败: {e}")
            return False
    
    def get_current_instruction(self) -> Optional[str]:
        """获取当前导航指令"""
        if self.context.navigation_state == NavigationState.IDLE:
            return None
        
        if self.context.navigation_mode == NavigationMode.ASSIST:
            return self._generate_assist_instruction()
        elif self.context.navigation_mode == NavigationMode.GUIDE:
            return self._generate_guide_instruction()
        elif self.context.navigation_mode == NavigationMode.FULL:
            return self._generate_full_instruction()
        
        return None
    
    def get_context(self) -> NavigationContext:
        """获取当前导航上下文"""
        return self.context
    
    def _is_valid_location(self, location: GPSLocation) -> bool:
        """检查GPS位置是否有效"""
        if location is None:
            return False
        
        # 检查纬度范围
        if not (-90 <= location.latitude <= 90):
            return False
        
        # 检查经度范围
        if not (-180 <= location.longitude <= 180):
            return False
        
        # 检查精度
        if location.accuracy and location.accuracy > 50:  # 精度超过50米认为不可靠
            return False
        
        return True
    
    def _update_navigation_state(self):
        """更新导航状态"""
        if not self.context.current_location or not self.context.destination:
            return
        
        # 计算到目的地的距离
        distance = self.context.current_location.distance_to(self.context.destination)
        self.context.distance_to_destination = distance
        
        # 检查是否到达
        if distance <= self.arrival_threshold:
            if self.context.navigation_state != NavigationState.ARRIVED:
                self.context.navigation_state = NavigationState.ARRIVED
                self.logger.info("[导航] 已到达目的地")
        elif self.context.navigation_state == NavigationState.PLANNING:
            self.context.navigation_state = NavigationState.NAVIGATING
            self.logger.info("[导航] 开始导航")
        
        # 更新路径点状态
        if self.context.route_points and self.context.current_route_index < len(self.context.route_points):
            next_point = self.context.route_points[self.context.current_route_index]
            distance_to_next = self.context.current_location.distance_to(next_point)
            
            if distance_to_next <= self.waypoint_threshold:
                # 到达当前路径点，移动到下一个
                self.context.current_route_index += 1
                self.logger.info(f"[导航] 到达路径点 {self.context.current_route_index}")
    
    def _plan_route(self):
        """规划路径（简化版本）"""
        if not self.context.current_location or not self.context.destination:
            return
        
        # 简化实现：直接连接起点和终点
        # 实际实现中应该调用地图服务API
        self.context.route_points = [
            self.context.current_location,
            self.context.destination
        ]
        self.context.current_route_index = 0
        
        self.logger.info("[导航] 路径规划完成")
    
    def _generate_assist_instruction(self) -> Optional[str]:
        """生成辅助模式指令"""
        if not self.context.destination or not self.context.current_location:
            return None
        
        distance = self.context.distance_to_destination
        if distance and distance > 1000:
            return f"目的地距离{distance/1000:.1f}公里"
        elif distance:
            return f"目的地距离{distance:.0f}米"
        
        return None
    
    def _generate_guide_instruction(self) -> Optional[str]:
        """生成引导模式指令"""
        if not self.context.destination or not self.context.current_location:
            return None
        
        bearing = self.context.current_location.bearing_to(self.context.destination)
        distance = self.context.distance_to_destination
        
        # 将方位角转换为方向描述
        direction = self._bearing_to_direction(bearing)
        
        if distance and distance > 100:
            return f"向{direction}方向前进{distance:.0f}米"
        elif distance:
            return f"目的地就在{direction}方向{distance:.0f}米处"
        
        return None
    
    def _generate_full_instruction(self) -> Optional[str]:
        """生成完全导航模式指令"""
        # 完整的导航指令生成
        # 实际实现中应该基于详细的路径规划
        return self._generate_guide_instruction()
    
    def _bearing_to_direction(self, bearing: float) -> str:
        """将方位角转换为方向描述"""
        directions = [
            "北", "东北", "东", "东南",
            "南", "西南", "西", "西北"
        ]

        # 将360度分为8个方向
        index = int((bearing + 22.5) / 45) % 8
        return directions[index]

    def set_map_service(self, map_service) -> bool:
        """
        设置地图服务

        Args:
            map_service: 地图服务实例

        Returns:
            是否设置成功
        """
        try:
            if map_service is None:
                self.logger.warning("[导航] 地图服务实例为空")
                return False

            self.map_service = map_service
            self.logger.info("[导航] 地图服务已设置")
            return True

        except Exception as e:
            self.logger.error(f"[导航] 设置地图服务失败: {e}")
            return False

    def set_poi_service(self, poi_service) -> bool:
        """
        设置POI查询服务

        Args:
            poi_service: POI查询服务实例

        Returns:
            是否设置成功
        """
        try:
            if poi_service is None:
                self.logger.warning("[导航] POI服务实例为空")
                return False

            self.poi_service = poi_service
            self.logger.info("[导航] POI服务已设置")
            return True

        except Exception as e:
            self.logger.error(f"[导航] 设置POI服务失败: {e}")
            return False

    def start_navigation(self, destination: GPSLocation) -> bool:
        """开始导航"""
        if self.set_destination(destination):
            self.context.navigation_state = NavigationState.NAVIGATING
            self.logger.info("[导航] 导航已开始")
            return True
        return False

    def stop_navigation(self):
        """停止导航"""
        self.context.navigation_state = NavigationState.IDLE
        self.context.destination = None
        self.context.route_points.clear()
        self.context.current_route_index = 0
        self.logger.info("[导航] 导航已停止")

    def pause_navigation(self):
        """暂停导航"""
        if self.context.navigation_state == NavigationState.NAVIGATING:
            self.context.navigation_state = NavigationState.IDLE
            self.logger.info("[导航] 导航已暂停")

    def resume_navigation(self):
        """恢复导航"""
        if self.context.destination and self.context.navigation_state == NavigationState.IDLE:
            self.context.navigation_state = NavigationState.NAVIGATING
            self.logger.info("[导航] 导航已恢复")

    def get_navigation_stats(self) -> Dict[str, Any]:
        """获取导航统计信息"""
        return {
            **self.stats,
            'current_state': self.context.navigation_state.value,
            'current_mode': self.context.navigation_mode.value,
            'has_destination': self.context.destination is not None,
            'distance_to_destination': self.context.distance_to_destination,
            'route_progress': self._get_route_progress()
        }

    def _should_recalculate_route(self, old_location: Optional[GPSLocation],
                                 new_location: GPSLocation) -> bool:
        """判断是否需要重新规划路径"""
        if not old_location or not self.context.destination:
            return False

        # 如果偏离路径太远，需要重新规划
        if self.context.route_points and len(self.context.route_points) > 1:
            # 简化判断：如果距离路径点太远
            if (self.context.current_route_index < len(self.context.route_points) and
                new_location.distance_to(self.context.route_points[self.context.current_route_index]) > 100):
                return True

        return False

    def _recalculate_route(self):
        """重新规划路径"""
        if self.map_service and self.context.current_location and self.context.destination:
            try:
                # 调用地图服务重新规划路径
                new_route = self.map_service.get_route(
                    self.context.current_location,
                    self.context.destination
                )
                if new_route:
                    self.context.route_points = new_route
                    self.context.current_route_index = 0
                    self.context.navigation_state = NavigationState.REROUTING
                    self.stats['route_recalculations'] += 1
                    self.logger.info("[导航] 路径重新规划完成")
            except Exception as e:
                self.logger.error(f"[导航] 路径重新规划失败: {e}")
        else:
            # 使用简化的重新规划
            self._plan_route()

    def _get_route_progress(self) -> float:
        """获取路径进度(0-1)"""
        if not self.context.route_points or not self.context.current_location:
            return 0.0

        total_points = len(self.context.route_points)
        if total_points <= 1:
            return 1.0 if self.context.navigation_state == NavigationState.ARRIVED else 0.0

        return min(self.context.current_route_index / (total_points - 1), 1.0)
