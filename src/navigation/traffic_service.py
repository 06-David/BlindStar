#!/usr/bin/env python3
"""
实时交通信息服务
基于高德地图交通态势查询API，提供实时路况信息和交通事件提醒
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


class TrafficStatus(Enum):
    """交通状态"""
    UNKNOWN = 0      # 未知
    SMOOTH = 1       # 畅通
    SLOW = 2         # 缓慢
    CONGESTED = 3    # 拥堵
    SEVERE = 4       # 严重拥堵


class IncidentType(Enum):
    """交通事件类型"""
    ACCIDENT = "accident"        # 交通事故
    CONSTRUCTION = "construction" # 道路施工
    CLOSURE = "closure"          # 道路封闭
    EVENT = "event"              # 特殊事件
    WEATHER = "weather"          # 天气影响
    OTHER = "other"              # 其他


@dataclass
class TrafficIncident:
    """交通事件"""
    id: str
    type: IncidentType
    title: str
    description: str
    location: LocationInfo
    start_time: str
    end_time: str = ""
    severity: int = 1  # 严重程度 1-5
    affected_roads: List[str] = field(default_factory=list)
    
    # 盲人友好信息
    audio_description: str = ""      # 语音描述
    safety_warning: str = ""         # 安全警告
    alternative_route: str = ""      # 替代路线建议
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'type': self.type.value,
            'title': self.title,
            'description': self.description,
            'location': self.location.to_dict() if self.location else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'severity': self.severity,
            'affected_roads': self.affected_roads,
            'audio_description': self.audio_description,
            'safety_warning': self.safety_warning,
            'alternative_route': self.alternative_route
        }


@dataclass
class TrafficInfo:
    """交通信息"""
    road_name: str
    status: TrafficStatus
    speed: float = 0.0           # 平均速度 km/h
    congestion_distance: float = 0.0  # 拥堵距离 km
    travel_time: int = 0         # 通行时间 秒
    description: str = ""        # 状态描述
    
    # 盲人友好信息
    blind_friendly_description: str = ""  # 盲人友好描述
    safety_level: int = 1        # 安全等级 1-5
    recommended: bool = True     # 是否推荐
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'road_name': self.road_name,
            'status': self.status.value,
            'speed': self.speed,
            'congestion_distance': self.congestion_distance,
            'travel_time': self.travel_time,
            'description': self.description,
            'blind_friendly_description': self.blind_friendly_description,
            'safety_level': self.safety_level,
            'recommended': self.recommended
        }


@dataclass
class TrafficReport:
    """交通报告"""
    area_name: str
    update_time: str
    overall_status: TrafficStatus
    traffic_info: List[TrafficInfo] = field(default_factory=list)
    incidents: List[TrafficIncident] = field(default_factory=list)
    
    # 盲人出行建议
    travel_advice: str = ""
    best_travel_time: str = ""
    avoid_areas: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'area_name': self.area_name,
            'update_time': self.update_time,
            'overall_status': self.overall_status.value,
            'traffic_info': [info.to_dict() for info in self.traffic_info],
            'incidents': [incident.to_dict() for incident in self.incidents],
            'travel_advice': self.travel_advice,
            'best_travel_time': self.best_travel_time,
            'avoid_areas': self.avoid_areas
        }


class TrafficService:
    """实时交通信息服务"""
    
    def __init__(self, api_key: str = "717d9a827c0ac3521932d3ae59aebbfe"):
        """
        初始化交通信息服务
        
        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3"
        
        # 获取相关服务
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
        
        # 交通信息缓存
        self._traffic_cache = {}
        self._cache_lock = threading.RLock()
        self.cache_ttl = 60  # 1分钟缓存（交通信息变化快）
        
        # 交通状态映射
        self.status_mapping = {
            "0": TrafficStatus.UNKNOWN,
            "1": TrafficStatus.SMOOTH,
            "2": TrafficStatus.SLOW,
            "3": TrafficStatus.CONGESTED,
            "4": TrafficStatus.SEVERE
        }
        
        # 盲人友好的交通状态描述
        self.blind_friendly_status = {
            TrafficStatus.SMOOTH: "道路畅通，通行顺畅",
            TrafficStatus.SLOW: "道路缓慢，请注意安全",
            TrafficStatus.CONGESTED: "道路拥堵，建议避开或等待",
            TrafficStatus.SEVERE: "严重拥堵，强烈建议选择其他路线",
            TrafficStatus.UNKNOWN: "交通状况未知，请谨慎通行"
        }
        
        logger.info("✅ 实时交通信息服务初始化完成")
    
    def get_traffic_status(self,
                          rectangle: str = None,
                          location: str = None,
                          radius: int = 1000) -> Optional[TrafficReport]:
        """
        获取区域交通状态（简化版，避免商业化API）

        Args:
            rectangle: 矩形区域坐标
            location: 中心点坐标
            radius: 查询半径（米）

        Returns:
            TrafficReport: 交通报告
        """
        try:
            # 获取位置信息
            if not location:
                if self.location_service:
                    current_location = self.location_service.get_current_location()
                    location = current_location.to_coordinates()
                    area_name = current_location.city or "当前区域"
                else:
                    location = "116.397428,39.99923"
                    area_name = "默认区域"
            else:
                area_name = "查询区域"

            # 检查缓存
            cache_key = f"traffic_simple_{location}_{radius}"
            cached_report = self._get_cached_traffic(cache_key)
            if cached_report:
                logger.debug("🎯 使用缓存的交通信息")
                return cached_report

            # 创建简化的交通报告
            report = self._create_simple_traffic_report(area_name, location)

            # 缓存结果
            self._cache_traffic(cache_key, report)

            return report

        except Exception as e:
            logger.error(f"获取交通状态失败: {e}")
            return self._create_simple_traffic_report("未知区域", location or "116.397428,39.99923")
    
    def get_traffic_incidents(self,
                            rectangle: str = None,
                            location: str = None,
                            radius: int = 2000) -> List[TrafficIncident]:
        """
        获取交通事件（简化版，避免商业化API）

        Args:
            rectangle: 矩形区域坐标
            location: 中心点坐标
            radius: 查询半径（米）

        Returns:
            交通事件列表（简化版返回空列表）
        """
        try:
            # 简化版：不调用商业化API，返回基础提醒
            logger.info("交通事件查询已简化，如需详细事件信息请使用商业版API")

            # 可以根据时间段返回一些通用提醒
            current_hour = time.localtime().tm_hour
            incidents = []

            # 在高峰时段提供通用提醒
            if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
                if location and self.location_service:
                    current_location = self.location_service.get_current_location()
                    location_info = LocationInfo(
                        longitude=float(location.split(',')[0]),
                        latitude=float(location.split(',')[1]),
                        accuracy=1000.0,
                        address=current_location.address,
                        city=current_location.city,
                        district="",
                        province=current_location.province,
                        location_type="traffic_reminder"
                    )

                    incident = TrafficIncident(
                        id="peak_hour_reminder",
                        type=IncidentType.OTHER,
                        title="高峰时段提醒",
                        description="当前为交通高峰时段，建议选择公共交通或预留充足时间",
                        location=location_info,
                        start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        severity=2,
                        audio_description="高峰时段提醒：建议选择公共交通",
                        safety_warning="当前为高峰时段，请注意出行安全",
                        alternative_route="建议使用公共交通或延后出行"
                    )
                    incidents.append(incident)

            return incidents

        except Exception as e:
            logger.error(f"获取交通事件失败: {e}")
            return []
    
    def _build_rectangle_from_location(self, location: str, radius: int) -> str:
        """根据中心点和半径构建矩形区域"""
        try:
            coords = location.split(',')
            if len(coords) < 2:
                return ""
            
            lon, lat = float(coords[0]), float(coords[1])
            
            # 简单的矩形计算（实际应该考虑地球曲率）
            # 1度经度约111km，1度纬度约111km
            radius_km = radius / 1000.0
            delta_lon = radius_km / (111.0 * abs(lat) / 90.0 + 1)
            delta_lat = radius_km / 111.0
            
            # 左下角和右上角坐标
            left_bottom = f"{lon - delta_lon},{lat - delta_lat}"
            right_top = f"{lon + delta_lon},{lat + delta_lat}"
            
            return f"{left_bottom};{right_top}"
            
        except Exception as e:
            logger.error(f"构建矩形区域失败: {e}")
            return ""
    
    def _call_traffic_api(self, rectangle: str) -> Optional[Dict[str, Any]]:
        """调用交通态势API"""
        try:
            params = {
                "rectangle": rectangle,
                "extensions": "all"
            }
            
            if self.client:
                success, data, error = self.client.request("traffic_status", **params)
            else:
                url = f"{self.base_url}/traffic/status/rectangle"
                params["key"] = self.api_key
                params["output"] = "json"
                
                response = requests.get(url, params=params, timeout=15, verify=False)
                response.raise_for_status()
                data = response.json()
                success = data.get('status') == '1'
            
            if success and data:
                return data
            else:
                logger.error(f"交通态势API调用失败: {data.get('info', '未知错误') if data else error}")
                return None
                
        except Exception as e:
            logger.error(f"交通态势API调用异常: {e}")
            return None
    
    def _call_incident_api(self, rectangle: str) -> Optional[Dict[str, Any]]:
        """调用交通事件API"""
        try:
            params = {
                "rectangle": rectangle,
                "extensions": "all"
            }
            
            if self.client:
                success, data, error = self.client.request(
                    "traffic_incidents",
                    url=f"{self.base_url}/traffic/events/rectangle",
                    **params
                )
            else:
                url = f"{self.base_url}/traffic/events/rectangle"
                params["key"] = self.api_key
                params["output"] = "json"
                
                response = requests.get(url, params=params, timeout=15, verify=False)
                response.raise_for_status()
                data = response.json()
                success = data.get('status') == '1'
            
            if success and data:
                return data
            else:
                logger.debug(f"交通事件API调用失败: {data.get('info', '未知错误') if data else error}")
                return None
                
        except Exception as e:
            logger.debug(f"交通事件API调用异常: {e}")
            return None
    
    def _parse_traffic_data(self, traffic_data: Dict[str, Any], rectangle: str) -> Optional[TrafficReport]:
        """解析交通数据"""
        try:
            trafficinfo = traffic_data.get('trafficinfo', {})
            roads = trafficinfo.get('roads', [])
            
            if not roads:
                return None
            
            # 创建交通报告
            report = TrafficReport(
                area_name="查询区域",
                update_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                overall_status=TrafficStatus.SMOOTH
            )
            
            total_status_value = 0
            road_count = 0
            
            for road_data in roads:
                try:
                    status_value = int(road_data.get('status', '0'))
                    status = self.status_mapping.get(str(status_value), TrafficStatus.UNKNOWN)
                    
                    traffic_info = TrafficInfo(
                        road_name=road_data.get('name', '未知道路'),
                        status=status,
                        speed=float(road_data.get('speed', 0)),
                        congestion_distance=float(road_data.get('congestion_distance', 0)),
                        travel_time=int(road_data.get('travel_time', 0)),
                        description=road_data.get('description', '')
                    )
                    
                    report.traffic_info.append(traffic_info)
                    
                    total_status_value += status_value
                    road_count += 1
                    
                except Exception as e:
                    logger.debug(f"解析道路交通信息失败: {e}")
                    continue
            
            # 计算整体状态
            if road_count > 0:
                avg_status = total_status_value / road_count
                if avg_status >= 3.5:
                    report.overall_status = TrafficStatus.SEVERE
                elif avg_status >= 2.5:
                    report.overall_status = TrafficStatus.CONGESTED
                elif avg_status >= 1.5:
                    report.overall_status = TrafficStatus.SLOW
                else:
                    report.overall_status = TrafficStatus.SMOOTH
            
            return report
            
        except Exception as e:
            logger.error(f"解析交通数据失败: {e}")
            return None
    
    def _parse_incident_data(self, incident_data: Dict[str, Any]) -> List[TrafficIncident]:
        """解析交通事件数据"""
        try:
            events = incident_data.get('events', [])
            incidents = []
            
            for event_data in events:
                try:
                    # 解析位置信息
                    location_str = event_data.get('location', '')
                    if location_str:
                        coords = location_str.split(',')
                        if len(coords) >= 2:
                            location = LocationInfo(
                                longitude=float(coords[0]),
                                latitude=float(coords[1]),
                                accuracy=100.0,
                                address=event_data.get('address', ''),
                                city="",
                                district="",
                                province="",
                                location_type="incident"
                            )
                        else:
                            continue
                    else:
                        continue
                    
                    # 确定事件类型
                    event_type_str = event_data.get('type', 'other').lower()
                    if 'accident' in event_type_str or '事故' in event_type_str:
                        event_type = IncidentType.ACCIDENT
                    elif 'construction' in event_type_str or '施工' in event_type_str:
                        event_type = IncidentType.CONSTRUCTION
                    elif 'closure' in event_type_str or '封闭' in event_type_str:
                        event_type = IncidentType.CLOSURE
                    elif 'weather' in event_type_str or '天气' in event_type_str:
                        event_type = IncidentType.WEATHER
                    else:
                        event_type = IncidentType.OTHER
                    
                    incident = TrafficIncident(
                        id=event_data.get('id', ''),
                        type=event_type,
                        title=event_data.get('title', ''),
                        description=event_data.get('description', ''),
                        location=location,
                        start_time=event_data.get('start_time', ''),
                        end_time=event_data.get('end_time', ''),
                        severity=int(event_data.get('severity', 1)),
                        affected_roads=event_data.get('affected_roads', [])
                    )
                    
                    incidents.append(incident)
                    
                except Exception as e:
                    logger.debug(f"解析交通事件失败: {e}")
                    continue
            
            return incidents
            
        except Exception as e:
            logger.error(f"解析交通事件数据失败: {e}")
            return []
    
    def _enhance_blind_friendly_traffic_info(self, report: TrafficReport):
        """增强盲人友好的交通信息"""
        try:
            # 为每个道路信息添加盲人友好描述
            for traffic_info in report.traffic_info:
                status_desc = self.blind_friendly_status.get(
                    traffic_info.status, 
                    "交通状况未知"
                )
                
                traffic_info.blind_friendly_description = f"{traffic_info.road_name}：{status_desc}"
                
                # 评估安全等级
                if traffic_info.status == TrafficStatus.SEVERE:
                    traffic_info.safety_level = 1
                    traffic_info.recommended = False
                elif traffic_info.status == TrafficStatus.CONGESTED:
                    traffic_info.safety_level = 2
                    traffic_info.recommended = False
                elif traffic_info.status == TrafficStatus.SLOW:
                    traffic_info.safety_level = 3
                    traffic_info.recommended = True
                else:
                    traffic_info.safety_level = 4
                    traffic_info.recommended = True
            
            # 生成整体出行建议
            if report.overall_status == TrafficStatus.SEVERE:
                report.travel_advice = "当前区域交通严重拥堵，建议延后出行或选择其他路线"
                report.best_travel_time = "建议避开高峰时段（7-9点，17-19点）"
            elif report.overall_status == TrafficStatus.CONGESTED:
                report.travel_advice = "当前区域交通拥堵，出行请预留充足时间"
                report.best_travel_time = "建议选择非高峰时段出行"
            elif report.overall_status == TrafficStatus.SLOW:
                report.travel_advice = "当前区域交通缓慢，出行注意安全"
            else:
                report.travel_advice = "当前区域交通畅通，适合出行"
            
            # 识别需要避开的区域
            for traffic_info in report.traffic_info:
                if not traffic_info.recommended:
                    report.avoid_areas.append(traffic_info.road_name)
                    
        except Exception as e:
            logger.error(f"增强盲人友好交通信息失败: {e}")
    
    def _enhance_blind_friendly_incident_info(self, incidents: List[TrafficIncident]):
        """增强盲人友好的交通事件信息"""
        try:
            for incident in incidents:
                # 生成语音描述
                type_desc = {
                    IncidentType.ACCIDENT: "交通事故",
                    IncidentType.CONSTRUCTION: "道路施工",
                    IncidentType.CLOSURE: "道路封闭",
                    IncidentType.WEATHER: "天气影响",
                    IncidentType.EVENT: "特殊事件",
                    IncidentType.OTHER: "其他事件"
                }.get(incident.type, "未知事件")
                
                incident.audio_description = f"{type_desc}：{incident.title}"
                
                # 生成安全警告
                if incident.type == IncidentType.ACCIDENT:
                    incident.safety_warning = "前方发生交通事故，请绕行并注意安全"
                elif incident.type == IncidentType.CONSTRUCTION:
                    incident.safety_warning = "前方道路施工，请减速慢行或选择绕行"
                elif incident.type == IncidentType.CLOSURE:
                    incident.safety_warning = "前方道路封闭，请选择其他路线"
                elif incident.type == IncidentType.WEATHER:
                    incident.safety_warning = "前方受天气影响，请注意安全"
                else:
                    incident.safety_warning = "前方有特殊情况，请谨慎通行"
                
                # 生成替代路线建议
                if incident.affected_roads:
                    incident.alternative_route = f"建议避开{', '.join(incident.affected_roads)}，选择其他道路"
                else:
                    incident.alternative_route = "建议选择其他路线绕行"
                    
        except Exception as e:
            logger.error(f"增强盲人友好事件信息失败: {e}")
    
    def _get_cached_traffic(self, cache_key: str) -> Optional[TrafficReport]:
        """获取缓存的交通信息"""
        with self._cache_lock:
            if cache_key in self._traffic_cache:
                cached_data = self._traffic_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['report']
                else:
                    # 缓存过期，删除
                    del self._traffic_cache[cache_key]
        return None
    
    def _cache_traffic(self, cache_key: str, report: TrafficReport):
        """缓存交通信息"""
        with self._cache_lock:
            self._traffic_cache[cache_key] = {
                'report': report,
                'timestamp': time.time()
            }

    def _create_simple_traffic_report(self, area_name: str, location: str) -> TrafficReport:
        """创建简化的交通报告（避免使用商业化API）"""
        try:
            # 基于时间推断交通状况
            current_hour = time.localtime().tm_hour

            # 根据时间段推断交通状况
            if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
                # 高峰时段
                overall_status = TrafficStatus.SLOW
                travel_advice = "当前为交通高峰时段，建议预留充足时间或选择公共交通"
                best_travel_time = "建议避开高峰时段（7-9点，17-19点）"
            elif 22 <= current_hour or current_hour <= 6:
                # 夜间时段
                overall_status = TrafficStatus.SMOOTH
                travel_advice = "当前为夜间时段，道路通畅，但请注意安全"
                best_travel_time = "夜间出行请特别注意安全"
            else:
                # 平峰时段
                overall_status = TrafficStatus.SMOOTH
                travel_advice = "当前交通状况良好，适合出行"
                best_travel_time = "当前时段适合出行"

            # 创建基础交通信息
            traffic_info = TrafficInfo(
                road_name="主要道路",
                status=overall_status,
                speed=30.0 if overall_status == TrafficStatus.SMOOTH else 15.0,
                description="基于时间段的交通状况评估",
                blind_friendly_description=f"当前时段交通{overall_status.name.lower()}",
                safety_level=4 if overall_status == TrafficStatus.SMOOTH else 3,
                recommended=overall_status != TrafficStatus.SEVERE
            )

            report = TrafficReport(
                area_name=area_name,
                update_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                overall_status=overall_status,
                traffic_info=[traffic_info],
                incidents=[],  # 简化版不提供具体事件
                travel_advice=travel_advice,
                best_travel_time=best_travel_time,
                avoid_areas=[]
            )

            return report

        except Exception as e:
            logger.error(f"创建简化交通报告失败: {e}")
            # 返回默认的良好状态
            return TrafficReport(
                area_name=area_name,
                update_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                overall_status=TrafficStatus.SMOOTH,
                travel_advice="交通状况信息暂时不可用，建议正常出行",
                best_travel_time="建议选择合适的出行时间"
            )


# 全局交通服务实例
_global_traffic_service = None

def get_traffic_service() -> TrafficService:
    """获取全局交通服务实例"""
    global _global_traffic_service
    if _global_traffic_service is None:
        _global_traffic_service = TrafficService()
    return _global_traffic_service
