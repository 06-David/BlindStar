#!/usr/bin/env python3
"""
增强定位服务
基于高德地图API的多种定位方式，提供高精度定位服务
"""

import time
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import threading
import urllib3

try:
    from ..core.amap_client import get_amap_client
    from ..config.poi_codes import get_poi_manager
except ImportError:
    try:
        # 尝试绝对导入
        from src.core.amap_client import get_amap_client
        from src.config.poi_codes import get_poi_manager
    except ImportError:
        # 兼容性处理
        get_amap_client = None
        get_poi_manager = None

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


@dataclass
class LocationInfo:
    """位置信息数据类"""
    longitude: float
    latitude: float
    accuracy: float  # 精度（米）
    address: str
    city: str
    district: str
    province: str
    country: str = "中国"
    location_type: str = "unknown"  # ip, wifi, gps, hardware
    timestamp: float = 0.0
    confidence: float = 0.0  # 置信度 0-1
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_coordinates(self) -> str:
        """转换为坐标字符串格式"""
        return f"{self.longitude},{self.latitude}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'longitude': self.longitude,
            'latitude': self.latitude,
            'accuracy': self.accuracy,
            'address': self.address,
            'city': self.city,
            'district': self.district,
            'province': self.province,
            'country': self.country,
            'location_type': self.location_type,
            'timestamp': self.timestamp,
            'confidence': self.confidence
        }


class EnhancedLocationService:
    """增强定位服务"""
    
    def __init__(self, api_key: str = "717d9a827c0ac3521932d3ae59aebbfe"):
        """
        初始化增强定位服务
        
        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3"
        
        # 获取API客户端
        if get_amap_client:
            try:
                self.client = get_amap_client()
            except Exception as e:
                logger.warning(f"无法初始化API客户端: {e}")
                self.client = None
        else:
            self.client = None
        
        # 位置缓存
        self._location_cache = {}
        self._cache_lock = threading.RLock()
        self.cache_ttl = 300  # 5分钟缓存
        
        # 默认位置（北京天安门）
        self.default_location = LocationInfo(
            longitude=116.397428,
            latitude=39.99923,
            accuracy=1000.0,
            address="天安门广场",
            city="北京市",
            district="东城区",
            province="北京市",
            location_type="default",
            confidence=0.1
        )
        
        logger.info("✅ 增强定位服务初始化完成")
    
    def get_current_location(self, force_refresh: bool = False) -> LocationInfo:
        """
        获取当前位置（多种定位方式融合）
        
        Args:
            force_refresh: 是否强制刷新，忽略缓存
            
        Returns:
            LocationInfo: 位置信息
        """
        # 检查缓存
        if not force_refresh:
            cached_location = self._get_cached_location("current")
            if cached_location:
                logger.debug("🎯 使用缓存的位置信息")
                return cached_location
        
        # 尝试多种定位方式
        location_methods = [
            ("browser_location", self._get_browser_location),
            ("city_location", self._get_city_location),
            ("ip_location", self._get_ip_location),
            ("advanced_ip_location", self._get_advanced_ip_location),
        ]
        
        best_location = None
        best_confidence = 0.0
        
        for method_name, method in location_methods:
            try:
                logger.debug(f"🌐 尝试定位方式: {method_name}")
                location = method()
                
                if location and location.confidence > best_confidence:
                    best_location = location
                    best_confidence = location.confidence
                    logger.info(f"✅ {method_name} 定位成功，置信度: {location.confidence:.2f}")
                
            except Exception as e:
                logger.warning(f"❌ {method_name} 定位失败: {e}")
                continue
        
        # 如果所有方法都失败，提示用户手动输入位置
        if not best_location:
            logger.warning("❌ 所有自动定位方法都失败")
            logger.info("💡 建议: 请使用位置校正功能输入准确地址")
            best_location = self._prompt_manual_location()
        else:
            # 如果定位精度不够，提示用户可以使用手动校正
            if best_location.confidence < 0.8:
                logger.info("💡 提示: 当前定位精度有限，如需更精确定位请使用位置校正功能")

        # 缓存结果
        if best_location:
            self._cache_location("current", best_location)

        return best_location

    def _get_browser_location(self) -> Optional[LocationInfo]:
        """浏览器定位（模拟实现，实际需要在Web环境中使用JavaScript）"""
        try:
            logger.info("🌐 尝试浏览器定位...")

            # 在Python环境中，我们无法直接使用浏览器定位
            # 这里提供一个模拟实现，实际应用中需要在Web前端实现
            logger.debug("浏览器定位需要在Web环境中实现")

            # 可以尝试使用系统位置服务（如果可用）
            return self._try_system_location()

        except Exception as e:
            logger.warning(f"浏览器定位失败: {e}")
            return None

    def _get_city_location(self) -> Optional[LocationInfo]:
        """城市级别定位 - 使用高德CitySearch API"""
        try:
            logger.info("🏙️ 尝试城市级别定位...")

            url = f"{self.base_url}/ip"
            params = {
                "key": self.api_key,
                "output": "json"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == '1':
                # 获取城市信息
                city = data.get('city', '')
                province = data.get('province', '')

                if city and province:
                    # 使用地理编码获取城市中心坐标
                    return self._get_city_center_location(city, province)

            return None

        except Exception as e:
            logger.warning(f"城市定位失败: {e}")
            return None

    def _try_system_location(self) -> Optional[LocationInfo]:
        """尝试系统位置服务"""
        try:
            # 在Windows上可以尝试使用位置API
            # 这里提供一个简化的实现
            logger.debug("尝试系统位置服务...")

            # 实际实现需要调用系统API
            # 这里返回None，表示不可用
            return None

        except Exception as e:
            logger.debug(f"系统位置服务不可用: {e}")
            return None

    def _get_city_center_location(self, city: str, province: str) -> Optional[LocationInfo]:
        """获取城市中心位置"""
        try:
            # 使用地理编码API获取城市中心坐标
            url = f"{self.base_url}/geocode/geo"
            params = {
                "key": self.api_key,
                "address": f"{province}{city}",
                "output": "json"
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == '1' and data.get('geocodes'):
                geocode = data['geocodes'][0]
                location_str = geocode.get('location', '')

                if location_str:
                    longitude, latitude = map(float, location_str.split(','))

                    return LocationInfo(
                        longitude=longitude,
                        latitude=latitude,
                        accuracy=5000.0,  # 城市级别精度
                        address=f"{province}{city}",
                        city=city,
                        district="",
                        province=province,
                        location_type="city",
                        confidence=0.6
                    )

            return None

        except Exception as e:
            logger.warning(f"获取城市中心位置失败: {e}")
            return None

    def _get_ip_location(self) -> Optional[LocationInfo]:
        """完善的IP定位 - 多重策略确保获取真实位置"""
        logger.info("🌐 开始IP定位...")

        # 策略1: 高德IP定位API
        location_info = self._try_amap_ip_location()
        if location_info:
            logger.info(f"✅ 高德IP定位成功: {location_info.address}")
            return location_info

        # 策略2: 第三方IP定位服务
        logger.info("🔄 高德IP定位失败，尝试第三方服务...")
        location_info = self._try_third_party_ip_location()
        if location_info:
            logger.info(f"✅ 第三方IP定位成功: {location_info.address}")
            return location_info

        # 策略3: 基于网络信息的粗略定位
        logger.info("🔄 第三方IP定位失败，尝试网络信息定位...")
        location_info = self._try_network_based_location()
        if location_info:
            logger.info(f"✅ 网络信息定位成功: {location_info.address}")
            return location_info

        logger.warning("❌ 所有IP定位策略都失败")
        return None

    def _try_amap_ip_location(self) -> Optional[LocationInfo]:
        """尝试高德IP定位API"""
        try:
            # 获取公网IP
            public_ip = self._get_public_ip()

            url = f"{self.base_url}/ip"
            params = {
                "key": self.api_key,
                "output": "json"
            }

            # 如果获取到公网IP，使用指定IP定位
            if public_ip:
                params["ip"] = public_ip
                logger.debug(f"使用指定IP定位: {public_ip}")
            else:
                logger.debug("使用自动IP检测")

            logger.debug(f"高德IP定位请求: {url}, 参数: {params}")

            response = requests.get(url, params=params, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()

            logger.debug(f"高德IP定位响应: {data}")

            if data.get('status') == '1':
                return self._parse_amap_ip_response(data)
            else:
                error_info = data.get('info', '未知错误')
                error_code = data.get('infocode', 'N/A')
                logger.warning(f"高德IP定位API失败: {error_info} (错误码: {error_code})")
                return None

        except Exception as e:
            logger.error(f"高德IP定位异常: {e}")
            return None

    def _parse_amap_ip_response(self, data: dict) -> Optional[LocationInfo]:
        """解析高德IP定位响应"""
        try:
            province = data.get('province', '')
            city = data.get('city', '')
            adcode = data.get('adcode', '')
            rectangle = data.get('rectangle', '')

            # 处理空数组问题
            if isinstance(province, list):
                province = province[0] if province else ''
            if isinstance(city, list):
                city = city[0] if city else ''
            if isinstance(adcode, list):
                adcode = adcode[0] if adcode else ''
            if isinstance(rectangle, list):
                rectangle = rectangle[0] if rectangle else ''

            logger.debug(f"解析高德IP响应: province={province}, city={city}, adcode={adcode}, rectangle={rectangle}")

            # 检查是否为局域网或无效IP
            if province == "局域网" or not province:
                logger.warning("检测到局域网IP或无效IP")
                return None

            # 尝试通过adcode获取地区信息
            if (not province or not city) and adcode:
                logger.info(f"通过adcode获取地区信息: {adcode}")
                province, city = self._get_location_by_adcode(adcode)

            # 如果仍然没有有效信息，返回None
            if not province or not city:
                logger.warning(f"高德IP定位信息不完整: province={province}, city={city}")
                return None

            # 构建地址
            address = f"{province}{city}"

            # 获取坐标
            longitude, latitude = None, None

            # 首先尝试从rectangle获取坐标
            if rectangle:
                longitude, latitude = self._parse_rectangle_coordinates(rectangle)
                logger.debug(f"从rectangle获取坐标: ({longitude}, {latitude})")

            # 如果rectangle无效，使用地理编码
            if longitude is None or latitude is None:
                logger.info(f"使用地理编码获取坐标: {address}")
                longitude, latitude = self._geocode_address(address)

            # 如果仍然无法获取坐标，返回None
            if longitude is None or latitude is None:
                logger.warning(f"无法获取地址 '{address}' 的坐标")
                return None

            return LocationInfo(
                longitude=longitude,
                latitude=latitude,
                accuracy=3000.0,  # IP定位精度较低
                address=address,
                city=city,
                district="",
                province=province,
                location_type="amap_ip",
                confidence=0.7
            )

        except Exception as e:
            logger.error(f"解析高德IP响应失败: {e}")
            return None

    def _try_third_party_ip_location(self) -> Optional[LocationInfo]:
        """尝试第三方IP定位服务"""
        services = [
            {
                'name': 'ipapi.co',
                'url': 'https://ipapi.co/json/',
                'parser': self._parse_ipapi_response,
                'timeout': 5
            },
            {
                'name': 'ip-api.com',
                'url': 'http://ip-api.com/json/',
                'parser': self._parse_ipapi_com_response,
                'timeout': 5
            },
            {
                'name': 'ipinfo.io',
                'url': 'https://ipinfo.io/json',
                'parser': self._parse_ipinfo_response,
                'timeout': 5
            }
        ]

        for service in services:
            try:
                logger.debug(f"尝试{service['name']}服务")
                response = requests.get(service['url'], timeout=service['timeout'])
                response.raise_for_status()
                data = response.json()

                location_info = service['parser'](data)
                if location_info:
                    # 转换为LocationInfo对象
                    longitude, latitude = self._geocode_address(f"{location_info['province']}{location_info['city']}")
                    if longitude and latitude:
                        return LocationInfo(
                            longitude=longitude,
                            latitude=latitude,
                            accuracy=5000.0,
                            address=f"{location_info['province']}{location_info['city']}",
                            city=location_info['city'],
                            district="",
                            province=location_info['province'],
                            location_type="third_party_ip",
                            confidence=0.6
                        )

            except Exception as e:
                logger.debug(f"{service['name']}服务失败: {e}")
                continue

        return None

    def _try_network_based_location(self) -> Optional[LocationInfo]:
        """基于网络信息的粗略定位"""
        try:
            # 尝试通过时区信息推断大致位置
            import time
            import datetime

            # 获取本地时区
            local_tz = datetime.datetime.now().astimezone().tzinfo
            tz_name = str(local_tz)

            logger.debug(f"本地时区: {tz_name}")

            # 简单的时区到地区映射
            timezone_mapping = {
                'China Standard Time': ('中国', '北京市'),
                'CST': ('中国', '北京市'),
                '+08:00': ('中国', '北京市'),
                'Asia/Shanghai': ('中国', '上海市'),
                'Asia/Beijing': ('中国', '北京市')
            }

            for tz_pattern, (country, city) in timezone_mapping.items():
                if tz_pattern in tz_name:
                    logger.info(f"基于时区推断位置: {country} {city}")

                    # 使用地理编码获取坐标
                    longitude, latitude = self._geocode_address(f"{country}{city}")
                    if longitude and latitude:
                        return LocationInfo(
                            longitude=longitude,
                            latitude=latitude,
                            accuracy=10000.0,  # 精度很低
                            address=f"{country}{city}",
                            city=city,
                            district="",
                            province=country,
                            location_type="timezone_based",
                            confidence=0.3
                        )

            return None

        except Exception as e:
            logger.error(f"网络信息定位失败: {e}")
            return None

    def _parse_rectangle_coordinates(self, rectangle: str) -> tuple[Optional[float], Optional[float]]:
        """
        从rectangle字段解析坐标
        rectangle格式: "116.0119343,39.66127144;116.7829835,40.2164962"
        返回中心点坐标
        """
        try:
            if not rectangle:
                return None, None

            # 分割左下角和右上角坐标
            coords_parts = rectangle.split(';')
            if len(coords_parts) != 2:
                logger.warning(f"rectangle格式不正确: {rectangle}")
                return None, None

            # 解析左下角坐标
            left_bottom = coords_parts[0].split(',')
            if len(left_bottom) != 2:
                return None, None

            # 解析右上角坐标
            right_top = coords_parts[1].split(',')
            if len(right_top) != 2:
                return None, None

            # 计算中心点坐标
            left_lng, left_lat = float(left_bottom[0]), float(left_bottom[1])
            right_lng, right_lat = float(right_top[0]), float(right_top[1])

            center_lng = (left_lng + right_lng) / 2
            center_lat = (left_lat + right_lat) / 2

            logger.debug(f"从rectangle解析坐标: ({center_lng}, {center_lat})")
            return center_lng, center_lat

        except Exception as e:
            logger.error(f"解析rectangle坐标失败: {e}")
            return None, None

    def _geocode_address(self, address: str, city: str = None) -> tuple[Optional[float], Optional[float]]:
        """
        增强地理编码API获取地址坐标
        根据高德API文档实现精确的地址解析
        """
        try:
            if not address:
                return None, None

            url = f"{self.base_url}/geocode/geo"
            params = {
                "key": self.api_key,
                "address": address,
                "output": "json"
            }

            # 如果指定了城市，添加城市参数提高精度
            if city:
                params["city"] = city

            logger.debug(f"地理编码请求: {url}, 参数: {params}")

            response = requests.get(url, params=params, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()

            logger.debug(f"地理编码响应: {data}")

            if data.get('status') == '1':
                geocodes = data.get('geocodes', [])
                if geocodes and len(geocodes) > 0:
                    # 选择最佳匹配结果
                    best_geocode = self._select_best_geocode(geocodes, address)

                    location = best_geocode.get('location', '')
                    if location:
                        lng_lat = location.split(',')
                        if len(lng_lat) == 2:
                            longitude = float(lng_lat[0])
                            latitude = float(lng_lat[1])

                            # 记录匹配级别和详细信息
                            level = best_geocode.get('level', '')
                            formatted_address = f"{best_geocode.get('province', '')}{best_geocode.get('city', '')}{best_geocode.get('district', '')}{best_geocode.get('street', '')}{best_geocode.get('number', '')}"

                            logger.info(f"地理编码成功: {address} -> ({longitude}, {latitude})")
                            logger.info(f"匹配级别: {level}, 格式化地址: {formatted_address}")
                            return longitude, latitude

            logger.warning(f"地理编码失败: {address}")
            return None, None

        except Exception as e:
            logger.error(f"地理编码异常: {e}")
            return None, None

    def _select_best_geocode(self, geocodes: list, original_address: str) -> dict:
        """选择最佳的地理编码结果"""
        try:
            if not geocodes:
                return {}

            if len(geocodes) == 1:
                return geocodes[0]

            # 优先级：门牌号 > 道路 > 兴趣点 > 区县
            level_priority = {
                '门牌号': 10,
                '道路': 8,
                '兴趣点': 6,
                '区县': 4,
                '市': 2,
                '省': 1
            }

            best_geocode = geocodes[0]
            best_score = 0

            for geocode in geocodes:
                level = geocode.get('level', '')
                score = level_priority.get(level, 0)

                # 如果地址包含具体街道或门牌号，优先选择门牌号级别
                if ('街' in original_address or '路' in original_address or '号' in original_address) and level == '门牌号':
                    score += 5

                if score > best_score:
                    best_score = score
                    best_geocode = geocode

            return best_geocode

        except Exception as e:
            logger.error(f"选择最佳地理编码结果失败: {e}")
            return geocodes[0] if geocodes else {}

    def _prompt_manual_location(self) -> Optional[LocationInfo]:
        """
        提示用户手动输入位置信息
        """
        try:
            logger.info("🗺️ 自动定位失败，请手动输入位置信息")
            print("\n🗺️ 自动定位失败，请手动输入您的位置信息：")

            # 获取用户输入的地址
            address = input("请输入您的详细地址（如：北京市朝阳区国贸）: ").strip()

            if not address:
                logger.warning("用户未输入地址，使用默认位置")
                return self.default_location

            # 尝试地理编码获取坐标
            longitude, latitude = self._geocode_address(address)

            if longitude is not None and latitude is not None:
                logger.info(f"手动输入地址地理编码成功: {address} -> ({longitude}, {latitude})")

                # 尝试逆地理编码获取详细信息
                detailed_info = self._reverse_geocode(longitude, latitude)

                if detailed_info:
                    return LocationInfo(
                        longitude=longitude,
                        latitude=latitude,
                        accuracy=100.0,  # 手动输入精度较高
                        address=detailed_info.get('formatted_address', address),
                        city=detailed_info.get('city', ''),
                        district=detailed_info.get('district', ''),
                        province=detailed_info.get('province', ''),
                        location_type="manual",
                        confidence=0.9  # 手动输入置信度高
                    )
                else:
                    return LocationInfo(
                        longitude=longitude,
                        latitude=latitude,
                        accuracy=100.0,
                        address=address,
                        city="",
                        district="",
                        province="",
                        location_type="manual",
                        confidence=0.8
                    )
            else:
                logger.warning(f"无法地理编码地址: {address}，使用默认位置")
                return self.default_location

        except Exception as e:
            logger.error(f"手动输入位置失败: {e}")
            return self.default_location

    def _reverse_geocode(self, longitude: float, latitude: float) -> Optional[dict]:
        """
        增强逆地理编码获取详细地址信息
        根据高德API文档实现完整的地址解析
        """
        try:
            url = f"{self.base_url}/geocode/regeo"
            params = {
                "key": self.api_key,
                "location": f"{longitude},{latitude}",
                "output": "json",
                "extensions": "all",  # 返回完整信息
                "radius": 1000,      # 搜索半径1000米
                "roadlevel": 0       # 显示所有道路
            }

            logger.debug(f"逆地理编码请求: {url}, 参数: {params}")
            response = requests.get(url, params=params, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()

            logger.debug(f"逆地理编码响应: {data}")

            if data.get('status') == '1':
                regeocode = data.get('regeocode', {})
                if regeocode:
                    formatted_address = regeocode.get('formatted_address', '')
                    address_component = regeocode.get('addressComponent', {})

                    # 获取基本地址组件
                    province = address_component.get('province', '')
                    city = address_component.get('city', '')
                    district = address_component.get('district', '')
                    township = address_component.get('township', '')

                    # 获取街道和门牌号信息
                    street_number = address_component.get('streetNumber', {})
                    street = street_number.get('street', '') if isinstance(street_number, dict) else ''
                    number = street_number.get('number', '') if isinstance(street_number, dict) else ''

                    # 获取建筑物和社区信息
                    building = address_component.get('building', {})
                    neighborhood = address_component.get('neighborhood', {})

                    building_name = building.get('name', '') if isinstance(building, dict) else ''
                    neighborhood_name = neighborhood.get('name', '') if isinstance(neighborhood, dict) else ''

                    # 获取商圈信息
                    business_areas = address_component.get('businessAreas', [])
                    business_area = ''
                    if business_areas and len(business_areas) > 0:
                        business_area = business_areas[0].get('name', '') if isinstance(business_areas[0], dict) else ''

                    result = {
                        'formatted_address': formatted_address,
                        'province': province,
                        'city': city,
                        'district': district,
                        'township': township,
                        'street': street,
                        'number': number,
                        'building': building_name,
                        'neighborhood': neighborhood_name,
                        'business_area': business_area
                    }

                    logger.info(f"逆地理编码成功: {formatted_address}")
                    return result

            else:
                error_info = data.get('info', '未知错误')
                logger.warning(f"逆地理编码API失败: {error_info}")
                return None

        except Exception as e:
            logger.error(f"逆地理编码失败: {e}")
            return None

    def _get_public_ip(self) -> Optional[str]:
        """获取公网IP地址"""
        try:
            services = [
                "https://httpbin.org/ip",
                "https://api.ip.sb/ip"
            ]

            for service in services:
                try:
                    response = requests.get(service, timeout=5)
                    if response.status_code == 200:
                        if "httpbin" in service:
                            return response.json().get('origin', '').split(',')[0].strip()
                        elif "ip.sb" in service:
                            return response.text.strip()
                except Exception:
                    continue

            return None
        except Exception:
            return None
    
    def _get_advanced_ip_location(self) -> Optional[LocationInfo]:
        """高级IP定位 - 综合多种策略的增强版本"""
        logger.info("🚀 开始高级IP定位...")

        try:
            # 策略1: 先尝试基础IP定位
            basic_location = self._get_ip_location()

            if basic_location and basic_location.confidence >= 0.6:
                # 如果基础定位成功且置信度较高，增强其属性
                enhanced_location = LocationInfo(
                    longitude=basic_location.longitude,
                    latitude=basic_location.latitude,
                    accuracy=basic_location.accuracy * 0.8,  # 提高精度
                    address=basic_location.address,
                    city=basic_location.city,
                    district=basic_location.district,
                    province=basic_location.province,
                    location_type="advanced_ip",
                    confidence=min(0.8, basic_location.confidence + 0.1)  # 提高置信度
                )

                # 尝试获取更详细的地址信息
                detailed_info = self._reverse_geocode(enhanced_location.longitude, enhanced_location.latitude)
                if detailed_info:
                    enhanced_location.address = detailed_info.get('formatted_address', enhanced_location.address)
                    enhanced_location.district = detailed_info.get('district', enhanced_location.district)

                logger.info(f"✅ 高级IP定位成功: {enhanced_location.address}")
                return enhanced_location

            # 策略2: 如果基础定位失败，尝试更激进的方法
            logger.info("🔄 基础定位失败，尝试激进策略...")

            # 尝试使用多个第三方服务的组合结果
            third_party_results = []

            # 收集多个服务的结果
            for i in range(3):  # 尝试3次以获得更稳定的结果
                result = self._try_third_party_ip_location()
                if result:
                    third_party_results.append(result)

            if third_party_results:
                # 使用最常见的结果
                best_result = self._select_best_location(third_party_results)
                if best_result:
                    best_result.location_type = "advanced_ip"
                    best_result.confidence = 0.7
                    logger.info(f"✅ 高级IP定位(第三方)成功: {best_result.address}")
                    return best_result

            logger.warning("❌ 高级IP定位失败")
            return None

        except Exception as e:
            logger.error(f"高级IP定位异常: {e}")
            return None

    def _select_best_location(self, locations: list) -> Optional[LocationInfo]:
        """从多个定位结果中选择最佳结果"""
        try:
            if not locations:
                return None

            if len(locations) == 1:
                return locations[0]

            # 统计相同地址的出现次数
            address_count = {}
            for loc in locations:
                key = f"{loc.province}{loc.city}"
                if key not in address_count:
                    address_count[key] = []
                address_count[key].append(loc)

            # 选择出现次数最多的地址
            best_key = max(address_count.keys(), key=lambda k: len(address_count[k]))
            best_locations = address_count[best_key]

            # 在相同地址中选择置信度最高的
            return max(best_locations, key=lambda loc: loc.confidence)

        except Exception as e:
            logger.error(f"选择最佳位置失败: {e}")
            return locations[0] if locations else None
    
    def _get_cached_location(self, key: str) -> Optional[LocationInfo]:
        """获取缓存的位置信息"""
        with self._cache_lock:
            if key in self._location_cache:
                cached_data = self._location_cache[key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['location']
                else:
                    # 缓存过期，删除
                    del self._location_cache[key]
        return None
    
    def _cache_location(self, key: str, location: LocationInfo):
        """缓存位置信息"""
        with self._cache_lock:
            self._location_cache[key] = {
                'location': location,
                'timestamp': time.time()
            }

    def _get_location_by_adcode(self, adcode: str) -> tuple[str, str]:
        """通过adcode获取省份和城市信息"""
        try:
            # 常见的adcode映射表（部分）
            adcode_mapping = {
                '110000': ('北京市', '北京市'),
                '120000': ('天津市', '天津市'),
                '130000': ('河北省', '石家庄市'),
                '140000': ('山西省', '太原市'),
                '150000': ('内蒙古自治区', '呼和浩特市'),
                '210000': ('辽宁省', '沈阳市'),
                '220000': ('吉林省', '长春市'),
                '230000': ('黑龙江省', '哈尔滨市'),
                '310000': ('上海市', '上海市'),
                '320000': ('江苏省', '南京市'),
                '330000': ('浙江省', '杭州市'),
                '340000': ('安徽省', '合肥市'),
                '350000': ('福建省', '福州市'),
                '360000': ('江西省', '南昌市'),
                '370000': ('山东省', '济南市'),
                '410000': ('河南省', '郑州市'),
                '420000': ('湖北省', '武汉市'),
                '430000': ('湖南省', '长沙市'),
                '440000': ('广东省', '广州市'),
                '450000': ('广西壮族自治区', '南宁市'),
                '460000': ('海南省', '海口市'),
                '500000': ('重庆市', '重庆市'),
                '510000': ('四川省', '成都市'),
                '520000': ('贵州省', '贵阳市'),
                '530000': ('云南省', '昆明市'),
                '540000': ('西藏自治区', '拉萨市'),
                '610000': ('陕西省', '西安市'),
                '620000': ('甘肃省', '兰州市'),
                '630000': ('青海省', '西宁市'),
                '640000': ('宁夏回族自治区', '银川市'),
                '650000': ('新疆维吾尔自治区', '乌鲁木齐市'),
            }

            # 首先尝试精确匹配
            if adcode in adcode_mapping:
                return adcode_mapping[adcode]

            # 尝试省级匹配（前两位）
            province_code = adcode[:2] + '0000'
            if province_code in adcode_mapping:
                return adcode_mapping[province_code]

            logger.warning(f"未找到adcode {adcode} 对应的地区信息")
            return '', ''

        except Exception as e:
            logger.error(f"通过adcode获取地区信息失败: {e}")
            return '', ''

    def _get_enhanced_ip_location(self) -> Optional[dict]:
        """增强IP定位 - 使用第三方服务获取更准确的IP位置信息"""
        try:
            import requests

            # 尝试多个IP定位服务
            services = [
                {
                    'name': 'ipapi.co',
                    'url': 'https://ipapi.co/json/',
                    'parser': self._parse_ipapi_response
                },
                {
                    'name': 'ip-api.com',
                    'url': 'http://ip-api.com/json/',
                    'parser': self._parse_ipapi_com_response
                }
            ]

            for service in services:
                try:
                    logger.debug(f"尝试使用{service['name']}获取IP位置")
                    response = requests.get(service['url'], timeout=5)
                    response.raise_for_status()
                    data = response.json()

                    location_info = service['parser'](data)
                    if location_info and location_info.get('province') and location_info.get('city'):
                        logger.info(f"通过{service['name']}获取到位置信息: {location_info}")
                        return location_info

                except Exception as e:
                    logger.debug(f"{service['name']}服务失败: {e}")
                    continue

            return None

        except Exception as e:
            logger.error(f"增强IP定位失败: {e}")
            return None

    def _parse_ipapi_response(self, data: dict) -> Optional[dict]:
        """解析ipapi.co的响应"""
        try:
            if data.get('error'):
                return None

            # 将英文地名转换为中文
            region = data.get('region', '')
            city = data.get('city', '')
            country = data.get('country_name', '')

            if country != 'China':
                return None  # 只处理中国的IP

            # 简单的英文到中文映射（可以扩展）
            region_mapping = {
                'Beijing': '北京市',
                'Shanghai': '上海市',
                'Guangdong': '广东省',
                'Zhejiang': '浙江省',
                'Jiangsu': '江苏省',
                'Shandong': '山东省',
                'Henan': '河南省',
                'Sichuan': '四川省',
                'Hubei': '湖北省',
                'Hunan': '湖南省'
            }

            city_mapping = {
                'Beijing': '北京市',
                'Shanghai': '上海市',
                'Guangzhou': '广州市',
                'Shenzhen': '深圳市',
                'Hangzhou': '杭州市',
                'Nanjing': '南京市'
            }

            province = region_mapping.get(region, region)
            city_name = city_mapping.get(city, city)

            return {
                'province': province,
                'city': city_name,
                'longitude': data.get('longitude'),
                'latitude': data.get('latitude')
            }

        except Exception as e:
            logger.error(f"解析ipapi响应失败: {e}")
            return None

    def _parse_ipapi_com_response(self, data: dict) -> Optional[dict]:
        """解析ip-api.com的响应"""
        try:
            if data.get('status') != 'success':
                return None

            country = data.get('country', '')
            if country != 'China':
                return None  # 只处理中国的IP

            region = data.get('regionName', '')
            city = data.get('city', '')

            # 简单的英文到中文映射
            region_mapping = {
                'Beijing': '北京市',
                'Shanghai': '上海市',
                'Guangdong': '广东省',
                'Zhejiang': '浙江省'
            }

            province = region_mapping.get(region, region)

            return {
                'province': province,
                'city': city,
                'longitude': data.get('lon'),
                'latitude': data.get('lat')
            }

        except Exception as e:
            logger.error(f"解析ip-api.com响应失败: {e}")
            return None

    def _get_enhanced_ip_location_direct(self) -> Optional[dict]:
        """直接获取增强IP定位信息，不依赖其他方法"""
        try:
            import requests

            # 尝试多个IP定位服务
            services = [
                {
                    'name': 'ipapi.co',
                    'url': 'https://ipapi.co/json/',
                    'parser': self._parse_ipapi_response
                },
                {
                    'name': 'ip-api.com',
                    'url': 'http://ip-api.com/json/',
                    'parser': self._parse_ipapi_com_response
                }
            ]

            for service in services:
                try:
                    logger.debug(f"尝试使用{service['name']}获取IP位置")
                    response = requests.get(service['url'], timeout=5)
                    response.raise_for_status()
                    data = response.json()

                    location_info = service['parser'](data)
                    if location_info and location_info.get('province') and location_info.get('city'):
                        logger.info(f"通过{service['name']}获取到位置信息: {location_info}")
                        return location_info

                except Exception as e:
                    logger.debug(f"{service['name']}服务失败: {e}")
                    continue

            return None

        except Exception as e:
            logger.error(f"直接增强IP定位失败: {e}")
            return None

    def _parse_ipinfo_response(self, data: dict) -> Optional[dict]:
        """解析ipinfo.io的响应"""
        try:
            if 'error' in data:
                return None

            country = data.get('country', '')
            region = data.get('region', '')
            city = data.get('city', '')

            # 只处理中国的IP
            if country != 'CN':
                return None

            # 简单的英文到中文映射
            region_mapping = {
                'Beijing': '北京市',
                'Shanghai': '上海市',
                'Guangdong': '广东省',
                'Zhejiang': '浙江省',
                'Jiangsu': '江苏省',
                'Shandong': '山东省'
            }

            city_mapping = {
                'Beijing': '北京市',
                'Shanghai': '上海市',
                'Guangzhou': '广州市',
                'Shenzhen': '深圳市',
                'Hangzhou': '杭州市'
            }

            province = region_mapping.get(region, region)
            city_name = city_mapping.get(city, city)

            if province and city_name:
                return {
                    'province': province,
                    'city': city_name
                }

            return None

        except Exception as e:
            logger.error(f"解析ipinfo响应失败: {e}")
            return None

    def _get_public_ip(self) -> Optional[str]:
        """获取公网IP地址"""
        try:
            # 尝试多个IP查询服务
            ip_services = [
                'https://api.ipify.org',
                'https://icanhazip.com',
                'https://ident.me',
                'http://httpbin.org/ip'
            ]

            for service in ip_services:
                try:
                    response = requests.get(service, timeout=3)
                    response.raise_for_status()

                    if 'httpbin.org' in service:
                        # httpbin返回JSON格式
                        ip = response.json().get('origin', '').split(',')[0].strip()
                    else:
                        # 其他服务返回纯文本
                        ip = response.text.strip()

                    # 验证IP格式
                    if self._is_valid_ip(ip):
                        logger.debug(f"获取到公网IP: {ip}")
                        return ip

                except Exception as e:
                    logger.debug(f"IP服务 {service} 失败: {e}")
                    continue

            logger.warning("无法获取公网IP")
            return None

        except Exception as e:
            logger.error(f"获取公网IP异常: {e}")
            return None

    def _is_valid_ip(self, ip: str) -> bool:
        """验证IP地址格式"""
        try:
            import ipaddress
            ipaddress.IPv4Address(ip)
            return True
        except:
            return False

    def correct_location(self, correct_address: str) -> Optional[LocationInfo]:
        """
        位置校正功能 - 允许用户输入准确的地址来校正定位

        Args:
            correct_address: 用户提供的准确地址

        Returns:
            校正后的位置信息
        """
        try:
            logger.info(f"🔧 开始位置校正: {correct_address}")

            # 使用地理编码获取准确坐标
            longitude, latitude = self._geocode_address(correct_address)

            if longitude and latitude:
                # 使用逆地理编码获取详细地址信息
                detailed_info = self._reverse_geocode(longitude, latitude)

                if detailed_info:
                    # 创建校正后的位置信息
                    corrected_location = LocationInfo(
                        longitude=longitude,
                        latitude=latitude,
                        accuracy=10.0,  # 手动校正的精度很高
                        address=detailed_info.get('formatted_address', correct_address),
                        city=detailed_info.get('city', ''),
                        district=detailed_info.get('district', ''),
                        province=detailed_info.get('province', ''),
                        location_type="manual_corrected",
                        confidence=0.95  # 手动校正的置信度很高
                    )

                    logger.info(f"✅ 位置校正成功: {corrected_location.address}")
                    return corrected_location
                else:
                    # 即使逆地理编码失败，也返回基本的位置信息
                    corrected_location = LocationInfo(
                        longitude=longitude,
                        latitude=latitude,
                        accuracy=50.0,
                        address=correct_address,
                        city="",
                        district="",
                        province="",
                        location_type="manual_corrected",
                        confidence=0.9
                    )

                    logger.info(f"✅ 位置校正成功(基础): {correct_address}")
                    return corrected_location
            else:
                logger.error(f"❌ 位置校正失败: 无法解析地址 {correct_address}")
                return None

        except Exception as e:
            logger.error(f"❌ 位置校正异常: {e}")
            return None

# 全局实例
_enhanced_location_service = None

def get_location_service() -> EnhancedLocationService:
    """获取增强定位服务单例"""
    global _enhanced_location_service
    if _enhanced_location_service is None:
        _enhanced_location_service = EnhancedLocationService()
    return _enhanced_location_service
