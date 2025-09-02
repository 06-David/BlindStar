#!/usr/bin/env python3
"""
增强POI搜索服务
基于高德地图POI搜索2.0 API，提供智能POI搜索和输入提示功能
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
    from ..config.poi_codes import get_poi_manager
    from .enhanced_location import LocationInfo, get_location_service
except ImportError:
    try:
        # 尝试绝对导入
        from src.core.amap_client import get_amap_client
        from src.config.poi_codes import get_poi_manager
        from src.navigation.enhanced_location import LocationInfo, get_location_service
    except ImportError:
        # 兼容性处理
        get_amap_client = None
        get_poi_manager = None
        LocationInfo = None
        get_location_service = None

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """搜索类型"""
    TEXT = "text"           # 文本搜索
    AROUND = "around"       # 周边搜索
    POLYGON = "polygon"     # 多边形搜索
    ID = "id"              # ID搜索


class SortRule(Enum):
    """排序规则"""
    DISTANCE = "distance"   # 按距离排序
    WEIGHT = "weight"      # 按权重排序
    BLIND_FRIENDLY = "blind_friendly"  # 按盲人友好度排序


@dataclass
class POIInfo:
    """POI信息"""
    id: str
    name: str
    type_code: str
    type_name: str
    address: str
    location: LocationInfo
    distance: float = 0.0
    tel: str = ""
    website: str = ""
    business_hours: str = ""
    rating: str = ""
    photos: List[str] = field(default_factory=list)
    
    # 盲人友好信息
    accessibility_score: float = 0.0    # 无障碍评分
    has_braille: bool = False           # 是否有盲文
    has_audio_guide: bool = False       # 是否有语音导览
    wheelchair_accessible: bool = False  # 是否轮椅可达
    guide_dog_friendly: bool = False    # 是否导盲犬友好
    accessibility_notes: str = ""       # 无障碍说明
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'name': self.name,
            'type_code': self.type_code,
            'type_name': self.type_name,
            'address': self.address,
            'location': self.location.to_dict() if self.location else None,
            'distance': self.distance,
            'tel': self.tel,
            'website': self.website,
            'business_hours': self.business_hours,
            'rating': self.rating,
            'photos': self.photos,
            'accessibility_score': self.accessibility_score,
            'has_braille': self.has_braille,
            'has_audio_guide': self.has_audio_guide,
            'wheelchair_accessible': self.wheelchair_accessible,
            'guide_dog_friendly': self.guide_dog_friendly,
            'accessibility_notes': self.accessibility_notes
        }


@dataclass
class SearchResult:
    """搜索结果"""
    pois: List[POIInfo] = field(default_factory=list)
    total_count: int = 0
    search_time: float = 0.0
    search_type: SearchType = SearchType.TEXT
    query: str = ""
    location: Optional[LocationInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'pois': [poi.to_dict() for poi in self.pois],
            'total_count': self.total_count,
            'search_time': self.search_time,
            'search_type': self.search_type.value,
            'query': self.query,
            'location': self.location.to_dict() if self.location else None
        }


class EnhancedPOIService:
    """增强POI搜索服务"""
    
    def __init__(self, api_key: str = "717d9a827c0ac3521932d3ae59aebbfe"):
        """
        初始化增强POI搜索服务
        
        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3"
        self.base_url_v5 = "https://restapi.amap.com/v5"
        
        # 获取相关服务
        if get_amap_client:
            try:
                self.client = get_amap_client()
            except Exception as e:
                logger.warning(f"无法初始化API客户端: {e}")
                self.client = None
        else:
            self.client = None
        
        if get_poi_manager:
            try:
                self.poi_manager = get_poi_manager()
            except Exception as e:
                logger.warning(f"无法初始化POI管理器: {e}")
                self.poi_manager = None
        else:
            self.poi_manager = None
        
        if get_location_service:
            try:
                self.location_service = get_location_service()
            except Exception as e:
                logger.warning(f"无法初始化定位服务: {e}")
                self.location_service = None
        else:
            self.location_service = None
        
        # 搜索缓存
        self._search_cache = {}
        self._cache_lock = threading.RLock()
        self.cache_ttl = 1800  # 30分钟缓存
        
        # 盲人友好的POI类型权重
        self.blind_friendly_weights = {
            # 高权重 - 盲人日常必需
            "医院": 1.0, "药店": 1.0, "银行": 1.0, "ATM": 1.0,
            "超市": 0.9, "便利店": 0.9, "餐厅": 0.8, "公交站": 1.0,
            "地铁站": 1.0, "邮局": 0.8, "政府机构": 0.7,
            
            # 中权重 - 盲人常用
            "咖啡厅": 0.6, "书店": 0.6, "理发店": 0.6, "洗衣店": 0.5,
            "公园": 0.5, "图书馆": 0.7, "学校": 0.6,
            
            # 低权重 - 盲人较少使用
            "电影院": 0.3, "KTV": 0.2, "酒吧": 0.2, "网吧": 0.1
        }
        
        logger.info("✅ 增强POI搜索服务初始化完成")
    
    def search_text(self, 
                   keywords: str,
                   city: str = None,
                   location: str = None,
                   radius: int = 1000,
                   page: int = 1,
                   page_size: int = 20,
                   sort_rule: SortRule = SortRule.BLIND_FRIENDLY) -> SearchResult:
        """
        文本搜索POI
        
        Args:
            keywords: 搜索关键词
            city: 城市名称
            location: 中心点坐标
            radius: 搜索半径（米）
            page: 页码
            page_size: 每页数量
            sort_rule: 排序规则
            
        Returns:
            SearchResult: 搜索结果
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = f"text_{keywords}_{city}_{location}_{radius}_{page}_{page_size}_{sort_rule.value}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug("🎯 使用缓存的搜索结果")
                return cached_result
            
            # 获取POI类型编码
            poi_type = ""
            if self.poi_manager:
                poi_type = self.poi_manager.get_poi_code(keywords) or ""

            # 构建搜索参数
            params = {
                "keywords": keywords,  # 保留关键词
                "city": city or "",
                "offset": page_size,
                "page": page,
                "extensions": "all"
            }

            # 如果有POI类型编码，添加types参数
            if poi_type:
                params["types"] = poi_type
            
            if location:
                params["location"] = location
                params["radius"] = radius
            
            # 优先使用v3版本的周边搜索API（更稳定）
            search_data = self._call_search_api("poi_around", params)

            if not search_data:
                # 尝试v5版本的周边搜索
                logger.info("v3 API失败，尝试v5 API")
                search_data = self._call_search_api_v5("around", params)
            
            if not search_data:
                return SearchResult(
                    search_time=time.time() - start_time,
                    search_type=SearchType.TEXT,
                    query=keywords
                )
            
            # 解析搜索结果
            result = self._parse_search_result(search_data, SearchType.TEXT, keywords)
            
            # 应用排序规则
            if sort_rule != SortRule.DISTANCE:
                result.pois = self._sort_pois(result.pois, sort_rule)
            
            # 增强盲人友好信息
            self._enhance_accessibility_info(result.pois)
            
            result.search_time = time.time() - start_time
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"文本搜索失败: {e}")
            return SearchResult(
                search_time=time.time() - start_time,
                search_type=SearchType.TEXT,
                query=keywords
            )
    
    def search_around(self,
                     location: str,
                     keywords: str = "",
                     radius: int = 1000,
                     page: int = 1,
                     page_size: int = 20,
                     sort_rule: SortRule = SortRule.BLIND_FRIENDLY) -> SearchResult:
        """
        周边搜索POI
        
        Args:
            location: 中心点坐标
            keywords: 搜索关键词（可选）
            radius: 搜索半径（米）
            page: 页码
            page_size: 每页数量
            sort_rule: 排序规则
            
        Returns:
            SearchResult: 搜索结果
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = f"around_{location}_{keywords}_{radius}_{page}_{page_size}_{sort_rule.value}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug("🎯 使用缓存的搜索结果")
                return cached_result
            
            # 获取POI类型编码
            poi_type = ""
            if keywords and self.poi_manager:
                poi_type = self.poi_manager.get_poi_code(keywords) or ""

            # 构建搜索参数
            params = {
                "location": location,
                "radius": radius,
                "offset": page_size,
                "page": page,
                "extensions": "all"
            }

            # 添加关键词或类型
            if keywords:
                params["keywords"] = keywords
            if poi_type:
                params["types"] = poi_type
            
            # 调用周边搜索API
            search_data = self._call_search_api("around_search", params)
            
            if not search_data:
                return SearchResult(
                    search_time=time.time() - start_time,
                    search_type=SearchType.AROUND,
                    query=keywords
                )
            
            # 解析搜索结果
            result = self._parse_search_result(search_data, SearchType.AROUND, keywords)
            
            # 应用排序规则
            if sort_rule != SortRule.DISTANCE:
                result.pois = self._sort_pois(result.pois, sort_rule)
            
            # 增强盲人友好信息
            self._enhance_accessibility_info(result.pois)
            
            result.search_time = time.time() - start_time
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"周边搜索失败: {e}")
            return SearchResult(
                search_time=time.time() - start_time,
                search_type=SearchType.AROUND,
                query=keywords
            )
    
    def get_input_tips(self, keywords: str, city: str = None, limit: int = 10) -> List[Dict[str, str]]:
        """
        获取输入提示
        
        Args:
            keywords: 输入关键词
            city: 城市名称
            limit: 返回数量限制
            
        Returns:
            提示列表
        """
        try:
            params = {
                "keywords": keywords,
                "type": "all",
                "location": "",
                "city": city or "",
                "datatype": "all"
            }
            
            if self.client:
                success, data, error = self.client.request("input_tips", **params)
            else:
                url = f"{self.base_url}/assistant/inputtips"
                params["key"] = self.api_key
                params["output"] = "json"
                
                response = requests.get(url, params=params, timeout=10, verify=False)
                response.raise_for_status()
                data = response.json()
                success = data.get('status') == '1'
            
            if success and data:
                tips = data.get('tips', [])
                results = []
                
                for tip in tips[:limit]:
                    results.append({
                        'name': tip.get('name', ''),
                        'district': tip.get('district', ''),
                        'adcode': tip.get('adcode', ''),
                        'location': tip.get('location', ''),
                        'address': tip.get('address', ''),
                        'typecode': tip.get('typecode', '')
                    })
                
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"获取输入提示失败: {e}")
            return []
    
    def _call_search_api(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """调用搜索API"""
        try:
            if self.client:
                success, data, error = self.client.request(endpoint, **params)
            else:
                # 根据端点选择URL
                if endpoint == "poi_search_v2":
                    url = f"{self.base_url_v5}/place/text"
                elif endpoint == "poi_search":
                    url = f"{self.base_url}/place/text"
                elif endpoint == "poi_around":
                    url = f"{self.base_url}/place/around"
                elif endpoint == "around_search":
                    url = f"{self.base_url}/place/around"
                else:
                    logger.warning(f"未知的API端点: {endpoint}")
                    return None
                
                params["key"] = self.api_key
                params["output"] = "json"

                logger.debug(f"POI搜索API调用: {url}")
                logger.debug(f"请求参数: {params}")

                response = requests.get(url, params=params, timeout=30, verify=False)
                response.raise_for_status()
                data = response.json()
                success = data.get('status') == '1'

                logger.debug(f"API响应状态: {data.get('status')}")
                logger.debug(f"API响应信息: {data.get('info')}")
                if not success:
                    logger.warning(f"API调用失败: {data.get('info')} (错误码: {data.get('infocode')})")
            
            if success and data:
                return data
            else:
                logger.error(f"搜索API调用失败: {data.get('info', '未知错误') if data else error}")
                return None
                
        except Exception as e:
            logger.error(f"搜索API调用异常: {e}")
            return None
    
    def _parse_search_result(self, 
                           search_data: Dict[str, Any], 
                           search_type: SearchType,
                           query: str) -> SearchResult:
        """解析搜索结果"""
        try:
            pois_data = search_data.get('pois', [])
            total_count = int(search_data.get('count', 0))
            
            pois = []
            for poi_data in pois_data:
                try:
                    # 解析位置信息
                    location_str = poi_data.get('location', '')
                    if location_str:
                        coords = location_str.split(',')
                        if len(coords) >= 2:
                            location = LocationInfo(
                                longitude=float(coords[0]),
                                latitude=float(coords[1]),
                                accuracy=10.0,
                                address=poi_data.get('address', ''),
                                city=poi_data.get('cityname', ''),
                                district=poi_data.get('adname', ''),
                                province=poi_data.get('pname', ''),
                                location_type="poi"
                            )
                        else:
                            continue
                    else:
                        continue
                    
                    # 创建POI信息
                    poi = POIInfo(
                        id=poi_data.get('id', ''),
                        name=poi_data.get('name', ''),
                        type_code=poi_data.get('typecode', ''),
                        type_name=poi_data.get('type', ''),
                        address=poi_data.get('address', ''),
                        location=location,
                        distance=float(poi_data.get('distance', 0)),
                        tel=poi_data.get('tel', ''),
                        website=poi_data.get('website', ''),
                        business_hours=poi_data.get('business_hours', ''),
                        rating=str(poi_data.get('biz_ext', {}).get('rating', '')),
                        photos=[photo.get('url', '') for photo in poi_data.get('photos', [])]
                    )
                    
                    pois.append(poi)
                    
                except Exception as e:
                    logger.debug(f"解析POI数据失败: {e}")
                    continue
            
            return SearchResult(
                pois=pois,
                total_count=total_count,
                search_type=search_type,
                query=query
            )
            
        except Exception as e:
            logger.error(f"解析搜索结果失败: {e}")
            return SearchResult(
                search_type=search_type,
                query=query
            )
    
    def _sort_pois(self, pois: List[POIInfo], sort_rule: SortRule) -> List[POIInfo]:
        """排序POI列表"""
        try:
            if sort_rule == SortRule.DISTANCE:
                return sorted(pois, key=lambda poi: poi.distance)
            elif sort_rule == SortRule.WEIGHT:
                return sorted(pois, key=lambda poi: float(poi.rating) if poi.rating else 0, reverse=True)
            elif sort_rule == SortRule.BLIND_FRIENDLY:
                return sorted(pois, key=lambda poi: self._calculate_blind_friendly_score(poi), reverse=True)
            else:
                return pois
                
        except Exception as e:
            logger.error(f"排序POI失败: {e}")
            return pois
    
    def _calculate_blind_friendly_score(self, poi: POIInfo) -> float:
        """计算盲人友好度评分"""
        try:
            score = 0.5  # 基础分
            
            # 基于POI类型的权重
            for poi_type, weight in self.blind_friendly_weights.items():
                if poi_type in poi.type_name or poi_type in poi.name:
                    score += weight * 0.3
                    break
            
            # 距离因子（距离越近分数越高）
            if poi.distance > 0:
                distance_factor = max(0, 1 - poi.distance / 2000)  # 2公里内有效
                score += distance_factor * 0.2
            
            # 评分因子
            if poi.rating:
                try:
                    rating_score = float(poi.rating) / 5.0
                    score += rating_score * 0.1
                except ValueError:
                    pass
            
            # 无障碍设施加分
            score += poi.accessibility_score * 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"计算盲人友好度评分失败: {e}")
            return 0.5
    
    def _enhance_accessibility_info(self, pois: List[POIInfo]):
        """增强无障碍信息"""
        try:
            for poi in pois:
                # 基于POI类型和名称推断无障碍信息
                poi_name = poi.name.lower()
                poi_type = poi.type_name.lower()
                
                # 医院、银行等通常有较好的无障碍设施
                if any(keyword in poi_name or keyword in poi_type 
                       for keyword in ['医院', '银行', '政府', '学校', '图书馆']):
                    poi.accessibility_score = 0.8
                    poi.wheelchair_accessible = True
                    poi.accessibility_notes = "通常具备基础无障碍设施"
                
                # 地铁站、公交站通常有盲道
                if any(keyword in poi_name or keyword in poi_type 
                       for keyword in ['地铁', '公交', '车站']):
                    poi.accessibility_score = 0.9
                    poi.has_braille = True
                    poi.has_audio_guide = True
                    poi.accessibility_notes = "通常有盲道和语音提示"
                
                # 商场、超市等商业场所
                if any(keyword in poi_name or keyword in poi_type 
                       for keyword in ['商场', '超市', '购物']):
                    poi.accessibility_score = 0.6
                    poi.wheelchair_accessible = True
                    poi.guide_dog_friendly = True
                    poi.accessibility_notes = "商业场所，通常允许导盲犬"
                
                # 默认评分
                if poi.accessibility_score == 0.0:
                    poi.accessibility_score = 0.5
                    poi.accessibility_notes = "无障碍信息待确认"
                    
        except Exception as e:
            logger.error(f"增强无障碍信息失败: {e}")
    
    def _get_cached_result(self, cache_key: str) -> Optional[SearchResult]:
        """获取缓存的搜索结果"""
        with self._cache_lock:
            if cache_key in self._search_cache:
                cached_data = self._search_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['result']
                else:
                    # 缓存过期，删除
                    del self._search_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: SearchResult):
        """缓存搜索结果"""
        with self._cache_lock:
            self._search_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }

    def _call_search_api_v5(self, endpoint_type: str, params: dict) -> Optional[dict]:
        """调用高德地图v5版本搜索API"""
        try:
            if self.client:
                # 构建v5 API URL
                if endpoint_type == "around":
                    url = "https://restapi.amap.com/v5/place/around"
                elif endpoint_type == "text":
                    url = "https://restapi.amap.com/v5/place/text"
                else:
                    return None

                # 调整参数格式以适配v5 API
                v5_params = self._convert_params_to_v5(params)

                logger.debug(f"调用v5 API: {url}, 参数: {v5_params}")

                # 直接使用requests调用v5 API，因为客户端可能不支持自定义URL
                import requests

                # 添加API密钥
                v5_params['key'] = self.client.api_key

                logger.debug(f"直接调用v5 API: {url}")
                logger.debug(f"参数: {v5_params}")

                response = requests.get(url, params=v5_params, timeout=10, verify=False)
                response.raise_for_status()
                data = response.json()

                if data.get('status') == '1':
                    success = True
                    error = None
                else:
                    success = False
                    error = data.get('info', '未知错误')
                    data = None

                if success and data:
                    logger.debug(f"v5 API调用成功，返回{data.get('count', 0)}个结果")
                    return data
                else:
                    logger.error(f"v5 API调用失败: {error}")
                    return None
            else:
                logger.warning("API客户端不可用")
                return None

        except Exception as e:
            logger.error(f"v5 API调用异常: {e}")
            return None

    def _convert_params_to_v5(self, params: dict) -> dict:
        """将参数转换为v5 API格式"""
        v5_params = {}

        # 基础参数映射
        if "keywords" in params:
            v5_params["keywords"] = params["keywords"]

        if "types" in params:
            v5_params["types"] = params["types"]

        if "location" in params:
            v5_params["location"] = params["location"]

        if "radius" in params:
            v5_params["radius"] = params["radius"]

        # v5特有参数
        v5_params["page_size"] = params.get("offset", 10)
        v5_params["page_num"] = params.get("page", 1)
        v5_params["sortrule"] = "distance"  # 按距离排序，适合盲人用户

        # 返回更多字段信息
        v5_params["show_fields"] = "business,navi,photos"

        return v5_params


# 全局POI搜索服务实例
_global_poi_service = None

def get_poi_service() -> EnhancedPOIService:
    """获取全局POI搜索服务实例"""
    global _global_poi_service
    if _global_poi_service is None:
        _global_poi_service = EnhancedPOIService()
    return _global_poi_service
