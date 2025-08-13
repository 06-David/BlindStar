import requests
import json
from typing import Dict, List, Optional
import logging
import urllib3
from pypinyin import lazy_pinyin  # 新增拼音库

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class POIQuery:
    def __init__(self, api_key: str="717d9a827c0ac3521932d3ae59aebbfe"):
        """
        初始化POI查询模块

        Args:
            api_key (str): 高德地图API密钥
        """
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3/place/around"
        self.ip_location_url = "https://restapi.amap.com/v3/ip"
        self.logger = logging.getLogger(__name__)

        # 默认位置（北京天安门）
        self.default_location = "116.397428,39.99923"
        
        # 缓存当前位置
        self._current_location = None

        # 完整的POI类型映射（根据高德API分类）
        self.poi_types = {
            # 餐饮服务
            "餐厅": "050000", "餐馆": "050000", "饭店": "050000", "中餐厅": "050000",
            "咖啡厅": "050100", "咖啡馆": "050100", "茶馆": "050200", "酒吧": "050400",
            "快餐": "050500", "肯德基": "050501", "麦当劳": "050502", "必胜客": "050503",

            # 购物服务
            "超市": "060100", "便利店": "060101", "商场": "060200", "购物中心": "060200",
            "市场": "060300", "菜市场": "060301", "专卖店": "060400", "药店": "061000",

            # 生活服务
            "银行": "160300", "ATM": "160301", "医院": "090100", "诊所": "090200",
            "药店": "090700", "邮局": "070000", "洗衣店": "070200", "美容美发": "070300",
            "维修": "070400",

            # 交通设施
            "地铁": "150500", "地铁站": "150500", "公交": "150700", "公交站": "150700",
            "火车站": "150200", "机场": "150100", "停车场": "150900",

            # 休闲娱乐
            "公园": "110100", "景区": "110200", "博物馆": "140400", "图书馆": "140500",
            "电影院": "130100", "KTV": "130200", "健身房": "130300", "体育馆": "130400",

            # 其他
            "酒店": "100000", "宾馆": "100000", "厕所": "200300"
        }

    def search_nearby(self, keyword: str, location: Optional[str] = None, radius: int = 1000) -> List[Dict]:
        """
        搜索周边POI（兴趣点）

        Args:
            keyword (str): 搜索关键词
            location (str, optional): 经纬度坐标，格式为"经度,纬度"
            radius (int, optional): 搜索半径（米），默认1000米（1公里）

        Returns:
            List[Dict]: POI结果列表，包含名称、类型、地址、距离等信息
        """
        # 确定位置
        if location:
            loc = location
        else:
            # 尝试通过IP获取当前位置
            current_loc = self.get_current_location_by_ip()
            loc = current_loc or self.default_location

        # 提取关键词对应的POI类型
        poi_type = self._match_poi_type(keyword)

        # 构建请求参数
        params = {
            "key": self.api_key,
            "location": loc,
            "keywords": keyword if not poi_type else "",
            "types": poi_type,
            "radius": radius,
            "offset": 20,  # 返回20条结果
            "page": 1,
            "extensions": "base",  # 基础信息
            "output": "json"
        }

        try:
            # 添加SSL验证禁用和更长的超时时间
            response = requests.get(
                self.base_url, 
                params=params, 
                timeout=30,
                verify=False,  # 禁用SSL验证
                headers={'User-Agent': 'BlindStar/1.0'}
            )
            response.raise_for_status()  # 检查HTTP错误

            data = response.json()
            self.logger.debug(f"API响应: {json.dumps(data, ensure_ascii=False)[:500]}...")

            if data.get("status") == "1":
                pois = data.get("pois", [])
                if isinstance(pois, list):
                    return self._process_results(pois)
                else:
                    self.logger.error(f"POI数据格式错误: {type(pois)}")
                    return []
            else:
                error_info = data.get('info', '未知错误')
                self.logger.error(f"POI查询失败: {error_info}")
                return []

        except requests.exceptions.RequestException as e:
            self.logger.error(f"POI请求异常: {e}")
            return []
        except json.JSONDecodeError:
            self.logger.error("POI响应解析失败")
            return []
        except Exception as e:
            self.logger.error(f"POI查询未知错误: {e}")
            return []

    def _match_poi_type(self, keyword: str) -> str:
        """
        匹配关键词对应的POI类型代码（增强版，支持拼音匹配）

        Args:
            keyword (str): 用户输入的关键词

        Returns:
            str: POI类型代码，如果未匹配则返回空字符串
        """
        # 首先尝试完全匹配
        if keyword in self.poi_types:
            return self.poi_types[keyword]

        # 尝试部分匹配
        for kw, poi_code in self.poi_types.items():
            if kw in keyword:
                return poi_code

        # 尝试拼音匹配（新增功能）
        try:
            keyword_pinyin = ''.join(lazy_pinyin(keyword))
            for kw, poi_code in self.poi_types.items():
                kw_pinyin = ''.join(lazy_pinyin(kw))
                # 检查拼音是否相似
                if kw_pinyin == keyword_pinyin:
                    return poi_code
                # 检查部分拼音匹配
                if keyword_pinyin in kw_pinyin:
                    return poi_code
        except Exception as e:
            self.logger.warning(f"拼音匹配失败: {e}")

        return ""

    def get_current_location_by_ip(self) -> Optional[str]:
        """
        通过IP获取当前位置

        Returns:
            Optional[str]: 经纬度坐标字符串，格式为"经度,纬度"，失败返回None
        """
        if self._current_location:
            return self._current_location

        try:
            params = {
                "key": self.api_key,
                "output": "json"
            }

            response = requests.get(
                self.ip_location_url,
                params=params,
                timeout=10,
                verify=False,
                headers={'User-Agent': 'BlindStar/1.0'}
            )
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "1":
                location = data.get("rectangle", "")
                if location:
                    # rectangle格式为"经度1,纬度1;经度2,纬度2"，取中心点
                    coords = location.split(";")
                    if len(coords) == 2:
                        # 计算中心点
                        coord1 = coords[0].split(",")
                        coord2 = coords[1].split(",")
                        if len(coord1) == 2 and len(coord2) == 2:
                            lng1, lat1 = float(coord1[0]), float(coord1[1])
                            lng2, lat2 = float(coord2[0]), float(coord2[1])
                            center_lng = (lng1 + lng2) / 2
                            center_lat = (lat1 + lat2) / 2

                            self._current_location = f"{center_lng:.6f},{center_lat:.6f}"
                            return self._current_location

            return None
            
        except Exception as e:
            self.logger.error(f"IP定位异常: {e}")
            return None

    def _process_results(self, pois: list) -> List[Dict]:
        """
        处理API返回的POI结果

        Args:
            pois (list): 原始POI数据列表

        Returns:
            List[Dict]: 处理后的POI信息列表
        """
        results = []
        for poi in pois:
            # 检查poi是否为字典类型
            if not isinstance(poi, dict):
                self.logger.warning(f"跳过非字典类型的POI数据: {type(poi)}")
                continue
                
            try:
                # 获取POI位置
                poi_location = poi.get("location", "")
                longitude, latitude = poi_location.split(",") if poi_location else (0, 0)

                # 计算距离（米）
                distance = float(poi.get("distance", 0))

                # 安全获取biz_ext字段
                biz_ext = poi.get("biz_ext", {})
                if isinstance(biz_ext, list):
                    biz_ext = {}
                
                results.append({
                    "id": poi.get("id", ""),
                    "name": poi.get("name", "未知地点"),
                    "type": poi.get("type", ""),
                    "address": poi.get("address", ""),
                    "distance": distance,  # 单位为米
                    "distance_km": distance / 1000,  # 单位为公里
                    "longitude": longitude,
                    "latitude": latitude,
                    "tel": poi.get("tel", ""),
                    "photos": poi.get("photos", []),
                    "rating": biz_ext.get("rating", "") if isinstance(biz_ext, dict) else "",
                    "cost": biz_ext.get("cost", "") if isinstance(biz_ext, dict) else ""
                })
            except Exception as e:
                self.logger.error(f"处理POI数据时出错: {e}, 数据: {poi}")
                continue

        # 按距离排序（从近到远）
        return sorted(results, key=lambda x: x["distance"])

    def format_poi_result(self, poi_data: List[Dict], max_results: int = 3) -> str:
        """
        格式化POI结果为自然语言描述

        Args:
            poi_data (List[Dict]): POI数据列表
            max_results (int, optional): 最大返回结果数，默认3

        Returns:
            str: 自然语言描述
        """
        if not poi_data:
            return "附近没有找到相关地点"

        # 只取前max_results个结果
        results = poi_data[:max_results]

        # 构建结果文本
        if len(poi_data) > max_results:
            text = f"附近找到{len(poi_data)}个地点，最近{max_results}个是: "
        else:
            text = f"找到{len(poi_data)}个地点: "

        for i, poi in enumerate(results):
            # 简化距离描述
            if poi["distance"] < 1000:  # 1公里以内用米表示
                distance_desc = f"{int(poi['distance'])}米"
            else:
                distance_desc = f"{poi['distance_km']:.1f}公里"

            text += f"{i + 1}. {poi['name']}（约{distance_desc}）；"

        return text

    def get_detailed_info(self, poi_id: str) -> Optional[Dict]:
        """
        获取POI的详细信息

        Args:
            poi_id (str): POI的ID

        Returns:
            Optional[Dict]: POI详细信息，如果失败返回None
        """
        detail_url = "http://restapi.amap.com/v3/place/detail"

        params = {
            "key": self.api_key,
            "id": poi_id,
            "output": "json"
        }

        try:
            response = requests.get(detail_url, params=params, timeout=5)
            data = response.json()

            if data.get("status") == "1" and data.get("pois"):
                poi_detail = data["pois"][0]
                return {
                    "name": poi_detail.get("name", ""),
                    "address": poi_detail.get("address", ""),
                    "tel": poi_detail.get("tel", ""),
                    "website": poi_detail.get("website", ""),
                    "photos": [{"title": photo.get("title", ""), "url": photo.get("url", "")}
                               for photo in poi_detail.get("photos", [])],
                    "business_hours": poi_detail.get("business_hours", ""),
                    "rating": poi_detail.get("biz_ext", {}).get("rating", ""),
                    "description": poi_detail.get("biz_ext", {}).get("description", "")
                }
            return None

        except Exception as e:
            self.logger.error(f"获取POI详情失败: {e}")
            return None