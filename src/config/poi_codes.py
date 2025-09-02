#!/usr/bin/env python3
"""
高德地图POI分类编码管理器
加载和管理POI分类编码表，支持从Excel文件加载最新编码
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from pypinyin import lazy_pinyin
import re

logger = logging.getLogger(__name__)


class POICodeManager:
    """POI分类编码管理器"""
    
    def __init__(self, poi_excel_path: Optional[str] = None):
        """
        初始化POI编码管理器
        
        Args:
            poi_excel_path: POI分类Excel文件路径
        """
        # 尝试多个可能的文件路径
        possible_paths = [
            poi_excel_path,
            "高德地图/楂樺痉POI鍒嗙被涓庣紪鐮侊紙涓嫳鏂囷級_V1.06_20230208.xlsx",
            "高德地图/高德POI分类与编码（中英文）_V1.06_20230208.xlsx",
            "高德地图/AMap_POI_Classification_V1.06_20230208.xlsx"
        ]

        self.poi_excel_path = None
        for path in possible_paths:
            if path and Path(path).exists():
                self.poi_excel_path = path
                break

        if not self.poi_excel_path:
            self.poi_excel_path = possible_paths[1]  # 使用默认路径
        
        # POI编码映射表
        self.poi_codes = {}
        self.poi_names = {}  # 反向映射：编码 -> 名称
        self.poi_hierarchy = {}  # 层级关系
        
        # 盲人友好的POI类型映射（基础版本）
        self.blind_friendly_types = {
            # 餐饮服务 - 盲人日常用餐需求
            "餐厅": "050000", "餐馆": "050000", "饭店": "050000", "中餐厅": "050100",
            "西餐厅": "050200", "快餐": "050500", "肯德基": "050501", "麦当劳": "050502",
            "咖啡厅": "050600", "咖啡馆": "050600", "茶馆": "050700", "酒吧": "050800",
            
            # 购物服务 - 盲人日常购物需求
            "超市": "060100", "便利店": "060101", "商场": "060200", "购物中心": "060200",
            "市场": "060300", "菜市场": "060301", "药店": "090700", "书店": "060500",
            
            # 生活服务 - 盲人重要的生活服务
            "银行": "160300", "ATM": "160301", "邮局": "070000", "快递": "070100",
            "洗衣店": "070200", "美容美发": "070300", "维修": "070400", "家政": "070500",
            
            # 医疗保健 - 盲人健康相关服务
            "医院": "090100", "诊所": "090200", "社区医院": "090300", "专科医院": "090400",
            "中医院": "090500", "牙科": "090600", "眼科": "090601", "按摩": "090800",
            
            # 交通设施 - 盲人出行关键设施
            "地铁": "150500", "地铁站": "150500", "公交": "150700", "公交站": "150700",
            "火车站": "150200", "高铁站": "150201", "机场": "150100", "长途汽车站": "150300",
            "停车场": "150900", "加油站": "010000",
            
            # 政府机构 - 盲人办事需求
            "政府": "130000", "派出所": "130200", "社区": "130300", "街道办": "130400",
            "民政局": "130500", "残联": "130600", "社保": "130700",
            
            # 教育文化 - 盲人学习和文化需求
            "学校": "140000", "大学": "140100", "中学": "140200", "小学": "140300",
            "图书馆": "140400", "博物馆": "140500", "盲校": "140600",
            
            # 体育休闲 - 盲人健身和休闲
            "公园": "110000", "广场": "110100", "健身房": "080300", "游泳馆": "080400",
            "按摩店": "070600", "足浴": "070700",
            
            # 住宿服务 - 盲人住宿需求
            "酒店": "100000", "宾馆": "100100", "旅馆": "100200", "招待所": "100300",
            
            # 汽车服务 - 盲人出行相关
            "汽车站": "150300", "出租车": "150800", "网约车": "150801"
        }
        
        # 加载POI编码
        self._load_poi_codes()
        
        logger.info(f"✅ POI编码管理器初始化完成，共加载{len(self.poi_codes)}个编码")
    
    def _load_poi_codes(self):
        """加载POI编码"""
        try:
            # 首先加载基础的盲人友好类型
            self.poi_codes.update(self.blind_friendly_types)
            
            # 尝试从Excel文件加载完整编码表
            if Path(self.poi_excel_path).exists():
                self._load_from_excel()
            else:
                logger.warning(f"POI编码Excel文件不存在: {self.poi_excel_path}")
                logger.info("使用内置的盲人友好POI编码")
            
            # 构建反向映射
            for name, code in self.poi_codes.items():
                if code not in self.poi_names:
                    self.poi_names[code] = []
                self.poi_names[code].append(name)
                
        except Exception as e:
            logger.error(f"加载POI编码失败: {e}")
            # 使用基础编码
            self.poi_codes = self.blind_friendly_types.copy()
    
    def _load_from_excel(self):
        """从Excel文件加载POI编码"""
        try:
            import pandas as pd
            
            # 读取Excel文件
            df = pd.read_excel(self.poi_excel_path, sheet_name=0)
            
            # 假设Excel文件有以下列：大类名称、中类名称、小类名称、编码
            for _, row in df.iterrows():
                try:
                    # 根据实际Excel文件结构调整列名
                    if '大类名称' in df.columns and '编码' in df.columns:
                        name = str(row['大类名称']).strip()
                        code = str(row['编码']).strip()
                        if name and code and name != 'nan' and code != 'nan':
                            self.poi_codes[name] = code
                    
                    # 如果有中类名称
                    if '中类名称' in df.columns:
                        name = str(row['中类名称']).strip()
                        code = str(row['编码']).strip()
                        if name and code and name != 'nan' and code != 'nan':
                            self.poi_codes[name] = code
                    
                    # 如果有小类名称
                    if '小类名称' in df.columns:
                        name = str(row['小类名称']).strip()
                        code = str(row['编码']).strip()
                        if name and code and name != 'nan' and code != 'nan':
                            self.poi_codes[name] = code
                            
                except Exception as e:
                    logger.debug(f"跳过无效行: {e}")
                    continue
            
            logger.info(f"✅ 从Excel文件加载了{len(self.poi_codes)}个POI编码")
            
        except ImportError:
            logger.warning("pandas未安装，无法读取Excel文件，使用内置编码")
        except Exception as e:
            logger.error(f"从Excel加载POI编码失败: {e}")
    
    def get_poi_code(self, keyword: str) -> Optional[str]:
        """
        获取关键词对应的POI编码
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            POI编码，如果未找到返回None
        """
        if not keyword:
            return None
        
        keyword = keyword.strip()
        
        # 1. 完全匹配
        if keyword in self.poi_codes:
            return self.poi_codes[keyword]
        
        # 2. 部分匹配
        for name, code in self.poi_codes.items():
            if keyword in name or name in keyword:
                return code
        
        # 3. 拼音匹配
        try:
            keyword_pinyin = ''.join(lazy_pinyin(keyword))
            for name, code in self.poi_codes.items():
                name_pinyin = ''.join(lazy_pinyin(name))
                if keyword_pinyin == name_pinyin or keyword_pinyin in name_pinyin:
                    return code
        except Exception as e:
            logger.debug(f"拼音匹配失败: {e}")
        
        # 4. 模糊匹配（去除常见后缀）
        cleaned_keyword = re.sub(r'(店|馆|厅|场|站|中心|服务|公司)$', '', keyword)
        if cleaned_keyword != keyword:
            return self.get_poi_code(cleaned_keyword)
        
        return None
    
    def get_poi_name(self, code: str) -> Optional[str]:
        """
        根据编码获取POI名称
        
        Args:
            code: POI编码
            
        Returns:
            POI名称，如果未找到返回None
        """
        names = self.poi_names.get(code, [])
        return names[0] if names else None
    
    def search_poi_types(self, keyword: str, limit: int = 10) -> List[Tuple[str, str]]:
        """
        搜索POI类型
        
        Args:
            keyword: 搜索关键词
            limit: 返回结果数量限制
            
        Returns:
            [(名称, 编码), ...] 的列表
        """
        results = []
        keyword = keyword.strip().lower()
        
        for name, code in self.poi_codes.items():
            if keyword in name.lower():
                results.append((name, code))
                if len(results) >= limit:
                    break
        
        return results
    
    def get_blind_friendly_types(self) -> Dict[str, str]:
        """获取盲人友好的POI类型"""
        return self.blind_friendly_types.copy()
    
    def is_blind_friendly(self, poi_type: str) -> bool:
        """判断POI类型是否对盲人友好"""
        return poi_type in self.blind_friendly_types
    
    def get_category_hierarchy(self, code: str) -> Dict[str, str]:
        """
        获取POI类型的层级信息
        
        Args:
            code: POI编码
            
        Returns:
            包含大类、中类、小类信息的字典
        """
        if len(code) >= 6:
            major_code = code[:2] + "0000"
            middle_code = code[:4] + "00"
            minor_code = code
            
            return {
                'major': self.get_poi_name(major_code) or "未知大类",
                'middle': self.get_poi_name(middle_code) or "未知中类", 
                'minor': self.get_poi_name(minor_code) or "未知小类",
                'major_code': major_code,
                'middle_code': middle_code,
                'minor_code': minor_code
            }
        
        return {}
    
    def save_codes_to_json(self, output_path: str = "config/poi_codes.json"):
        """将POI编码保存到JSON文件"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'poi_codes': self.poi_codes,
                'poi_names': self.poi_names,
                'blind_friendly_types': self.blind_friendly_types,
                'total_count': len(self.poi_codes),
                'update_time': str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ POI编码已保存到 {output_path}")
            
        except Exception as e:
            logger.error(f"保存POI编码失败: {e}")


# 全局POI编码管理器实例
_global_poi_manager = None

def get_poi_manager() -> POICodeManager:
    """获取全局POI编码管理器实例"""
    global _global_poi_manager
    if _global_poi_manager is None:
        _global_poi_manager = POICodeManager()
    return _global_poi_manager
