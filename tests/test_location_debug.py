#!/usr/bin/env python3
"""
调试位置问题
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.poi_query import POIQuery
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_location_debug():
    """调试位置问题"""
    print("调试位置问题...")
    
    poi = POIQuery()
    
    # 测试IP定位
    print("\n1. 测试IP定位:")
    location = poi.get_current_location_by_ip()
    print(f"IP定位结果: {location}")
    
    # 测试POI查询
    print("\n2. 测试POI查询:")
    results = poi.search_nearby('餐厅')
    
    # 检查前几个结果的地址
    print(f"\n找到 {len(results)} 个餐厅:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['name']}")
        print(f"   地址: {result['address']}")
        print(f"   距离: {result['distance']}米")
        print(f"   坐标: {result['longitude']}, {result['latitude']}")
        print()

if __name__ == "__main__":
    test_location_debug() 