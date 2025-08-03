#!/usr/bin/env python3
"""
测试POI语音查询功能
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.poi_query import POIQuery
from core.tts_engine import TTSEngine
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_poi_voice():
    """测试POI语音查询"""
    print("测试POI语音查询功能...")
    
    try:
        # 初始化POI模块
        print("初始化POI模块...")
        poi = POIQuery()
        print("✓ POI模块初始化成功")
        
        # 初始化TTS
        print("初始化TTS模块...")
        tts = TTSEngine()
        print("✓ TTS模块初始化成功")
        
        # 模拟语音命令处理
        command = "附近有什么餐厅"
        print(f"模拟语音命令: {command}")
        
        # 提取关键词
        keyword = "餐厅"  # 默认值
        for kw in ["餐厅", "咖啡", "超市", "商场", "银行", "医院", "地铁", "公交", "公园"]:
            if kw in command:
                keyword = kw
                break
        
        print(f"提取的关键词: {keyword}")
        
        # 执行POI查询
        print("开始POI查询...")
        results = poi.search_nearby(keyword)
        print(f"查询完成，找到 {len(results)} 个结果")
        
        # 格式化结果
        response = poi.format_poi_result(results)
        print(f"格式化结果: {response}")
        
        # 语音播报
        print("开始语音播报...")
        tts.speak(response, blocking=True)
        print("✓ 语音播报完成")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_poi_voice() 