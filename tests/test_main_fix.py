#!/usr/bin/env python3
"""
测试主程序修复
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from main import BlindStarApp
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_main_fix():
    """测试主程序修复"""
    print("测试主程序修复...")
    
    try:
        # 创建参数对象
        args = argparse.Namespace()
        args.model = 'small'
        args.midas_model = 'MiDaS_small'
        args.confidence = 0.6
        args.device = 'auto'
        args.module = ['voice', 'poi']  # 只启用语音和POI，不启用视觉
        
        # 创建应用实例
        app = BlindStarApp(source=0, args=args)
        print("✓ BlindStarApp创建成功")
        
        # 初始化组件
        app.initialize()
        print("✓ 组件初始化成功")
        
        # 测试语音命令处理
        app.handle_voice_command("附近有什么餐厅")
        print("✓ 语音命令处理成功")
        
        print("✓ 主程序修复验证完成")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_fix() 