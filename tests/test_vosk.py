#!/usr/bin/env python3
"""
测试Vosk语音识别功能
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.stt_engine import STTEngine
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_vosk():
    """测试Vosk语音识别"""
    print("测试Vosk语音识别...")
    
    try:
        # 初始化STT引擎
        stt = STTEngine()
        print("✓ Vosk引擎初始化成功")
        
        # 定义回调函数
        def on_speech(text):
            print(f"识别到语音: {text}")
        
        # 开始监听
        print("开始监听语音，请说话...")
        stt.start_listening(on_speech)
        
        # 运行10秒
        time.sleep(10)
        
        # 停止监听
        stt.stop_listening()
        print("测试完成")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")

if __name__ == "__main__":
    test_vosk() 