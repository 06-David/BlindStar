#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlindStar语音识别依赖安装脚本
"""

import subprocess
import sys
import os

def install_package(package_name):
    """安装Python包"""
    try:
        print(f"正在安装 {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {package_name} 安装成功")
            return True
        else:
            print(f"❌ {package_name} 安装失败:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {package_name} 安装超时")
        return False
    except Exception as e:
        print(f"❌ 安装 {package_name} 时出错: {e}")
        return False

def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """主函数"""
    print("🎤 BlindStar语音识别依赖安装程序")
    print("=" * 50)
    
    # 需要安装的包
    packages = [
        ("speechrecognition", "speech_recognition"),
        ("pyaudio", "pyaudio")
    ]
    
    installed_count = 0
    total_count = len(packages)
    
    for pip_name, import_name in packages:
        print(f"\n📦 检查 {pip_name}...")
        
        if check_package(import_name):
            print(f"✅ {pip_name} 已安装")
            installed_count += 1
        else:
            print(f"⚠️ {pip_name} 未安装，开始安装...")
            if install_package(pip_name):
                installed_count += 1
            else:
                print(f"❌ {pip_name} 安装失败")
    
    print("\n" + "=" * 50)
    print(f"📊 安装结果: {installed_count}/{total_count} 个包安装成功")
    
    if installed_count == total_count:
        print("🎉 所有语音识别依赖安装完成！")
        print("\n💡 现在您可以使用语音命令功能了：")
        print("   - 启动BlindStar: python blindstar_main.py")
        print("   - 测试语音功能: python test_voice_commands.py")
    else:
        print("⚠️ 部分依赖安装失败")
        print("\n🔧 手动安装命令:")
        for pip_name, import_name in packages:
            if not check_package(import_name):
                print(f"   pip install {pip_name}")
        
        print("\n💡 如果遇到PyAudio安装问题，请尝试:")
        print("   1. Windows: 下载预编译的wheel文件")
        print("   2. 或使用conda: conda install pyaudio")
        print("   3. 或安装Microsoft Visual C++ Build Tools")
    
    print("\n📚 更多信息请查看:")
    print("   - BlindStar_完整使用指南.md")
    print("   - PROJECT_STRUCTURE_FINAL.md")

if __name__ == "__main__":
    main()
