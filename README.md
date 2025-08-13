# BlindStar - 智能视觉辅助系统

> 基于YOLOv8/YOLO11和ZoeDepth的模块化计算机视觉系统，专为视障人士提供智能导航和环境感知，集成语音交互、POI查询和高德地图导航功能

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org)
[![ZoeDepth](https://img.shields.io/badge/ZoeDepth-Depth-purple.svg)](https://github.com/isl-org/ZoeDepth)
[![Vosk](https://img.shields.io/badge/Vosk-STT-red.svg)](https://alphacephei.com/vosk)
[![高德地图](https://img.shields.io/badge/高德地图-Navigation-blue.svg)](https://lbs.amap.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 项目概述

BlindStar是一个专为视障人士设计的智能视觉辅助系统，集成了先进的计算机视觉技术，提供实时的环境感知、物体识别和导航辅助功能。

### ✨ 核心功能

- **🔍 智能物体检测** - 基于YOLOv8/YOLO11，内置 COCO-80 类别，支持自定义训练权重
- **📏 精确距离测量** - ZoeDepth深度估计，提供更准确的空间信息
- **🏃 运动分析** - 实时速度计算和轨迹跟踪
- **🎵 语音交互** - 基于Vosk的离线语音识别和pyttsx3+SAPI双引擎语音合成
- **🗺️ 智能导航** - 集成高德地图API，支持POI查询和语音导航
- **🎬 视频处理** - 支持实时和批量视频分析
- **🔧 模块化设计** - 易于集成和扩展的组件架构

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA支持的GPU (推荐)
- 摄像头设备 (实时检测)

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/BlindStar.git
cd BlindStar

# 2. 激活conda环境
conda activate yolov8

# 3. 安装依赖
pip install -r requirements.txt

# 4. 开始使用
python main.py --source 0  # 摄像头检测

```

### 基本使用

```bash
# 实时摄像头检测（完整功能）
python main.py --source 0

# 启用语音导航功能
python main.py --source 0 --module vision voice distance poi navigation

# 处理视频文件
python main.py --source video.mp4 --save-video output.mp4

# 批量处理图片（使用测试脚本）
python tests/batch_image_analysis.py --input images/ --weights yolov8s.pt --with-depth

# 调整检测参数
python main.py --source 0 --conf 0.7 --model large

# 生成深度可视化视频 (ZoeDepth)
python tests/generate_depth_video.py --source input.mp4 --model DPT_Large --device cuda
```

## 📚 更多文档

- [项目背景与技术栈](docs/project_context.md)
- [实时启动器指南](docs/realtime_launcher.md)
- [项目架构](docs/architecture.md)
- [编程接口与输出格式](docs/api_reference.md)
- [系统配置](docs/configuration.md)
- [性能指标](docs/performance.md)
- [开发指南](docs/development_guide.md)
- [故障排除与技术支持](docs/troubleshooting.md)

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 物体检测框架
- [ZoeDepth](https://github.com/isl-org/ZoeDepth) - 高精度深度估计模型
- [OpenCV](https://opencv.org) - 计算机视觉库
- [Vosk](https://alphacephei.com/vosk) - 离线语音识别引擎
- [高德地图API](https://lbs.amap.com) - 地图服务和导航功能
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) - 文本转语音引擎

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

**BlindStar** - 让视觉无障碍，让世界更清晰 🌟
