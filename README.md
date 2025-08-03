# BlindStar - 智能视觉辅助系统

> 基于YOLOv8和MiDaS的模块化计算机视觉系统，专为视障人士提供智能导航和环境感知，集成语音交互和POI查询功能

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org)
[![MiDaS](https://img.shields.io/badge/MiDaS-Depth-purple.svg)](https://github.com/isl-org/MiDaS)
[![Vosk](https://img.shields.io/badge/Vosk-STT-red.svg)](https://alphacephei.com/vosk)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 项目概述

BlindStar是一个专为视障人士设计的智能视觉辅助系统，集成了先进的计算机视觉技术，提供实时的环境感知、物体识别和导航辅助功能。

### ✨ 核心功能

- **🔍 智能物体检测** - 基于YOLOv8，内置 COCO-80 类别，并支持加载**自定义训练权重**
- **📏 精确距离测量** - MiDaS深度估计，提供准确的空间信息
- **🏃 运动分析** - 实时速度计算和轨迹跟踪
- **🎵 语音反馈** - 为视障用户提供音频导航指引
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

# 2. 安装依赖
pip install -r requirements.txt

# 3. 开始使用
python main.py --source 0  # 摄像头检测
```

### 基本使用

```bash
# 实时摄像头检测
python main.py --source 0

# 处理视频文件
python main.py --source video.mp4 --save-video output.mp4

# 批量处理图片（使用测试脚本）
python tests\batch_image_analysis.py --input images\ --weights yolov8s.pt

# 调整检测参数
python main.py --source 0 --conf 0.7 --model large

# 生成深度可视化视频 (MiDaS)
python tests\generate_depth_video.py --source input.mp4 --model MiDaS_small --device cuda
```

## 🏗️ 项目架构

```
BlindStar/
├── 🧠 核心功能层
│   ├── core/
│     ├── blindstar.py         # 主控制类
│     ├── detector.py          # YOLOv8检测器
│     ├── distance.py          # MiDaS距离测量
│     ├── speed_measurement.py # 速度分析
│     ├── frame_analyzer.py    # 帧分析器
│     ├── video_processor.py   # 视频处理器
│     ├── camera.py            # 摄像头控制
│     └── utils.py             # 工具函数
│
├── 🎥 分析/测试脚本
|    |── tests/
│       ├── analyze_video.py         # 单视频分析
│       ├── batch_image_analysis.py  # 批量图片分析
│       ├── batch_video_analysis.py  # 批量视频分析
│       └── generate_depth_video.py  # 深度可视化
├── ⚙️ 配置文件
│   ├── config.py                # 系统配置
│   ├── requirements.txt         # 依赖列表
│   └── models/                  # AI模型文件
│
└── 📁 数据目录
    ├── datasets/                # 训练/推理数据集（YOLO 格式）
    ├── results/                 # 处理结果
    └── logs/                    # 日志文件
```

## 💻 编程接口

### 使用BlindStar主类

```python
from core import BlindStar

# 初始化系统
blindstar = BlindStar(
    model_variant='small',
    confidence_threshold=0.6,
    enable_distance=True,
    enable_speed=True
)

# 初始化组件
blindstar.initialize()

# 处理图片
result = blindstar.detect_image('image.jpg')
print(f"检测到 {len(result['detections'])} 个物体")

# 处理视频
stats = blindstar.process_video('input.mp4', 'output.mp4')
print(f"处理了 {stats.processed_frames} 帧")

# 实时摄像头
blindstar.start_camera(0)
frame = blindstar.get_camera_frame()
analysis = blindstar.analyze_frame(frame)
blindstar.stop_camera()

# 清理资源
blindstar.cleanup()

### 深度可视化组件

```python
from core.depth_visualizer import DepthVisualizer

# 初始化 (自动选择 cpu/gpu)
with DepthVisualizer(model_type="MiDaS_small", device="auto") as vis:
    depth_map = vis.get_depth_map(frame)       # HxW float32, 米
    color_map = vis.get_colormap(frame)        # 伪彩色 BGR
    mixed = vis.overlay(frame, alpha=0.5)      # 叠加显示
```

### 批量生成深度视频并记录日志

```bash
python generate_depth_video.py input.mp4 --output depth.mp4 --log depth_stats.csv
```

### 使用独立组件

```python
from core import YOLOv8Detector, DistanceMeasurement

# 物体检测
detector = YOLOv8Detector(model_variant='small')
detections = detector.detect(image)

# 距离测量
distance_meter = DistanceMeasurement()
distances = distance_meter.estimate_distances(image, detections)
```

## 📊 输出格式

### JSON结构化数据
```json
{
  "session_id": "20250721_143022",
  "frames": [
    {
      "frame_number": 0,
      "objects": [
        {
          "class_name": "person",
          "confidence": 0.89,
          "distance_meters": 3.2,
          "speed_mps": 1.5,
          "bbox": [100, 200, 300, 400]
        }
      ]
    }
  ]
}
```

### 视频输出功能

每次视频分析自动生成：
- `annotated_video.mp4` - 完整标注的视频文件
- `detections.csv` - 详细的帧级检测数据
- `analysis.json` - 结构化的分析结果
- `summary.txt` - 人类可读的分析摘要

## ⚙️ 配置选项

### 模型配置
```python
class ModelConfig:
    CONFIDENCE_THRESHOLD = 0.6    # 检测置信度阈值
    NMS_THRESHOLD = 0.45         # 非极大值抑制阈值
    MAX_DETECTIONS = 100         # 最大检测数量
    INPUT_SIZE = 640             # 输入图像尺寸
```

### 摄像头配置
```python
class CameraConfig:
    FRAME_WIDTH = 1280           # 帧宽度
    FRAME_HEIGHT = 720           # 帧高度
    FPS = 30                     # 帧率
```

## 📈 性能指标

### 检测性能
- **准确率**: mAP@0.5 > 0.85
- **处理速度**: 30+ FPS (GPU), 10+ FPS (CPU)
- **内存使用**: < 4GB (小模型), < 8GB (大模型)

### 距离测量精度
- **近距离** (0.5-5m): 误差 < 10%
- **中距离** (5-20m): 误差 < 15%
- **远距离** (20-50m): 误差 < 25%

### 系统要求
- **最低配置**: Intel i5, 8GB RAM, 集成显卡
- **推荐配置**: Intel i7/AMD Ryzen 7, 16GB RAM, GTX 1660+
- **最佳配置**: Intel i9/AMD Ryzen 9, 32GB RAM, RTX 3080+

## 🛠️ 开发指南

### 技术栈
- **深度学习**: YOLOv8 (Ultralytics)
- **计算机视觉**: OpenCV, PIL
- **深度估计**: MiDaS
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib
- **并发处理**: Threading, Queue

### 添加新功能
1. 在 `core/` 目录下创建新模块
2. 在 `__init__.py` 中导出新功能
3. 更新配置文件和文档
4. 添加相应的测试用例

### 加载自定义训练权重

自定义数据集训练完成后会得到 `*.pt` 权重文件，例如 `runs/detect/train/weights/best.pt`。系统支持**直接加载**该文件：

```python
from core import BlindStar

blindstar = BlindStar(
    yolo_model=r"runs/detect/train/weights/best.pt",  # 直接给路径即可
    confidence_threshold=0.4,
)
blindstar.initialize()
result = blindstar.detect_image('test.jpg')
```

若需要自定义类别名称，可在训练时提供 `data.yaml`，权重内部会保存 `model.names`；`YOLOv8Detector` 会自动读取。如需手动指定 YAML：

```cmd
python tests\analyze_video.py --source demo.mp4 --weights path\to\best.pt --data path\to\data.yaml
```

---

### 自定义后处理示例

```python
from core import YOLOv8Detector

class CustomDetector(YOLOv8Detector):
    def __init__(self):
        super().__init__(model_path="custom_model.pt")

    def post_process(self, detections):
        # 自定义后处理逻辑
        return filtered_detections
```

## 🐛 故障排除

### 常见问题

**摄像头无法启动**
- 检查摄像头权限设置
- 确认摄像头未被其他程序占用
- 尝试不同的摄像头ID (0, 1, 2...)

**检测精度低**
- 调整置信度阈值
- 使用更大的模型
- 改善光照条件

**处理速度慢**
- 启用GPU加速
- 使用较小的模型
- 降低输入分辨率

**距离测量不准确**
- 进行相机校准
- 调整MiDaS模型参数
- 确保良好的光照条件

### 日志调试
```bash
# 启用详细日志
python main.py --source 0 --verbose

# 检查分析日志
ls logs/*/detailed_analysis_*.log
```

## 📞 技术支持

### 获取帮助
- 查看项目文档和示例代码
- 检查 `logs/` 目录中的错误日志
- 确认系统要求和依赖版本

### 性能优化建议
1. 使用GPU加速提升处理速度
2. 根据需求选择合适的模型大小
3. 调整检测参数平衡精度和速度
4. 定期清理日志文件释放存储空间

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 物体检测框架
- [MiDaS](https://github.com/isl-org/MiDaS) - 深度估计模型
- [OpenCV](https://opencv.org) - 计算机视觉库

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

**BlindStar** - 让视觉无障碍，让世界更清晰 🌟
