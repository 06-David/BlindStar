# tests 目录脚本使用说明

> 所有示例脚本位于 `tests/` 目录，可在 **项目根** 直接使用 Windows CMD 运行。脚本默认把结果写入 `logs/` 并自动避免重名。

---

## 1 . `analyze_video.py` — 单视频 / 摄像头分析

| 作用 | 对单个视频或摄像头 (`--source 0`) 执行 YOLOv8 目标检测 + MiDaS 深度估计，输出带框视频 `yolo.mp4`、深度视频 `depth.mp4`、深度统计 `depth_stats.csv` |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|

### 常用参数

| 参数 | 含义 | 默认 |
|------|------|------|
| `--source <path|0>` | 输入视频路径或摄像头 ID | **必填** |
| `--weights <pt>` | YOLOv8 权重文件 | `yolov8s.pt` |
| `--data <yaml>` | 类别定义 YAML，留空则用权重内置 names | — |
| `--conf <float>` | 置信度阈值 | `0.6` |
| `--device cpu|cuda` | 推理设备 | `auto` |

```cmd
REM 摄像头示例（GPU 自动选择）
python tests\analyze_video.py --source 0 --conf 0.5

REM 文件示例
python tests\analyze_video.py --source input\road.mp4 --weights runs\detect\train\weights\best.pt
```

输出目录：`logs/road_<时间戳>/`

---

## 2 . `batch_image_analysis.py` — 批量图像分析

| 作用 | 解析一个文件夹内所有图片，生成：`originals/` 原图、`yolo/` 检测图、`depth/` 深度伪彩色图 |
|------|-----------------------------------------------------------------------------------------------------------|

### 额外参数

| 参数 | 含义 |
|------|------|
| `--input <dir>` | **必填**，图片目录 |

```cmd
python tests\batch_image_analysis.py --input samples\imgs --weights yolov8s.pt
```

输出目录：`logs/batch_image_<时间戳>/`

---

## 3 . `batch_video_analysis.py` — 多视频批处理

| 作用 | 递归遍历目录中的所有视频文件，分别调用 *analyze_video* 逻辑生成结果 |
|------|------------------------------------------------------------------------------------------|

```cmd
python tests\batch_video_analysis.py --input samples\videos --conf 0.4
```

---

## 4 . `generate_depth_video.py` — 仅深度可视化

| 作用 | 仅使用 MiDaS 生成深度可视化视频与 CSV 统计，无目标检测 |
|------|---------------------------------------------------------------|

| 参数 | 含义 | 默认 |
|------|------|------|
| `--source <video>` | 输入视频 | **必填** |
| `--model <variant>` | MiDaS / DPT 变体 | `MiDaS_small` |
| `--device` | 推理设备 | `auto` |

```cmd
python tests\generate_depth_video.py --source input\road.mp4 --model DPT_Large
```

输出： `input\road_depth.mp4` 、 `input\road_depth_stats.csv`

---

## 环境准备

```cmd
pip install -r requirements.txt
```

- GPU 用户请确保 CUDA/cuDNN 与 PyTorch 版本匹配。
- 首次使用新权重或 MiDaS 变体时脚本会自动下载并缓存。

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 权重加载失败 | 确认 `--weights` 路径正确或已放入 `models/` |
| 类别名不正确 | 显式指定 `--data` 指向正确 YAML |
| CUDA 不可用 | 使用 `--device cpu` 或检查驱动 / torch 版本 | 