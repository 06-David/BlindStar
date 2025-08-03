#!/usr/bin/env python3
"""batch_image_analysis.py

批量图片检测脚本（Windows cmd 友好）。

功能：
1. 递归扫描输入目录下的所有图片（jpg、jpeg、png、bmp、webp）。
2. 使用 `core.detector.YOLOv8Detector` 对每张图片进行目标检测。
3. 将带框结果保存到输出目录，文件名保持一致。
4. 生成一个 `summary.csv`：image_path, num_detections, 每个检测的 class/conf。

命令行示例：
    python batch_image_analysis.py C:\Images \
           --output C:\DetectOut \
           --weights runs/detect/train/weights/best.pt \
           --data data.yaml \
           --conf 0.3
"""
from __future__ import annotations
import sys
import os

# Add project root to Python path to allow running from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import csv
import logging
from pathlib import Path
from typing import List
import shutil
import cv2
from tqdm import tqdm
import time

from core.detector import YOLOv8Detector, draw_detections
from core.depth_visualizer import DepthVisualizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

SUPPORTED_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def find_images(root: Path) -> List[Path]:
    """Recursively list image files under root."""
    images: List[Path] = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMG_EXTS:
            images.append(p)
    return images


def analyze_images(input_dir: Path,
                   batch_dir: Path,
                   detector: YOLOv8Detector) -> None:
    """Detect objects in all images under input_dir and save annotated copies."""
    images = find_images(input_dir)
    if not images:
        logger.warning("⚠️  未找到任何图片文件")
        return

    originals_dir = batch_dir / 'originals'
    yolo_dir = batch_dir / 'yolo'
    depth_dir = batch_dir / 'depth'
    originals_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    summary_path = batch_dir / 'summary.csv'

    # 初始化深度可视化器一次即可
    depth_vis = DepthVisualizer(device='auto')

    with summary_path.open('w', newline='', encoding='utf-8') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['image', 'num_detections', 'detections'])

        for img_path in tqdm(images, desc="Processing images"):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"无法读取图片: {img_path}")
                continue

            detections = detector.detect(img)
            annotated = draw_detections(img, detections)

            # 保存原图
            orig_path = originals_dir / img_path.relative_to(input_dir)
            orig_path.parent.mkdir(parents=True, exist_ok=True)
            if not orig_path.exists():
                shutil.copy2(str(img_path), str(orig_path))

            # 保存标注图
            yolo_path = yolo_dir / img_path.relative_to(input_dir)
            yolo_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(yolo_path), annotated)

            # 生成并保存深度伪彩图
            depth_color = depth_vis.get_colormap(img)
            depth_path = depth_dir / img_path.relative_to(input_dir)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(depth_path), depth_color)

            # 写入 CSV 行
            det_summary = '; '.join([f"{d.class_name}:{d.confidence:.2f}" for d in detections])
            writer.writerow([str(img_path), len(detections), det_summary])

    logger.info(f"✅ 完成，处理 {len(images)} 张图片，结果保存在: {batch_dir}")
    logger.info(f"📄 汇总 CSV: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='批量图片检测')
    parser.add_argument('input_dir', help='输入图片文件夹')
    parser.add_argument('--weights', default='yolov8s.pt', help='权重文件路径（.pt）')
    parser.add_argument('--data', dest='data_yaml', default=None, help='数据集 YAML（可选）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', default='auto', help='设备 cpu/cuda (默认自动)')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细日志')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input_dir)

    # 自动生成批处理目录
    ts = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = Path("logs") / f"batch_image_{ts}"

    if not input_dir.exists():
        logger.error(f"❌ 输入目录不存在: {input_dir}")
        return 1

    logger.info(f"🚀 批量图片检测启动")
    logger.info(f"📁 输入目录: {input_dir}")
    logger.info(f"📁 输出目录: {batch_dir}")

    # 初始化检测器
    detector = YOLOv8Detector(
        model_variant=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
        data_yaml=args.data_yaml
    )

    analyze_images(input_dir, batch_dir, detector)
    return 0


if __name__ == '__main__':
    exit(main()) 