#!/usr/bin/env python3
"""batch_image_analysis.py

批量图片检测脚本（Windows cmd 友好）。

功能：
1. 递归扫描输入目录下的所有图片（jpg、jpeg、png、bmp、webp）。
2. 使用 `core.detector.YOLOv8Detector` 对每张图片进行目标检测。
3. 将带框结果保存到输出目录，文件名保持一致。
4. 生成一个 `summary.csv`：image_path, num_detections, 每个检测的 class/conf。
5. 可选生成包含深度信息的组合图片。

命令行示例：
    python batch_image_analysis.py C:\Images \
           --output C:\DetectOut \
           --weights runs/detect/train/weights/best.pt \
           --data data.yaml \
           --conf 0.3 \
           --with-depth
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


import json
import traceback
import datetime

def analyze_images(input_dir: Path,
                   batch_dir: Path,
                   detector: YOLOv8Detector,
                   input_size=(384,512),
                   max_images=None,
                   parallel=False,
                   workers=1,
                   recursive=False) -> None:
    """
    批量处理图片，依次执行：
    1. 目标检测与检测框保存
    2. 深度伪彩色图生成
    3. 检测框深度数值标注
    4. 深度信息与检测框融合
    输出到originals/yolo/depth/combined四个子目录，统计信息写入summary.csv。
    """
    # 递归查找图片
    images = find_images(input_dir) if recursive else list(input_dir.glob('*'))
    images = [p for p in images if p.is_file() and p.suffix.lower() in SUPPORTED_IMG_EXTS]
    if max_images:
        images = images[:max_images]
    if not images:
        logger.warning("⚠️  未找到任何图片文件")
        return

    originals_dir = batch_dir / 'originals'  # 原图
    yolo_dir = batch_dir / 'yolo'            # 检测框+深度数值
    depth_dir = batch_dir / 'depth'          # 深度伪彩色
    combined_dir = batch_dir / 'combined'    # 检测与深度融合

    originals_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    summary_path = batch_dir / 'summary.csv'

    # 初始化深度可视化器和距离测量器
    depth_vis = DepthVisualizer(device='auto')
    from core.distance import ZoeDepthDistanceMeasurement
    distance_calc = ZoeDepthDistanceMeasurement(device='auto')

    # 统计信息
    start_time = time.time()
    total_files = len(images)
    successful_files = 0
    failed_files = 0
    error_files = []
    results = []

    with summary_path.open('w', newline='', encoding='utf-8') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['image', 'num_detections', 'detections', 'distances'])

        for img_path in tqdm(images, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]', desc='批量图片检测', unit='file'):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"⚠️  读取失败: {img_path}")
                    failed_files += 1
                    error_files.append(str(img_path))
                    continue

                # ------ 1. 目标检测 ------
                detections = detector.detect(img)

                # ------ 2. 深度数值计算 ------
                depth_infos = distance_calc.calculate_distances_batch(img, detections) if detections else []
                for detection, depth_info in zip(detections, depth_infos):
                    detection.distance = float(depth_info.distance_meters) if depth_info.distance_meters is not None else None

                # ------ 3. 检测框+深度数值图 ------
                annotated = draw_detections(img, detections, show_distance=True)

                # ------ 4. 深度伪彩色图 ------
                depth_color = depth_vis.get_colormap(img)

                # ------ 5. 检测与深度融合 ------
                combined = depth_vis.overlay(annotated, alpha=0.4)

                # ------ 6. 保存所有结果 ------
                rel_path = img_path.relative_to(input_dir)
                # 保存原图
                orig_path = originals_dir / rel_path
                orig_path.parent.mkdir(parents=True, exist_ok=True)
                if not orig_path.exists():
                    shutil.copy2(str(img_path), str(orig_path))
                # 检测框+深度数值
                yolo_path = yolo_dir / rel_path
                yolo_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(yolo_path), annotated)
                # 深度伪彩色
                depth_path = depth_dir / rel_path
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(depth_path), depth_color)
                # 检测与深度融合
                combined_path = combined_dir / rel_path
                combined_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(combined_path), combined)

                # ------ 7. 写入CSV统计 ------
                det_summary = '; '.join([f"{d.class_name}:{d.confidence:.2f}" for d in detections])
                dist_summary = '; '.join([f"{d.distance:.2f}m" if hasattr(d, 'distance') and d.distance is not None else "N/A" for d in detections])
                writer.writerow([str(img_path), len(detections), det_summary, dist_summary])

                # 记录结果
                results.append({
                    'image': str(img_path),
                    'num_detections': len(detections),
                    'detections': det_summary,
                    'distances': dist_summary
                })
                successful_files += 1
            except Exception as ex:
                failed_files += 1
                error_files.append(str(img_path))
                logger.error(f"❌ 处理失败: {img_path}, 错误: {ex}\n{traceback.format_exc()}")

    elapsed = time.time() - start_time
    # 自动生成详细json报告
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'summary': {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': (successful_files / total_files * 100) if total_files > 0 else 0,
            'total_processing_time': elapsed
        },
        'results': results,
        'errors': error_files
    }
    json_report_path = batch_dir / f"batch_image_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"✅ 完成，处理 {len(images)} 张图片，结果保存在: {batch_dir}")
    logger.info(f"📄 汇总 CSV: {summary_path}")
    logger.info(f"📝 详细JSON报告: {json_report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='批量图片检测')
    parser.add_argument('input_dir', help='输入图片文件夹')
    parser.add_argument('--weights', default='yolov8s.pt', help='YOLO权重文件路径（.pt）')
    parser.add_argument('--data', dest='data_yaml', default=None, help='数据集 YAML（可选）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', default='auto', help='设备 cpu/cuda (默认自动)')
    parser.add_argument('--input-size', default='384,512', help='深度模型输入尺寸，例如384,512')
    parser.add_argument('--max-images', type=int, default=None, help='最多处理图片数量')
    parser.add_argument('--parallel', action='store_true', help='是否并行处理图片')
    parser.add_argument('--workers', type=int, default=1, help='并行处理线程数')
    parser.add_argument('--recursive', action='store_true', help='递归查找图片')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细日志')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input_dir)

    # 自动生成批处理目录
    ts = time.strftime("%Y%m%d_%H%M%S")
    input_folder_name = input_dir.name
    batch_dir = Path("logs") / f"{input_folder_name}_{ts}"

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

    analyze_images(
        input_dir, batch_dir, detector,
        input_size=tuple(map(int, args.input_size.split(','))) if args.input_size else (384,512),
        max_images=args.max_images,
        parallel=args.parallel,
        workers=args.workers,
        recursive=args.recursive
    )
    return 0


if __name__ == '__main__':
    exit(main())