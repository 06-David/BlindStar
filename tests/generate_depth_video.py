#!/usr/bin/env python3
"""
Generate Depth Video
====================
根据输入视频，为每一帧计算 MiDaS 深度并输出：
1. 伪彩色深度视频（mp4）
2. 帧级深度统计日志（CSV）

用法：
    python generate_depth_video.py input.mp4 --output depth.mp4 --log depth_stats.csv
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
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

from core.depth_visualizer import DepthVisualizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def depth_stats(depth_map: np.ndarray) -> Dict[str, Any]:
    """返回一个深度图的统计信息。"""
    stats: Dict[str, Any] = {}
    if depth_map.size == 0:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    stats["min"] = float(np.min(depth_map))
    stats["max"] = float(np.max(depth_map))
    stats["mean"] = float(np.mean(depth_map))
    stats["median"] = float(np.median(depth_map))
    return stats


def generate_depth_video(
    video_path: str,
    output_path: str | None = None,
    log_csv: str | None = None,
    device: str = "auto",
    model_type: str = "MiDaS_small",
    max_depth: float = 10.0,
    min_depth: float = 0.1,
):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_depth.mp4")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if log_csv is None:
        log_csv = video_path.with_name(f"{video_path.stem}_depth_stats.csv")
    log_csv = Path(log_csv)

    logger.info(f"▶ 生成深度视频: {video_path} -> {output_path}")
    logger.info(f"📄 日志文件: {log_csv}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    depth_vis = DepthVisualizer(
        model_type=model_type,
        device=device,
        max_depth=max_depth,
        min_depth=min_depth,
    )

    start_time = time.time()
    stats_rows: List[Dict[str, Any]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        color_depth = depth_vis.get_colormap(frame)
        writer.write(color_depth)

        # 记录统计信息
        raw_depth = depth_vis.get_depth_map(frame)
        s = depth_stats(raw_depth)
        s.update({"frame": frame_idx})
        stats_rows.append(s)

        if frame_idx % 100 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames")

        frame_idx += 1

    cap.release()
    writer.release()
    depth_vis.cleanup()

    # 写入 CSV
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame", "min", "max", "mean", "median"])
        writer_csv.writeheader()
        writer_csv.writerows(stats_rows)

    duration = time.time() - start_time
    logger.info("✅ 深度视频生成完成")
    logger.info(f"帧数: {frame_idx}, 耗时: {duration:.2f}s, 平均 FPS: {frame_idx/duration:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate depth visualization video using MiDaS")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("--output", "-o", help="输出深度视频路径")
    parser.add_argument("--log", help="CSV 日志文件路径")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="推理设备")
    parser.add_argument("--model", default="MiDaS_small",
                        choices=["MiDaS_small", "MiDaS", "DPT_Large", "DPT_Hybrid", "DPT_SwinV2_L_384"],
                        help="MiDaS 模型类型")
    parser.add_argument("--max-depth", type=float, default=10.0, help="最大深度 (米)")
    parser.add_argument("--min-depth", type=float, default=0.1, help="最小深度 (米)")

    args = parser.parse_args()

    generate_depth_video(
        video_path=args.video,
        output_path=args.output,
        log_csv=args.log,
        device=args.device,
        model_type=args.model,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
    )


if __name__ == "__main__":
    main() 