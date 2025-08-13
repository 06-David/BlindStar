#!/usr/bin/env python3
"""
Generate Depth Video
====================
根据输入视频，为每一帧计算 ZoeDepth 深度并输出：
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
from tqdm import tqdm  # 进度条库，如未安装请先 pip install tqdm

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
    model_type: str = "ZoeD_M12_NK",
    max_depth: float = 10.0,
    min_depth: float = 0.1,
    max_duration: float | None = None,
):
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if output_path is None:
        output_path = str(video_path_obj.with_name(f"{video_path_obj.stem}_depth.mp4"))
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if log_csv is None:
        log_csv = str(video_path_obj.with_name(f"{video_path_obj.stem}_depth_stats.csv"))
    log_csv_obj = Path(log_csv)

    logger.info(f"▶ 生成深度视频: {video_path} -> {output_path}")
    logger.info(f"📄 日志文件: {log_csv}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算需要处理的帧数（受 max_duration 限制）
    frames_to_process = total_frames
    if max_duration and max_duration > 0:
        try:
            max_frames = int(max_duration * float(fps))
            if max_frames > 0:
                frames_to_process = min(total_frames, max_frames)
        except Exception:
            pass

    fourcc = getattr(cv2, 'VideoWriter_fourcc')(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    depth_vis = DepthVisualizer(
        model_type=model_type,
        device=device,
        max_depth=max_depth,
        min_depth=min_depth,
    )

    start_time = time.time()
    stats_rows: List[Dict[str, Any]] = []
    processed_frames = 0

    # 使用tqdm进度条显示帧处理进度
    for frame_idx in tqdm(range(1, frames_to_process + 1), desc="处理帧", unit="帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # 计算深度图
        depth_map = depth_vis.get_depth_map(frame)
        
        # 生成伪彩色深度图
        depth_colored = depth_vis.get_colormap(frame)
        
        # 写入视频
        writer.write(depth_colored)
        processed_frames += 1

        # 记录统计信息
        stats = depth_stats(depth_map)
        stats["frame"] = frame_idx
        stats["timestamp"] = frame_idx / fps
        stats_rows.append(stats)

    cap.release()
    writer.release()
    depth_vis.cleanup()

    # 保存统计日志
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        if stats_rows:
            writer = csv.DictWriter(f, fieldnames=stats_rows[0].keys())
            writer.writeheader()
            writer.writerows(stats_rows)

    processing_time = time.time() - start_time
    logger.info(f"✅ 深度视频生成完成!")
    logger.info(f"   处理帧数: {processed_frames}")
    logger.info(f"   处理时间: {processing_time:.2f}s")
    logger.info(f"   平均FPS: {(processed_frames/processing_time) if processing_time > 0 else 0:.2f}")
    logger.info(f"   输出文件: {output_path}")
    logger.info(f"   统计日志: {log_csv}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成深度视频")
    parser.add_argument("video_path", help="输入视频文件路径")
    parser.add_argument("--output", "-o", help="输出视频文件路径")
    parser.add_argument("--log", "-l", help="统计日志CSV文件路径")
    parser.add_argument("--device", default="auto", 
                       help="计算设备 (auto/cpu/cuda)")
    parser.add_argument("--model", default="ZoeD_M12_NK",
                       help="深度模型类型 (ZoeD_M12_NK/ZoeD_N/ZoeD_K)")
    parser.add_argument("--max-depth", type=float, default=10.0,
                       help="最大深度值 (米)")
    parser.add_argument("--min-depth", type=float, default=0.1,
                       help="最小深度值 (米)")
    parser.add_argument("--max-duration", type=float, default=None,
                       help="最大处理时长（秒），将按 FPS 限制处理帧数")

    args = parser.parse_args()

    generate_depth_video(
        video_path=args.video_path,
        output_path=args.output,
        log_csv=args.log,
        device=args.device,
        model_type=args.model,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        max_duration=args.max_duration,
    )


if __name__ == "__main__":
    main() 