"""
Depth Visualizer Module
----------------------
为 BlindStar 提供整帧深度估计与可视化功能。

主要用途：
1. 在视频/摄像头处理时，为每一帧生成深度图 (float32, 单通道)。
2. 生成伪彩色深度图，便于直接保存或拼接展示。
3. 支持将深度伪彩色图叠加到原始帧，形成融合效果。

示例使用：
```python
from core.depth_visualizer import DepthVisualizer
vis = DepthVisualizer()
raw_depth = vis.get_depth_map(frame)   # HxW float32, 单位米
color = vis.get_colormap(frame)        # HxWx3 BGR, 0-255
mixed = vis.overlay(frame, alpha=0.6)  # 叠加显示
```
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple
from typing import Optional

from .distance import ZoeDepthDistanceMeasurement


class DepthVisualizer:
    """为单帧生成深度图与可视化的辅助类"""

    def __init__(
        self,
        model_type: str = "ZoeD_M12_NK",
        device: str = "auto",
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        model_path: Optional[str] = None,
    ) -> None:
        # 复用已有的 ZoeDepth 距离测量类
        self._distance = ZoeDepthDistanceMeasurement(
            model_type=model_type,
            device=device,
            max_depth=max_depth,
            min_depth=min_depth,
            model_path=model_path
        )

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """返回当前帧的深度图 (米)【float32, HxW】"""
        return self._distance.predict_depth(frame)

    def get_colormap(self, frame: np.ndarray) -> np.ndarray:
        """返回伪彩色深度图 (BGR uint8)【HxWx3】"""
        depth_map = self.get_depth_map(frame)
        return self._distance.get_depth_visualization(depth_map)

    def overlay(self, frame: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """将伪彩色深度图叠加到原图上。

        Args:
            frame: 原始 BGR 图像。
            alpha: 叠加权重；0 表示仅原图，1 表示仅深度图。
        """
        color_depth = self.get_colormap(frame)
        # resize 可能因 ZoeDepth 插值不精确造成 1px 偏差，做一次安全对齐
        if color_depth.shape[:2] != frame.shape[:2]:
            color_depth = cv2.resize(color_depth, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 1 - alpha, color_depth, alpha, 0)

    # ---------------------------------------------------------------------
    # Internal proxy
    # ---------------------------------------------------------------------
    def cleanup(self) -> None:
        """释放 GPU / 内存"""
        self._distance.cleanup()

    # 使其可用作上下文管理器
    def __enter__(self) -> "DepthVisualizer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup() 