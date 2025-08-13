"""
ZoeDepth Distance Measurement Module
Uses ZoeDepth depth estimation model to calculate accurate object distances
"""

import warnings
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import time
from dataclasses import dataclass
import sys
import os

# Ensure zoedepth local package is discoverable before first import
models_root = Path(__file__).parent.parent / "models"
if str(models_root) not in sys.path:
    sys.path.insert(0, str(models_root))
from functools import lru_cache

from .detector import DetectionResult

from models.zoedepth.utils.config import get_config
from models.zoedepth.models.builder import build_model
logger = logging.getLogger(__name__)

os.environ["HF_HUB_OFFLINE"] = "1" 
@dataclass
class DepthInfo:
    """Depth information for an object"""
    distance_meters: float
    depth_confidence: float
    depth_map_region: np.ndarray
    min_depth: float
    max_depth: float
    median_depth: float


class ZoeDepthDistanceMeasurement:
    """
    ZoeDepth-based distance measurement calculator
    
    Uses ZoeDepth depth estimation model to calculate accurate object distances
    from monocular images. Provides more accurate depth estimation compared
    to geometric methods.
    
    主要优化：
    1. 减少图像预处理开销
    2. 优化推理流程
    3. 添加结果缓存
    4. 减少不必要的转换
    5. 优化内存使用
    """
    
    # 全局模型缓存
    _model_cache = {}
    _config_cache = {}
    
    def __init__(self, 
                 model_type: str = "ZoeD_N",
                 device: str = "auto",
                 max_depth: float = 10.0,
                 min_depth: float = 0.1,
                 distance_unit: str = "meters",
                 input_size: Tuple[int, int] = (256, 384),  # 更小的默认尺寸
                 enable_cache: bool = True,
                 enable_result_cache: bool = True,
                 cache_size: int = 10,
                 enable_absolute_depth: bool = True,
                 depth_scale_factor: float = 1.0,
                 calibration_distance: Optional[float] = None,
                 model_path: Optional[str] = None):
        """
        Initialize ZoeDepth distance measurement calculator
        
        Args:
            model_type: ZoeDepth model variant ('ZoeD_N', 'ZoeD_K', 'ZoeD_NK')
            device: Device to use ('auto', 'cpu', 'cuda')
            max_depth: Maximum depth in meters
            min_depth: Minimum depth in meters
            distance_unit: Unit for distance measurements ('meters', 'cm', 'inches')
            input_size: Input image size for faster inference (width, height)
            enable_cache: Enable model caching for faster loading
            enable_result_cache: Enable result caching for repeated images
            cache_size: Size of result cache
            enable_absolute_depth: Enable absolute depth estimation
            depth_scale_factor: Scale factor for depth calibration
            calibration_distance: Known distance for calibration (in meters)
            model_path: Path to ZoeDepth model weights (.pt file)
        """
        self.model_type = model_type
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.distance_unit = distance_unit
        self.input_size = input_size
        self.enable_cache = enable_cache
        self.enable_result_cache = enable_result_cache
        self.cache_size = cache_size
        
        # Absolute depth configuration
        self.enable_absolute_depth = enable_absolute_depth
        self.depth_scale_factor = depth_scale_factor
        self.calibration_distance = calibration_distance
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model: Optional[torch.nn.Module] = None # Added type hint
        self.model_loaded = False
        
        # Result cache
        self._result_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance tracking
        self._inference_times = []
        self._preprocess_times = []
        
        self.model_path = model_path  # 新增
        logger.info(f"ZoeDepthDistanceMeasurement initialized with model_type: {model_type}, device: {device}, input_size: {input_size}, Max depth: {max_depth}, Min depth: {min_depth}, Distance unit: {distance_unit}")
        logger.info(f"Result cache: {'enabled' if enable_result_cache else 'disabled'}")
        
        # 加载模型（不再使用离线备用方案）
        self.load_model()
    

    def load_model(self):
        try:
            cache_key = f"{self.model_type}_{self.device}"
            if self.enable_cache and cache_key in self._model_cache:
                self.model = self._model_cache[cache_key]
                self.model_loaded = True
                logger.info(f"✅ ZoeDepth model {self.model_type} loaded from cache")
                return

            logger.info(f"Loading ZoeDepth model: {self.model_type}")

            # 仅示例  — 你已在 __init__ 里将 models 目录加入 sys.path
            project_root = Path(__file__).parent.parent
            if self.model_path is not None:
                local_weights = Path(self.model_path)
            else:
                local_weights = project_root / "models" / "ZoeD_M12_NK.pt"

            # 强制使用 zoedepth_nk 结构
            config = get_config("zoedepth_nk", "infer")
            config.pretrained_resource = f"local::{local_weights}"

            # 其他自定义配置（如输入尺寸）
            config.input_width, config.input_height = self.input_size
            config.absolute_depth = self.enable_absolute_depth
            config.scale_factor   = self.depth_scale_factor

            self.model = build_model(config)
            self.model.to(self.device).eval()

            if self.enable_cache:
                self._model_cache[cache_key] = self.model

            self.model_loaded = True
            logger.info("✅ ZoeDepth model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load ZoeDepth model: {e}", exc_info=True)
            raise   
    
    def ensure_model_loaded(self):
            """Ensure model is loaded before use"""
            if not self.model_loaded:
                self.load_model()
    
    def _fast_preprocess(self, image: np.ndarray) -> np.ndarray:
        """快速图像预处理"""
        start_time = time.time()
        
        # 快速尺寸调整
        if image.shape[:2] != self.input_size[::-1]:
            # 使用更快的插值方法
            image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # 快速颜色转换
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        preprocess_time = time.time() - start_time
        self._preprocess_times.append(preprocess_time)
        
        return rgb_image
    
    def _generate_image_hash(self, image: np.ndarray) -> str:
        """生成图像哈希用于缓存"""
        # 使用简单的哈希方法，平衡速度和准确性
        return str(hash(image.tobytes()))
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for faster inference (backward compatibility)"""
        return self._fast_preprocess(image)
    
    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth map for entire image (optimized version)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Depth map as numpy array in meters
        """
        self.ensure_model_loaded()
        
        # 快速预处理
        rgb_image = self._fast_preprocess(image)
        
        # 检查缓存
        if self.enable_result_cache:
            image_hash = self._generate_image_hash(rgb_image)
            if image_hash in self._result_cache:
                self._cache_hits += 1
                return self._result_cache[image_hash]
            self._cache_misses += 1
        
        # 记录推理开始时间
        inference_start = time.time()
        
        # 转换为PIL Image (优化版本)
        from PIL import Image
        pil_image = Image.fromarray(rgb_image)
        
        # 优化的推理
        with torch.no_grad():
            assert self.model is not None, "Model must be loaded before predicting depth."
            depth_tensor = self.model.infer_pil(pil_image) # type: ignore
            depth_numpy = depth_tensor
        
        # 快速后处理
        depth_map = np.clip(depth_numpy, self.min_depth, self.max_depth)
        
        # 应用绝对深度处理
        if self.enable_absolute_depth:
            # 应用深度尺度因子
            depth_map = depth_map * self.depth_scale_factor
            
            # 如果有校准距离，进行额外校准
            if self.calibration_distance:
                # 使用校准距离调整深度
                median_depth = np.median(depth_map)
                if median_depth > 0:
                    calibration_scale = self.calibration_distance / median_depth
                    depth_map = depth_map * calibration_scale
                    logger.debug(f"应用校准尺度: {calibration_scale:.3f}")
        
        # 调整回原始尺寸（如果需要）
        if rgb_image.shape[:2] != image.shape[:2]:
            depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # 记录推理时间
        inference_time = time.time() - inference_start
        self._inference_times.append(inference_time)
        
        # 缓存结果
        if self.enable_result_cache:
            if len(self._result_cache) >= self.cache_size:
                # 简单的LRU：删除第一个条目
                self._result_cache.pop(next(iter(self._result_cache)))
            self._result_cache[image_hash] = depth_map
        
        return depth_map
    
    def predict_depth_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict depth for multiple images in batch (optimized version)
        
        Args:
            images: List of input images (BGR format)
            
        Returns:
            List of depth maps as numpy arrays
        """
        if not images:
            return []
        
        self.ensure_model_loaded()
        
        # 批量预处理
        processed_images = []
        original_sizes = []
        
        for img in images:
            rgb_image = self._fast_preprocess(img)
            processed_images.append(rgb_image)
            original_sizes.append((img.shape[1], img.shape[0]))
        
        # 批量推理
        depth_maps = []
        with torch.no_grad():
            for i, rgb_image in enumerate(processed_images):
                # 检查缓存
                if self.enable_result_cache:
                    image_hash = self._generate_image_hash(rgb_image)
                    if image_hash in self._result_cache:
                        depth_map = self._result_cache[image_hash]
                    else:
                        # 推理
                        from PIL import Image
                        pil_image = Image.fromarray(rgb_image)
                        with torch.no_grad():
                            assert self.model is not None, "Model must be loaded before predicting depth."
                            depth_tensor = self.model.infer_pil(pil_image) # type: ignore
                        depth_numpy = depth_tensor
                        depth_map = np.clip(depth_numpy, self.min_depth, self.max_depth)
                        
                        # 应用绝对深度处理
                        if self.enable_absolute_depth:
                            depth_map = depth_map * self.depth_scale_factor
                            if self.calibration_distance:
                                median_depth = np.median(depth_map)
                                if median_depth > 0:
                                    calibration_scale = self.calibration_distance / median_depth
                                    depth_map = depth_map * calibration_scale
                        
                        # 缓存
                        if len(self._result_cache) < self.cache_size:
                            self._result_cache[image_hash] = depth_map
                else:
                    # 直接推理
                    from PIL import Image
                    pil_image = Image.fromarray(rgb_image)
                    with torch.no_grad():
                        assert self.model is not None, "Model must be loaded before predicting depth."
                        depth_tensor = self.model.infer_pil(pil_image) # type: ignore
                    depth_numpy = depth_tensor
                    depth_map = np.clip(depth_numpy, self.min_depth, self.max_depth)
                    
                    # 应用绝对深度处理
                    if self.enable_absolute_depth:
                        depth_map = depth_map * self.depth_scale_factor
                        if self.calibration_distance:
                            median_depth = np.median(depth_map)
                            if median_depth > 0:
                                calibration_scale = self.calibration_distance / median_depth
                                depth_map = depth_map * calibration_scale
                
                # 调整尺寸
                if processed_images[i].shape[:2] != images[i].shape[:2]:
                    depth_map = cv2.resize(depth_map, original_sizes[i], 
                                         interpolation=cv2.INTER_LINEAR)
                
                depth_maps.append(depth_map)
        
        return depth_maps
    
    def calculate_distance(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> DepthInfo:
        """
        Calculate distance for object in bounding box
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            DepthInfo object with distance information
        """
        # 获取深度图（失败则抛出异常）
        try:
            depth_map = self.predict_depth(image)
        except Exception as e:
            logger.error(f"⚠️ 深度预测失败: {e}")
            raise
        
        # 提取感兴趣区域
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(image.shape[1], x2)), int(min(image.shape[0], y2))

        roi_depth = depth_map[y1:y2, x1:x2]

        # 初始化变量，确保无论哪个分支都被赋值
        min_depth: float = self.max_depth
        max_depth: float = self.max_depth
        median_depth: float = self.max_depth
        distance_meters: float = self.max_depth
        depth_std: float = 0.0
        depth_confidence: float = 0.0

        if roi_depth.size == 0:
            # 保持默认值或设置特定值表示无效区域
            pass # 变量已在上方初始化
        else:
            # 快速统计计算
            min_depth = float(np.min(roi_depth))
            max_depth = float(np.max(roi_depth))
            median_depth = float(np.median(roi_depth))

            # 使用中位数作为距离估计
            distance_meters = median_depth

            # 快速置信度计算
            depth_std = float(np.std(roi_depth))
            depth_confidence = max(0.0, 1.0 - (depth_std / (max_depth - min_depth + 1e-6)))

            # 单位转换
            if self.distance_unit == "cm":
                distance_meters *= 100
                min_depth *= 100
                max_depth *= 100
                median_depth *= 100
            elif self.distance_unit == "inches":
                distance_meters *= 39.3701
                min_depth *= 39.3701
                max_depth *= 39.3701
                median_depth *= 39.3701
        
        return DepthInfo(
            distance_meters=distance_meters,
            depth_confidence=depth_confidence,
            depth_map_region=roi_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            median_depth=median_depth
        )

    def calculate_distances_batch(self, image: np.ndarray, detections: List[DetectionResult]) -> List[DepthInfo]:
        """
        Calculate distances for multiple objects in batch

        Args:
            image: Input image
            detections: List of detection results

        Returns:
            List of DepthInfo objects
        """
        if not detections:
            return []

        # 获取深度图（只生成一次）
        depth_map = self.predict_depth(image)

        results = []
        for detection in detections:
            bbox = detection.bbox
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(image.shape[1], x2)), int(min(image.shape[0], y2))

            roi_depth = depth_map[y1:y2, x1:x2]

            # 初始化变量，确保无论哪个分支都被赋值
            min_depth_current: float = self.max_depth
            max_depth_current: float = self.max_depth
            median_depth_current: float = self.max_depth
            distance_meters_current: float = self.max_depth
            depth_std_current: float = 0.0
            depth_confidence_current: float = 0.0

            if roi_depth.size == 0:
                # 保持默认值或设置特定值表示无效区域
                pass # 变量已在上方初始化
            else:
                # 快速统计计算
                min_depth_current = float(np.min(roi_depth))
                max_depth_current = float(np.max(roi_depth))
                median_depth_current = float(np.median(roi_depth))

                distance_meters_current = median_depth_current
                depth_std_current = float(np.std(roi_depth))
                depth_confidence_current = max(0.0, 1.0 - (depth_std_current / (max_depth_current - min_depth_current + 1e-6)))

                # 单位转换
                if self.distance_unit == "cm":
                    distance_meters_current *= 100
                    min_depth_current *= 100
                    max_depth_current *= 100
                    median_depth_current *= 100
                elif self.distance_unit == "inches":
                    distance_meters_current *= 39.3701
                    min_depth_current *= 39.3701
                    max_depth_current *= 39.3701
                    median_depth_current *= 39.3701

            depth_info = DepthInfo(
                distance_meters=distance_meters_current,
                depth_confidence=depth_confidence_current,
                depth_map_region=roi_depth,
                min_depth=min_depth_current,
                max_depth=max_depth_current,
                median_depth=median_depth_current
            )

            results.append(depth_info)

        return results

    def get_depth_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create visualization of depth map

        Args:
            depth_map: Depth map array

        Returns:
            Colorized depth map for visualization
        """
        # 快速归一化
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

        # 应用颜色映射
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)

        return depth_colored

    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        stats = {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "avg_inference_time": np.mean(self._inference_times) if self._inference_times else 0,
            "avg_preprocess_time": np.mean(self._preprocess_times) if self._preprocess_times else 0,
            "total_inferences": len(self._inference_times),
            "cache_size": len(self._result_cache)
        }
        return stats

    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_loaded = False
        self._result_cache.clear()
        logger.info("ZoeDepth distance measurement cleaned up")

    @classmethod
    def clear_cache(cls):
        """Clear model cache"""
        cls._model_cache.clear()
        cls._config_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ZoeDepth model cache cleared")


# Backward compatibility alias
DistanceMeasurement = ZoeDepthDistanceMeasurement
