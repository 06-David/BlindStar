"""
MiDaS Distance Measurement Module
Uses MiDaS depth estimation model to calculate accurate object distances
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

from .detector import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class DepthInfo:
    """Depth information for an object"""
    distance_meters: float
    depth_confidence: float
    depth_map_region: np.ndarray
    min_depth: float
    max_depth: float
    median_depth: float


class MiDaSDistanceMeasurement:
    """
    MiDaS-based distance measurement calculator
    
    Uses MiDaS depth estimation model to calculate accurate object distances
    from monocular images. Provides more accurate depth estimation compared
    to geometric methods.
    """
    
    def __init__(self, 
                 model_type: str = "MiDaS_small",
                 device: str = "auto",
                 max_depth: float = 10.0,
                 min_depth: float = 0.1,
                 distance_unit: str = "meters"):
        """
        Initialize MiDaS distance measurement calculator
        
        Args:
            model_type: MiDaS model variant ('MiDaS_small', 'MiDaS', 'DPT_Large', etc.)
            device: Device to use ('auto', 'cpu', 'cuda')
            max_depth: Maximum depth in meters
            min_depth: Minimum depth in meters
            distance_unit: Unit for distance measurements ('meters', 'cm', 'inches')
        """
        self.model_type = model_type
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.distance_unit = distance_unit
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = None
        self.transform = None
        self.model_loaded = False
        
        logger.info(f"MiDaS distance measurement initialized with model: {self.model_type}")
        logger.info(f"Device: {self.device}, Distance unit: {self.distance_unit}")
    
    def load_model(self):
        """Load MiDaS model and transforms"""
        try:
            # Import MiDaS
            import torch.hub
            
            logger.info(f"Loading MiDaS model: {self.model_type}")
            
            # Load model from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            elif self.model_type == "MiDaS_small":
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.default_transform
            
            self.model_loaded = True
            logger.info(f"✅ MiDaS model {self.model_type} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load MiDaS model: {e}")
            logger.info("💡 Installing MiDaS dependencies...")
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "timm"])
                logger.info("✅ MiDaS dependencies installed, please restart")
            except:
                logger.error("❌ Failed to install dependencies")
            raise e
    
    def ensure_model_loaded(self):
        """Ensure model is loaded before use"""
        if not self.model_loaded:
            self.load_model()
    
    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth map for entire image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Depth map as numpy array
        """
        self.ensure_model_loaded()
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_tensor = self.transform(rgb_image).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_tensor)
            
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy and normalize
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth values to physical distances
        depth_map = self._normalize_depth(depth_map)
        
        return depth_map
    
    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to physical distances
        
        Args:
            depth_map: Raw depth prediction
            
        Returns:
            Normalized depth map in meters
        """
        # MiDaS outputs inverse depth, so we need to invert
        # Normalize to 0-1 range first
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max > depth_min:
            normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth_map)
        
        # Convert to distance (invert and scale)
        # Higher values in normalized map = closer objects
        distances = self.min_depth + (1.0 - normalized) * (self.max_depth - self.min_depth)
        
        return distances
    
    def calculate_distance(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> DepthInfo:
        """
        Calculate distance for object in bounding box
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            DepthInfo object with distance information
        """
        # Get depth map for entire image
        depth_map = self.predict_depth(image)
        
        # Extract region of interest
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(image.shape[1], x2)), int(min(image.shape[0], y2))

        roi_depth = depth_map[y1:y2, x1:x2]
        
        if roi_depth.size == 0:
            return DepthInfo(
                distance_meters=self.max_depth,
                depth_confidence=0.0,
                depth_map_region=np.array([]),
                min_depth=self.max_depth,
                max_depth=self.max_depth,
                median_depth=self.max_depth
            )
        
        # Calculate statistics
        min_depth = float(np.min(roi_depth))
        max_depth = float(np.max(roi_depth))
        median_depth = float(np.median(roi_depth))
        
        # Use median depth as primary distance estimate (more robust than mean)
        distance_meters = median_depth
        
        # Calculate confidence based on depth consistency
        depth_std = np.std(roi_depth)
        depth_confidence = max(0.0, 1.0 - (depth_std / (max_depth - min_depth + 1e-6)))
        
        # Convert units if needed
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

        # Get depth map once for all objects
        depth_map = self.predict_depth(image)

        results = []
        for detection in detections:
            bbox = detection.bbox
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(image.shape[1], x2)), int(min(image.shape[0], y2))

            roi_depth = depth_map[y1:y2, x1:x2]

            if roi_depth.size == 0:
                depth_info = DepthInfo(
                    distance_meters=self.max_depth,
                    depth_confidence=0.0,
                    depth_map_region=np.array([]),
                    min_depth=self.max_depth,
                    max_depth=self.max_depth,
                    median_depth=self.max_depth
                )
            else:
                min_depth = float(np.min(roi_depth))
                max_depth = float(np.max(roi_depth))
                median_depth = float(np.median(roi_depth))

                distance_meters = median_depth
                depth_std = np.std(roi_depth)
                depth_confidence = max(0.0, 1.0 - (depth_std / (max_depth - min_depth + 1e-6)))

                # Convert units if needed
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

                depth_info = DepthInfo(
                    distance_meters=distance_meters,
                    depth_confidence=depth_confidence,
                    depth_map_region=roi_depth,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    median_depth=median_depth
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
        # Normalize for visualization
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

        # Apply colormap
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)

        return depth_colored

    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_loaded = False
        logger.info("MiDaS distance measurement cleaned up")


# Backward compatibility alias
DistanceMeasurement = MiDaSDistanceMeasurement
