"""
YOLOv8 Object Detection Module
Provides high-level interface for object detection using YOLOv8 models
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from ultralytics import YOLO

from config import ModelConfig, COCO_CLASSES, PROJECT_ROOT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionResult:
    """Container for detection results"""
    
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.distance = None  # Will be set by distance measurement
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        """Get width of bounding box"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get height of bounding box"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Get area of bounding box"""
        return self.width * self.height
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'distance': self.distance,
            'center': self.center,
            'width': self.width,
            'height': self.height,
            'area': self.area
        }


class YOLOv8Detector:
    """
    YOLOv8 Object Detection Engine
    
    Provides a high-level interface for object detection using YOLOv8 models.
    Supports multiple model variants and export formats for mobile deployment.
    """
    
    def __init__(self, 
                 model_variant: str = ModelConfig.DEFAULT_MODEL,
                 confidence_threshold: float = ModelConfig.CONFIDENCE_THRESHOLD,
                 nms_threshold: float = ModelConfig.NMS_THRESHOLD,
                 device: str = 'auto',
                 data_yaml: Optional[Union[str, Path]] = None):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_variant: Model size ('nano', 'small', 'medium', 'large', 'xlarge')
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_variant = model_variant
        # Optional YAML containing class names
        self.data_yaml = Path(data_yaml) if data_yaml is not None else None
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()

        # Prefer class names embedded in the weight file (Ultralytics stores them
        # in `model.names`) so that custom-trained models with non-COCO classes
        # are handled correctly. Fallback to COCO_NAMES for vanilla weights.
        try:
            names_attr = self.model.names  # Ultralytics returns list or dict
            if isinstance(names_attr, dict):
                # Dict[int, str] – ensure ordered by key
                self.class_names = [names_attr[k] for k in sorted(names_attr.keys())]
            else:
                self.class_names = list(names_attr)
        except Exception:
            self.class_names = COCO_CLASSES  # Fallback for safety
        
        # If a YAML with names is supplied, override
        if self.data_yaml is not None:
            try:
                import yaml  # Local import to avoid hard dependency at top level
                if self.data_yaml.exists():
                    with self.data_yaml.open('r', encoding='utf-8') as f:
                        data_cfg = yaml.safe_load(f)
                    names = data_cfg.get('names') if data_cfg else None
                    if names:
                        if isinstance(names, dict):
                            names = [names[k] for k in sorted(names.keys())]
                        self.class_names = list(names)
                        logger.info(f"Class names overridden from YAML: {self.data_yaml} (nc={len(self.class_names)})")
                    else:
                        logger.warning(f"No 'names' field found in YAML: {self.data_yaml}, keeping existing class names")
                else:
                    logger.warning(f"data_yaml path not found: {self.data_yaml}, keeping existing class names")
            except Exception as e:
                logger.warning(f"Failed to load data_yaml '{self.data_yaml}': {e}")

        logger.info(f"YOLOv8 detector initialized with {model_variant} model (classes={len(self.class_names)})")
    
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model.

        The method supports two usage patterns:

        1. Pre-defined model variants (e.g. "nano", "small"), which are looked up
           in `ModelConfig.MODEL_VARIANTS` and会 be downloaded / cached automatically.
        2. Explicit weight file path provided by the caller (e.g.
           "runs/detect/train/weights/best.pt"). If the supplied string looks like a
           file path and存在, load it directly.
        """

        variant_or_path = self.model_variant

        # Case 1 ‑ treat as direct file path when file exists
        potential_path = Path(variant_or_path)
        if potential_path.exists():
            logger.info(f"Loading custom YOLOv8 weights from path: {potential_path}")
            return YOLO(str(potential_path))

        # Case 2 ‑ treat as pre-defined variant name
        model_name = ModelConfig.MODEL_VARIANTS.get(variant_or_path)
        if not model_name:
            raise ValueError(f"Unknown model variant or weight path not found: {variant_or_path}")

        # Look for cached copy under models directory
        model_path = PROJECT_ROOT / "models" / model_name

        if model_path.exists():
            logger.info(f"Loading cached model: {model_path}")
            return YOLO(str(model_path))

        # Otherwise download via ultralytics which will handle caching in ~/.cache
        logger.info(f"Downloading YOLOv8 model variant '{variant_or_path}' ({model_name})")
        model = YOLO(model_name)

        try:
            # Save a copy inside project for offline reuse
            model.save(str(model_path))
            logger.info(f"Model cached locally at: {model_path}")
        except Exception as e:
            logger.warning(f"Could not cache model to {model_path}: {e}")

        return model
    
    def detect(self, 
               source: Union[str, np.ndarray, Path],
               save_results: bool = False,
               show_results: bool = False) -> List[DetectionResult]:
        """
        Perform object detection on image/video
        
        Args:
            source: Input source (image path, numpy array, or video path)
            save_results: Whether to save detection results
            show_results: Whether to display results
            
        Returns:
            List of DetectionResult objects
        """
        try:
            # Run inference
            results = self.model(
                source,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                device=self.device,
                save=save_results,
                show=show_results,
                verbose=False  # suppress per-frame inference prints
            )
            
            # Process results
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        if cls_id < len(self.class_names):
                            detection = DetectionResult(
                                bbox=box.tolist(),
                                confidence=float(conf),
                                class_id=int(cls_id),
                                class_name=self.class_names[cls_id]
                            )
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return []
    
    def detect_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in a single frame (optimized for real-time processing)
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of DetectionResult objects
        """
        return self.detect(frame)
    
    def export_model(self, 
                     format: str = 'onnx',
                     optimize: bool = True,
                     half: bool = False,
                     dynamic: bool = False) -> str:
        """
        Export model for mobile deployment
        
        Args:
            format: Export format ('onnx', 'tflite', 'coreml', 'torchscript')
            optimize: Whether to optimize the model
            half: Whether to use half precision
            dynamic: Whether to use dynamic batch size
            
        Returns:
            Path to exported model
        """
        try:
            export_path = self.model.export(
                format=format,
                optimize=optimize,
                half=half,
                dynamic=dynamic
            )
            logger.info(f"Model exported to: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Model export failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'variant': self.model_variant,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'num_classes': len(self.class_names),
            'input_size': ModelConfig.INPUT_SIZE
        }
    
    def update_thresholds(self, confidence: float = None, nms: float = None):
        """Update detection thresholds"""
        if confidence is not None:
            self.confidence_threshold = confidence
        if nms is not None:
            self.nms_threshold = nms
        
        logger.info(f"Updated thresholds - Confidence: {self.confidence_threshold}, NMS: {self.nms_threshold}")


def draw_detections(image: np.ndarray,
                   detections: List[DetectionResult],
                   show_confidence: bool = True,
                   show_distance: bool = True,
                   show_speed: bool = True,
                   font_scale: float = 0.6,
                   thickness: int = 2) -> np.ndarray:
    """
    Draw detection results on image

    Args:
        image: Input image
        detections: List of detection results
        show_confidence: Whether to show confidence scores
        show_distance: Whether to show distance measurements
        show_speed: Whether to show speed measurements
        font_scale: Font scale for text
        thickness: Line thickness for boxes

    Returns:
        Image with drawn detections
    """
    result_image = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        
        # Generate color based on class ID
        color = (
            int(detection.class_id * 50 % 255),
            int(detection.class_id * 80 % 255),
            int(detection.class_id * 120 % 255)
        )
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label_parts = [detection.class_name]

        if show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")

        if show_distance and detection.distance is not None:
            label_parts.append(f"{detection.distance:.1f}m")

        # Add speed information if available (but not ID as requested)
        if show_speed and hasattr(detection, 'speed_info') and detection.speed_info:
            speed_kmh = detection.speed_info.speed_kmh
            if speed_kmh > 0.1:  # Only show if speed is significant
                label_parts.append(f"{speed_kmh:.1f}km/h")

        label = " | ".join(label_parts)
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw text background
        cv2.rectangle(
            result_image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            result_image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
    
    return result_image
