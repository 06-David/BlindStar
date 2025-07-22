"""
Utility functions for the YOLOv8 Object Detection system
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import logging
import time
from datetime import datetime
import os

from .detector import DetectionResult

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        output_path: Output file path
        
    Returns:
        True if successful
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(output_path), image)
        if success:
            logger.info(f"Image saved: {output_path}")
        else:
            logger.error(f"Failed to save image: {output_path}")
        return success
    except Exception as e:
        logger.error(f"Error saving image {output_path}: {e}")
        return False


def resize_image(image: np.ndarray, 
                target_size: Tuple[int, int],
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if not maintain_aspect:
        return cv2.resize(image, target_size)
    
    # Calculate scaling factor to maintain aspect ratio
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas with target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the resized image on canvas
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def get_image_files(directory: Union[str, Path], 
                   extensions: List[str] = None) -> List[Path]:
    """
    Get list of image files in directory
    
    Args:
        directory: Directory path
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    directory = Path(directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def save_detections_json(detections: List[DetectionResult], 
                        output_path: Union[str, Path],
                        image_info: Optional[Dict] = None) -> bool:
    """
    Save detection results to JSON file
    
    Args:
        detections: List of detection results
        output_path: Output JSON file path
        image_info: Optional image information
        
    Returns:
        True if successful
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = {
            'timestamp': datetime.now().isoformat(),
            'image_info': image_info or {},
            'detections': [detection.to_dict() for detection in detections]
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Detections saved to JSON: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving detections to JSON: {e}")
        return False


def save_detections_csv(detections: List[DetectionResult], 
                       output_path: Union[str, Path]) -> bool:
    """
    Save detection results to CSV file
    
    Args:
        detections: List of detection results
        output_path: Output CSV file path
        
    Returns:
        True if successful
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare CSV data
        fieldnames = [
            'class_name', 'class_id', 'confidence', 
            'x1', 'y1', 'x2', 'y2', 'width', 'height', 
            'center_x', 'center_y', 'area', 'distance'
        ]
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for detection in detections:
                center_x, center_y = detection.center
                row = {
                    'class_name': detection.class_name,
                    'class_id': detection.class_id,
                    'confidence': detection.confidence,
                    'x1': detection.bbox[0],
                    'y1': detection.bbox[1],
                    'x2': detection.bbox[2],
                    'y2': detection.bbox[3],
                    'width': detection.width,
                    'height': detection.height,
                    'center_x': center_x,
                    'center_y': center_y,
                    'area': detection.area,
                    'distance': detection.distance
                }
                writer.writerow(row)
        
        logger.info(f"Detections saved to CSV: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving detections to CSV: {e}")
        return False


def load_detections_json(json_path: Union[str, Path]) -> Tuple[List[DetectionResult], Dict]:
    """
    Load detection results from JSON file
    
    Args:
        json_path: JSON file path
        
    Returns:
        Tuple of (detections, image_info)
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        detections = []
        for det_data in data.get('detections', []):
            detection = DetectionResult(
                bbox=det_data['bbox'],
                confidence=det_data['confidence'],
                class_id=det_data['class_id'],
                class_name=det_data['class_name']
            )
            detection.distance = det_data.get('distance')
            detections.append(detection)
        
        image_info = data.get('image_info', {})
        return detections, image_info
        
    except Exception as e:
        logger.error(f"Error loading detections from JSON: {e}")
        return [], {}


def create_detection_summary(detections: List[DetectionResult]) -> Dict:
    """
    Create summary statistics for detections
    
    Args:
        detections: List of detection results
        
    Returns:
        Summary dictionary
    """
    if not detections:
        return {'total_detections': 0}
    
    # Count by class
    class_counts = {}
    confidence_scores = []
    distances = []
    
    for detection in detections:
        # Count classes
        class_name = detection.class_name
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Collect confidence scores
        confidence_scores.append(detection.confidence)
        
        # Collect distances
        if detection.distance is not None:
            distances.append(detection.distance)
    
    summary = {
        'total_detections': len(detections),
        'unique_classes': len(class_counts),
        'class_counts': class_counts,
        'confidence_stats': {
            'mean': np.mean(confidence_scores),
            'min': np.min(confidence_scores),
            'max': np.max(confidence_scores),
            'std': np.std(confidence_scores)
        }
    }
    
    if distances:
        summary['distance_stats'] = {
            'mean': np.mean(distances),
            'min': np.min(distances),
            'max': np.max(distances),
            'std': np.std(distances),
            'count': len(distances)
        }
    
    return summary


def benchmark_detector(detector, test_images: List[str], num_runs: int = 5) -> Dict:
    """
    Benchmark detector performance
    
    Args:
        detector: Detector instance
        test_images: List of test image paths
        num_runs: Number of runs for averaging
        
    Returns:
        Benchmark results
    """
    if not test_images:
        logger.error("No test images provided")
        return {}
    
    logger.info(f"Benchmarking detector with {len(test_images)} images, {num_runs} runs each")
    
    total_time = 0
    total_detections = 0
    inference_times = []
    
    for run in range(num_runs):
        run_start = time.time()
        run_detections = 0
        
        for img_path in test_images:
            image = load_image(img_path)
            if image is None:
                continue
            
            start_time = time.time()
            detections = detector.detect_frame(image)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            run_detections += len(detections)
        
        run_time = time.time() - run_start
        total_time += run_time
        total_detections += run_detections
        
        logger.info(f"Run {run + 1}/{num_runs}: {run_time:.2f}s, {run_detections} detections")
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    results = {
        'total_images': len(test_images) * num_runs,
        'total_time': total_time,
        'total_detections': total_detections,
        'avg_inference_time': avg_inference_time,
        'min_inference_time': np.min(inference_times),
        'max_inference_time': np.max(inference_times),
        'std_inference_time': np.std(inference_times),
        'avg_fps': avg_fps,
        'avg_detections_per_image': total_detections / (len(test_images) * num_runs)
    }
    
    logger.info(f"Benchmark completed: {avg_fps:.1f} FPS, {avg_inference_time*1000:.1f}ms avg inference")
    return results


def create_detection_video(image_dir: Union[str, Path],
                          detector,
                          output_path: Union[str, Path],
                          fps: int = 30,
                          distance_calculator=None) -> bool:
    """
    Create detection video from images in directory
    
    Args:
        image_dir: Directory containing images
        detector: Detector instance
        output_path: Output video path
        fps: Frames per second
        distance_calculator: Optional distance calculator
        
    Returns:
        True if successful
    """
    try:
        # Get image files
        image_files = get_image_files(image_dir)
        if not image_files:
            logger.error(f"No images found in {image_dir}")
            return False
        
        logger.info(f"Creating detection video from {len(image_files)} images")
        
        # Initialize video writer
        first_image = load_image(image_files[0])
        if first_image is None:
            return False
        
        height, width = first_image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process each image
        for i, img_path in enumerate(image_files):
            image = load_image(img_path)
            if image is None:
                continue
            
            # Resize if needed
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            
            # Detect objects
            detections = detector.detect_frame(image)
            
            # Calculate distances if calculator provided
            if distance_calculator and detections:
                distance_calculator.measure_distances_batch(detections, width, height)
            
            # Draw detections
            from .detector import draw_detections
            result_frame = draw_detections(
                image,
                detections,
                show_distance=distance_calculator is not None,
                show_speed=False  # Utils function doesn't have speed tracking
            )
            
            # Write frame
            video_writer.write(result_frame)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")
        
        video_writer.release()
        logger.info(f"Detection video created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating detection video: {e}")
        return False


def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = True,
                      create_parent: bool = False) -> bool:
    """
    Validate file path
    
    Args:
        file_path: File path to validate
        must_exist: Whether file must exist
        create_parent: Whether to create parent directory
        
    Returns:
        True if valid
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Invalid file path {file_path}: {e}")
        return False
