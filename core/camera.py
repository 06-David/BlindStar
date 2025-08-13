"""
Camera and Video Processing Module
Handles camera capture, video streaming, and real-time processing
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable, Generator, Union
import logging
from pathlib import Path

from config import CameraConfig

logger = logging.getLogger(__name__)


class CameraHandler:
    """
    Camera capture and video processing handler
    
    Supports webcam capture, video file processing, and real-time streaming
    with threading for improved performance.
    """
    
    def __init__(self, 
                 source: Union[int, str] = CameraConfig.DEFAULT_CAMERA_ID,
                 width: int = CameraConfig.FRAME_WIDTH,
                 height: int = CameraConfig.FRAME_HEIGHT,
                 fps: int = CameraConfig.FPS,
                 buffer_size: int = CameraConfig.BUFFER_SIZE,
                 threading_enabled: bool = CameraConfig.THREADING_ENABLED):
        """
        Initialize camera handler
        
        Args:
            source: Camera ID (int) or video file path (str)
            width: Frame width
            height: Frame height
            fps: Frames per second
            buffer_size: Buffer size for frame queue
            threading_enabled: Whether to use threading for capture
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.threading_enabled = threading_enabled
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size) if threading_enabled else None
        self.capture_thread = None
        self.is_running = False
        self.is_camera = isinstance(source, int)
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = None
        
        logger.info(f"Camera handler initialized for source: {source}")
    
    def start(self) -> bool:
        """
        Start camera capture
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera/video: {self.source}")
                return False
            
            # Set camera properties for webcam
            if self.is_camera:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera opened - Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            self.is_running = True
            self.start_time = time.time()
            
            # Start capture thread if threading is enabled
            if self.threading_enabled:
                self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
                self.capture_thread.start()
                logger.info("Capture thread started")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear frame queue
        if self.frame_queue:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
        
        logger.info("Camera stopped")
    
    def _capture_frames(self):
        """Internal method for threaded frame capture"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                if self.is_camera:
                    logger.warning("Failed to read frame from camera")
                    continue
                else:
                    logger.info("End of video file reached")
                    break
            
            # Resize frame if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
                self.frame_count += 1
            except queue.Full:
                # Drop oldest frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame
        
        Returns:
            Frame as numpy array or None if failed
        """
        if not self.is_running or not self.cap:
            return None
        
        if self.threading_enabled:
            # Get frame from queue
            try:
                frame = self.frame_queue.get(timeout=1.0)
                return frame
            except queue.Empty:
                logger.warning("Frame queue empty")
                return None
        else:
            # Direct capture
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            # Resize if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.frame_count += 1
            return frame
    
    def get_frame_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Get frame generator for continuous processing
        
        Yields:
            Frames as numpy arrays
        """
        while self.is_running:
            frame = self.read_frame()
            if frame is not None:
                yield frame
            else:
                if not self.is_camera:  # End of video file
                    break
                time.sleep(0.01)  # Small delay for camera
    
    def get_properties(self) -> dict:
        """Get camera/video properties"""
        if not self.cap:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else self.frame_count,
            'is_camera': self.is_camera,
            'is_running': self.is_running
        }
    
    def get_statistics(self) -> dict:
        """Get capture statistics"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_captured': self.frame_count,
            'frames_dropped': self.dropped_frames,
            'elapsed_time': elapsed_time,
            'actual_fps': actual_fps,
            'drop_rate': self.dropped_frames / max(1, self.frame_count) * 100
        }


class VideoProcessor:
    """
    Video processing pipeline for object detection
    
    Handles video input/output, frame processing, and result visualization
    """
    
    def __init__(self, 
                 detector,
                 distance_calculator=None,
                 output_path: Optional[str] = None,
                 show_display: bool = True,
                 save_results: bool = False):
        """
        Initialize video processor
        
        Args:
            detector: Object detector instance
            distance_calculator: Distance measurement instance
            output_path: Path to save output video
            show_display: Whether to show live display
            save_results: Whether to save detection results
        """
        self.detector = detector
        self.distance_calculator = distance_calculator
        self.output_path = output_path
        self.show_display = show_display
        self.save_results = save_results
        
        self.video_writer = None
        self.processing_stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'processing_time': 0
        }
    
    def process_video(self, 
                     camera_handler: CameraHandler,
                     frame_callback: Optional[Callable] = None) -> dict:
        """
        Process video stream with object detection
        
        Args:
            camera_handler: Camera handler instance
            frame_callback: Optional callback for each processed frame
            
        Returns:
            Processing statistics
        """
        if not camera_handler.start():
            logger.error("Failed to start camera")
            return self.processing_stats
        
        try:
            # Initialize video writer if output path is specified
            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.output_path,
                    fourcc,
                    camera_handler.fps,
                    (camera_handler.width, camera_handler.height)
                )
            
            start_time = time.time()
            
            # Process frames
            for frame in camera_handler.get_frame_generator():
                frame_start = time.time()
                
                # Perform detection
                detections = self.detector.detect_frame(frame)
                
                # Calculate distances if calculator is provided
                if self.distance_calculator and detections:
                    self.distance_calculator.measure_distances_batch(
                        detections, 
                        camera_handler.width, 
                        camera_handler.height
                    )
                
                # Draw results on frame
                from .detector import draw_detections
                result_frame = draw_detections(
                    frame, 
                    detections,
                    show_distance=self.distance_calculator is not None
                )
                
                # Update statistics
                self.processing_stats['frames_processed'] += 1
                self.processing_stats['total_detections'] += len(detections)
                self.processing_stats['processing_time'] += time.time() - frame_start
                
                # Save frame if video writer is available
                if self.video_writer:
                    self.video_writer.write(result_frame)
                
                # Show display
                if self.show_display:
                    cv2.imshow('Object Detection', result_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Call frame callback if provided
                if frame_callback:
                    frame_callback(result_frame, detections)
            
            # Calculate final statistics
            total_time = time.time() - start_time
            self.processing_stats['total_time'] = total_time
            self.processing_stats['avg_fps'] = (
                self.processing_stats['frames_processed'] / total_time 
                if total_time > 0 else 0
            )
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during video processing: {e}")
        finally:
            # Cleanup
            camera_handler.stop()
            
            if self.video_writer:
                self.video_writer.release()
            
            if self.show_display:
                cv2.destroyAllWindows()
        
        return self.processing_stats
    
    def process_single_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, detections)
        """
        # Perform detection
        detections = self.detector.detect_frame(frame)
        
        # Calculate distances if calculator is provided
        if self.distance_calculator and detections:
            height, width = frame.shape[:2]
            self.distance_calculator.measure_distances_batch(
                detections, width, height
            )
        
        # Draw results
        from .detector import draw_detections
        result_frame = draw_detections(
            frame, 
            detections,
            show_distance=self.distance_calculator is not None
        )
        
        return result_frame, detections


def create_video_from_images(image_paths: list, 
                           output_path: str,
                           fps: int = 30,
                           size: Optional[tuple] = None) -> bool:
    """
    Create video from list of images
    
    Args:
        image_paths: List of image file paths
        output_path: Output video path
        fps: Frames per second
        size: Video size (width, height), auto-detected if None
        
    Returns:
        True if successful
    """
    if not image_paths:
        return False
    
    try:
        # Get size from first image if not specified
        if size is None:
            first_img = cv2.imread(str(image_paths[0]))
            if first_img is None:
                return False
            height, width = first_img.shape[:2]
            size = (width, height)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        # Add images to video
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                if img.shape[:2][::-1] != size:
                    img = cv2.resize(img, size)
                video_writer.write(img)
        
        video_writer.release()
        logger.info(f"Video created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        return False
