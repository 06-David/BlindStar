#!/usr/bin/env python3
"""
BlindStar - Intelligent Visual Assistance System
Main CLI Application

A modular computer vision system for object detection, distance measurement,
and motion analysis designed for visual assistance applications.

Usage:
    python main.py --source 0                    # Webcam
    python main.py --source video.mp4            # Video file
    python main.py --source image.jpg            # Single image
    python main.py --source images/              # Image directory
"""

import argparse
import sys
from pathlib import Path
import logging
import time
from typing import Optional, Union, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core import BlindStar
from core.detector import draw_detections
from core.utils import setup_logging
from config import ModelConfig, LogConfig, VideoConfig, PerformanceConfig


class BlindStarCLI:
    """Command Line Interface for BlindStar"""

    def __init__(self):
        self.blindstar = None
        self.logger = logging.getLogger(__name__)

    def parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="BlindStar - Intelligent Visual Assistance System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py --source 0                    # Use webcam
  python main.py --source video.mp4            # Process video file
  python main.py --source image.jpg            # Process single image
  python main.py --source images/ --batch      # Process image directory
  python main.py --source 0 --model large      # Use large YOLO model
  python main.py --source 0 --midas-model MiDaS # Use standard MiDaS model
  python main.py --source 0 --no-distance      # Disable distance measurement
  python main.py --source 0 --save-video out.mp4  # Save output video
            """
        )

        # Input source
        parser.add_argument(
            '--source', '-s',
            type=str,
            default='0',
            help='Input source: camera ID (0), video file, image file, or image directory'
        )

        # Model configuration
        parser.add_argument(
            '--model', '-m',
            type=str,
            default='small',
            choices=['nano', 'small', 'medium', 'large', 'xlarge'],
            help='YOLOv8 model variant'
        )

        parser.add_argument(
            '--midas-model',
            type=str,
            default='MiDaS_small',
            choices=['MiDaS_small', 'MiDaS', 'DPT_Large', 'DPT_Hybrid', 'DPT_SwinV2_L_384'],
            help='MiDaS depth estimation model variant'
        )

        parser.add_argument(
            '--confidence', '-c',
            type=float,
            default=0.6,
            help='Confidence threshold for detections'
        )

        # Feature toggles
        parser.add_argument(
            '--no-distance',
            action='store_true',
            help='Disable distance measurement'
        )

        parser.add_argument(
            '--no-speed',
            action='store_true',
            help='Disable speed measurement'
        )

        parser.add_argument(
            '--no-analysis',
            action='store_true',
            help='Disable frame analysis'
        )

        # Output options
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output path for processed video/image'
        )

        parser.add_argument(
            '--max-duration',
            type=float,
            help='Maximum processing duration for videos (seconds)'
        )

        # Device selection
        parser.add_argument(
            '--device',
            type=str,
            default='auto',
            choices=['auto', 'cpu', 'cuda'],
            help='Device to use for processing'
        )

        # Logging
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )

        return parser.parse_args()

    def run_camera(self, args):
        """Run camera detection"""
        self.logger.info("Starting camera detection...")

        # Initialize BlindStar
        self.blindstar = BlindStar(
            yolo_model=args.model,
            midas_model=args.midas_model,
            confidence_threshold=args.confidence,
            enable_distance=not args.no_distance,
            device=args.device
        )

        if not self.blindstar.initialize():
            self.logger.error("Failed to initialize BlindStar")
            return False

        # Start camera
        camera_id = int(args.source)
        if not self.blindstar.start_camera(camera_id):
            self.logger.error(f"Failed to start camera {camera_id}")
            return False

        self.logger.info("Camera started. Press 'q' to quit, 's' to save snapshot")

        try:
            while True:
                frame = self.blindstar.get_camera_frame()
                if frame is None:
                    continue

                # Analyze frame
                result = self.blindstar.analyze_frame(frame)

                # Draw detections (convert dict back to DetectionResult objects)
                from core.detector import DetectionResult
                detections_list = []
                for d in result['detections']:
                    detection = DetectionResult(
                        class_id=d['class_id'],
                        class_name=d['class_name'],
                        confidence=d['confidence'],
                        bbox=d['bbox'],
                        center=d['center'],
                        area=d['area']
                    )
                    # Add optional attributes
                    if d.get('distance_meters'):
                        detection.distance_meters = d['distance_meters']
                    if d.get('speed_mps'):
                        detection.speed_mps = d['speed_mps']
                    detections_list.append(detection)

                display_frame = draw_detections(frame, detections_list)

                # Show frame
                cv2.imshow('BlindStar - Camera Detection', display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"snapshot_{timestamp}.jpg", display_frame)
                    self.logger.info(f"Snapshot saved: snapshot_{timestamp}.jpg")

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")

        finally:
            self.blindstar.stop_camera()
            cv2.destroyAllWindows()

        return True

    def run_video(self, args):
        """Process video file"""
        self.logger.info(f"Processing video: {args.source}")

        # Initialize BlindStar
        self.blindstar = BlindStar(
            yolo_model=args.model,
            midas_model=args.midas_model,
            confidence_threshold=args.confidence,
            enable_distance=not args.no_distance,
            device=args.device
        )

        if not self.blindstar.initialize():
            self.logger.error("Failed to initialize BlindStar")
            return False

        try:
            # Process video
            stats = self.blindstar.process_video(
                video_path=args.source,
                output_path=args.output,
                max_duration=args.max_duration
            )

            self.logger.info(f"Video processing completed:")
            self.logger.info(f"  Processed frames: {stats.processed_frames}")
            self.logger.info(f"  Total detections: {stats.total_detections}")
            self.logger.info(f"  Average FPS: {stats.average_fps:.2f}")
            self.logger.info(f"  Processing time: {stats.processing_time:.2f}s")

            return True

        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            return False

        finally:
            self.blindstar.cleanup()

    def run_image(self, args):
        """Process single image"""
        self.logger.info(f"Processing image: {args.source}")

        # Initialize BlindStar
        self.blindstar = BlindStar(
            yolo_model=args.model,
            midas_model=args.midas_model,
            confidence_threshold=args.confidence,
            enable_distance=not args.no_distance,
            device=args.device
        )

        if not self.blindstar.initialize():
            self.logger.error("Failed to initialize BlindStar")
            return False

        try:
            # Process image
            result = self.blindstar.detect_image(
                image=args.source,
                save_result=args.output is not None,
                output_path=args.output
            )

            self.logger.info(f"Image processing completed:")
            self.logger.info(f"  Detections: {len(result['detections'])}")
            self.logger.info(f"  Processing time: {result['processing_time']:.3f}s")

            # Print detection details
            for i, detection in enumerate(result['detections']):
                self.logger.info(f"  {i+1}. {detection['class_name']} "
                               f"(confidence: {detection['confidence']:.2f})")
                if detection.get('distance_meters'):
                    self.logger.info(f"      Distance: {detection['distance_meters']:.1f}m")

            return True

        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return False

        finally:
            self.blindstar.cleanup()

    def run(self):
        """Main run method"""
        args = self.parse_arguments()

        # Setup logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(level=log_level)

        self.logger.info("BlindStar - Intelligent Visual Assistance System")
        self.logger.info(f"Model: {args.model}, Confidence: {args.confidence}")

        # Determine source type and run appropriate method
        source = args.source

        if source.isdigit():
            # Camera
            return self.run_camera(args)
        elif Path(source).is_file():
            # Check if it's a video or image
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            if Path(source).suffix.lower() in video_extensions:
                return self.run_video(args)
            else:
                return self.run_image(args)
        else:
            self.logger.error(f"Invalid source: {source}")
            return False


def main():
    """Main entry point"""
    import cv2

    cli = BlindStarCLI()

    try:
        success = cli.run()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
