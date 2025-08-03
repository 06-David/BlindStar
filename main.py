#!/usr/bin/env python3
"""
BlindStar - Intelligent Visual Assistance System
Main CLI Application with Voice Integration

A modular computer vision system with voice interaction and POI query capabilities.

Usage:
    python main.py --source 0                    # Webcam with voice
    python main.py --source video.mp4            # Video file
    python main.py --source image.jpg            # Single image
    python main.py --source images/              # Image directory
    python main.py --source 0 --disable-voice    # Webcam without voice
"""

import argparse
import sys
import cv2
import time
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core import BlindStar
from core.detector import draw_detections
from core.utils import setup_logging
from config import ModelConfig, LogConfig, VideoConfig, PerformanceConfig

# 导入新增的语音和POI模块
from core.tts_engine import TTSEngine
from core.stt_engine import STTEngine
from core.poi_query import POIQuery


# -----------------------------------------------------------------------------
# High-level application helpers
# -----------------------------------------------------------------------------


class BlindStarApp:
    """Realtime camera application (optionally with voice integration)."""

    def __init__(self, source: int | str = 0, enable_voice: bool = True, args: Optional[argparse.Namespace] = None):
        # 初始化核心视觉引擎
        if args is None:
            raise ValueError("'args' must be provided")

        self.vision = BlindStar(
            yolo_model=args.model,
            midas_model=args.midas_model,
            confidence_threshold=args.confidence,
            enable_distance=not args.no_distance,
            device=args.device,
        )
        self.vision.initialize()

        # 设置视频源
        self.source = source
        self.cap = cv2.VideoCapture(source)

        # 状态管理
        self.running = False
        self.last_detection = []
        self.enable_voice = enable_voice

        # 语音和POI模块
        if self.enable_voice:
            self.tts = TTSEngine()
            self.stt = STTEngine()

            # 初始化POI模块（需要替换为你的高德API Key）
            self.poi = POIQuery(api_key="717d9a827c0ac3521932d3ae59aebbfe")
            self.setup_voice_commands()

            # 初始提示
            self.tts.speak("系统已启动，请说'附近有什么'开始查询", blocking=True)

    def setup_voice_commands(self):
        """设置语音命令处理"""
        # 开始监听语音
        self.stt.start_listening(self.handle_voice_command)

    def handle_voice_command(self, command: str):
        """处理语音命令"""
        logging.info(f"收到语音命令: {command}")

        # 命令路由
        if "附近" in command or "查找" in command:
            # 提取关键词
            keyword = "餐厅"  # 默认值
            for kw in ["餐厅", "咖啡", "超市", "商场", "银行", "医院", "地铁", "公交", "公园"]:
                if kw in command:
                    keyword = kw
                    break

            # 执行POI查询
            results = self.poi.search_nearby(keyword)
            response = self.poi.format_poi_result(results)
            self.tts.speak(response)

        elif "停止" in command or "暂停" in command:
            # 暂停系统（演示用）
            self.tts.speak("系统暂停")

        elif "开始" in command or "继续" in command:
            # 恢复系统
            self.tts.speak("系统继续运行")

        elif "重复" in command or "再说" in command:
            # 重复上次检测结果
            if self.last_detection:
                self.tts.speak_detection_result(self.last_detection)

        elif "帮助" in command:
            self.tts.speak("可用命令：附近餐厅，附近超市，停止播报，开始播报")

        elif "退出" in command or "关闭" in command:
            self.running = False

    def run(self):
        """主运行循环"""
        self.running = True
        frame_count = 0
        last_analysis_time = time.time()

        while self.running:
            # 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                break

            # 控制视觉分析频率（避免过载）
            current_time = time.time()
            if current_time - last_analysis_time > 0.1:  # 约10FPS
                # 视觉分析
                result = self.vision.analyze_frame(frame)

                # 将结果转换为检测列表
                from core.detector import DetectionResult
                detections_list = []
                for d in result['detections']:
                    detection = DetectionResult(
                        bbox=d['bbox'],
                        confidence=d['confidence'],
                        class_id=d['class_id'],
                        class_name=d['class_name'],
                    )

                    # Optional distance
                    if d.get('distance_meters') is not None:
                        detection.distance = d['distance_meters']  # type: ignore[attr-defined]

                    detections_list.append(detection)

                self.last_detection = detections_list
                last_analysis_time = current_time

                # 语音播报检测结果（避障功能）
                if self.enable_voice:
                    # 将检测结果转换为TTS可用的格式
                    tts_detections = []
                    for detection in detections_list:
                        tts_detections.append({
                            'name': detection.class_name,
                            'distance': detection.distance_meters,
                            'position_x': detection.center[0] / frame.shape[1]  # 归一化位置
                        })
                    self.tts.speak_detection_result(tts_detections)

            # 显示结果
            display_frame = draw_detections(frame, self.last_detection)
            cv2.imshow('BlindStar - Camera Detection', display_frame)

            # 处理按键事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存快照
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"snapshot_{timestamp}.jpg", display_frame)
                logging.info(f"Snapshot saved: snapshot_{timestamp}.jpg")
                if self.enable_voice:
                    self.tts.speak("已保存快照")

        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()
        if self.enable_voice:
            self.stt.stop_listening()
        self.vision.cleanup()


class BlindStarCLI:
    """Command Line Interface for BlindStar with Voice Support"""

    def __init__(self):
        self.blindstar = None
        self.logger = logging.getLogger(__name__)

    def parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="BlindStar - Intelligent Visual Assistance System with Voice",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py --source 0                    # Use webcam with voice
  python main.py --source video.mp4            # Process video file
  python main.py --source image.jpg            # Process single image
  python main.py --source images/ --batch      # Process image directory
  python main.py --source 0 --model large      # Use large YOLO model
  python main.py --source 0 --midas-model MiDaS # Use standard MiDaS model
  python main.py --source 0 --no-distance      # Disable distance measurement
  python main.py --source 0 --save-video out.mp4  # Save output video
  python main.py --source 0 --disable-voice    # Disable voice interaction
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

        # Voice interaction
        parser.add_argument(
            '--disable-voice',
            action='store_true',
            help='Disable voice interaction (TTS and STT) for camera mode'
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
        """Start webcam session (voice optional)."""
        self.logger.info("Starting camera detection%s..." % (" with voice" if not args.disable_voice else ""))
        app = BlindStarApp(
            source=int(args.source),
            enable_voice=not args.disable_voice,
            args=args,
        )
        app.run()
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
                max_duration=args.max_duration,
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
                image_path=args.source,
                save_result=args.output is not None,
                output_path=args.output,
            )

            self.logger.info(f"Image processing completed:")
            self.logger.info(f"  Detections: {len(result['detections'])}")
            self.logger.info(f"  Processing time: {result['processing_time']:.3f}s")

            # Print detection details
            for i, detection in enumerate(result['detections']):
                self.logger.info(f"  {i + 1}. {detection['class_name']} "
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
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(log_level=log_level)  # 修复这里

        self.logger.info("BlindStar - Intelligent Visual Assistance System with Voice")
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