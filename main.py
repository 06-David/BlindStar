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

    def __init__(self, source: int | str = 0, args: Optional[argparse.Namespace] = None):
        # 保存参数，延迟初始化
        if args is None:
            raise ValueError("'args' must be provided")

        self.args = args
        self.source = source
        
        # 从参数中获取模块配置
        modules = getattr(args, 'module', ['vision', 'voice', 'distance', 'poi', 'tracking', 'analysis'])
        self.enable_vision = 'vision' in modules
        self.enable_voice = 'voice' in modules
        self.enable_distance = 'distance' in modules
        self.enable_poi = 'poi' in modules
        self.enable_tracking = 'tracking' in modules
        self.enable_analysis = 'analysis' in modules
        
        # 延迟初始化组件
        self.vision = None
        self.cap = None
        self.running = False
        self.last_detection = []
        self.tts = None
        self.stt = None
        self.poi = None

    def initialize(self):
        """初始化所有组件"""
        # 初始化核心视觉引擎
        if self.enable_vision:
            self.vision = BlindStar(
                yolo_model=self.args.model,
                midas_model=self.args.midas_model,
                confidence_threshold=self.args.confidence,
                enable_distance=self.enable_distance,
                device=self.args.device,
            )
            self.vision.initialize()

        # 设置视频源
        self.cap = cv2.VideoCapture(self.source)

        # 语音和POI模块
        if self.enable_voice:
            self.tts = TTSEngine()
            self.stt = STTEngine()
            self.setup_voice_commands()

        # POI模块
        if self.enable_poi:
            # 初始化POI模块（需要替换为你的高德API Key）
            self.poi = POIQuery(api_key="717d9a827c0ac3521932d3ae59aebbfe")
            logging.info("POI模块初始化完成")

        # 初始提示
        if self.enable_voice and self.tts:
            self.tts.speak("系统已启动，请说'附近有什么'开始查询", blocking=True)

    def setup_voice_commands(self):
        """设置语音命令处理"""
        # 开始监听语音
        if self.stt:
            self.stt.start_listening(self.handle_voice_command)

    def handle_voice_command(self, command: str):
        """处理语音命令"""
        logging.info(f"收到语音命令: {command}")

        # 检查语音组件是否已初始化
        if not self.tts:
            return

        # 命令路由
        if "附近" in command or "查找" in command:
            if not self.enable_poi or not self.poi:
                self.tts.speak("POI查询功能未启用")
                return
                
            # 提取关键词
            keyword = "餐厅"  # 默认值
            for kw in ["餐厅", "咖啡", "超市", "商场", "银行", "医院", "地铁", "公交", "公园"]:
                if kw in command:
                    keyword = kw
                    break

            # 执行POI查询
            logging.info(f"开始查询POI，关键词: {keyword}")
            try:
                results = self.poi.search_nearby(keyword)
                logging.info(f"POI查询完成，找到 {len(results)} 个结果")
                response = self.poi.format_poi_result(results)
                logging.info(f"POI响应: {response}")
                self.tts.speak(response)
            except Exception as e:
                logging.error(f"POI查询失败: {e}")
                self.tts.speak("查询失败，请稍后重试")

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
        # 初始化组件
        self.initialize()
        
        self.running = True
        frame_count = 0
        last_analysis_time = time.time()

        while self.running:
            # 读取摄像头帧
            if not self.cap:
                break
            ret, frame = self.cap.read()
            if not ret:
                break

            # 控制视觉分析频率（避免过载）
            current_time = time.time()
            if current_time - last_analysis_time > 0.1:  # 约10FPS
                # 视觉分析
                detections_list = []
                if self.enable_vision and self.vision:
                    result = self.vision.analyze_frame(frame)

                    # 将结果转换为检测列表
                    from core.detector import DetectionResult
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
                if self.enable_voice and self.tts and detections_list:
                    # 将检测结果转换为TTS可用的格式
                    tts_detections = []
                    for detection in detections_list:
                        tts_detections.append({
                            'name': detection.class_name,
                            'distance': detection.distance,
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
                if self.enable_voice and self.tts:
                    self.tts.speak("已保存快照")

        # 清理资源
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.enable_voice and self.stt:
            self.stt.stop_listening()
        if self.vision:
            self.vision.cleanup()


class BlindStarCLI:
    """Command Line Interface for BlindStar with Voice Support"""

    def __init__(self):
        self.blindstar = None
        self.logger = logging.getLogger(__name__)

    def parse_arguments(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="BlindStar - 智能视觉辅助系统（带语音）",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  python main.py --source 0                    # 使用带语音的摄像头
  python main.py --source video.mp4            # 处理视频文件（输出到logs文件夹）
  python main.py --source image.jpg            # 处理单张图像（输出到logs文件夹）
  python main.py --source images/ --batch      # 处理图像目录
  python main.py --source 0 --model large      # 使用大型YOLO模型
  python main.py --source 0 --midas-model MiDaS # 使用标准MiDaS模型
  python main.py --module vision distance      # 只启用视觉和距离测量
  python main.py --module voice poi            # 只启用语音和POI查询
  python main.py --module vision tracking      # 启用视觉和物体跟踪

注意: 所有处理结果自动保存到logs文件夹中

            """
        )

        # 输入源
        parser.add_argument(
            '--source', '-s',
            type=str,
            default='0',
            help='输入源: 摄像头ID (0), 视频文件, 图像文件, 或图像目录'
        )

        # 模型配置
        parser.add_argument(
            '--model', '-m',
            type=str,
            default='small',
            choices=['nano', 'small', 'medium', 'large', 'xlarge'],
            help='YOLOv8模型变体'
        )

        parser.add_argument(
            '--midas-model',
            type=str,
            default='MiDaS_small',
            choices=['MiDaS_small', 'MiDaS', 'DPT_Large', 'DPT_Hybrid', 'DPT_SwinV2_L_384'],
            help='MiDaS深度估计模型变体'
        )

        parser.add_argument(
            '--confidence', '-c',
            type=float,
            default=0.6,
            help='检测置信度阈值'
        )

        # 模块选择
        parser.add_argument(
            '--module',
            nargs='+',
            choices=['vision', 'voice', 'distance', 'poi', 'tracking', 'analysis'],
            default=['vision', 'voice', 'distance', 'poi', 'tracking', 'analysis'],
            help='指定要启用的模块（默认全部启用）'
        )

        # 语音交互

        # 输出选项 - 所有输出自动保存到logs文件夹

        parser.add_argument(
            '--max-duration',
            type=float,
            help='视频最大处理时长（秒）'
        )

        # 设备选择
        parser.add_argument(
            '--device',
            type=str,
            default='auto',
            choices=['auto', 'cpu', 'cuda'],
            help='使用的处理设备'
        )

        # 日志设置
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='启用详细日志'
        )

        return parser.parse_args()

    def run_camera(self, args):
        """启动带语音的摄像头会话"""
        self.logger.info("正在启动带语音的摄像头检测...")
        app = BlindStarApp(
            source=int(args.source),
            args=args,
        )
        app.run()
        return True

    def run_video(self, args):
        """处理视频文件"""
        self.logger.info(f"正在处理视频: {args.source}")

        # 初始化BlindStar
        self.blindstar = BlindStar(
            yolo_model=args.model,
            midas_model=args.midas_model,
            confidence_threshold=args.confidence,
            enable_distance=True,
            device=args.device
        )

        if not self.blindstar.initialize():
            self.logger.error("BlindStar初始化失败")
            return False

        try:
            # 确保logs文件夹存在
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # 生成输出文件名
            source_name = Path(args.source).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"{source_name}_processed_{timestamp}.mp4"
            output_path = logs_dir / output_filename
            
            # 处理视频
            stats = self.blindstar.process_video(
                video_path=args.source,
                output_path=str(output_path),
                max_duration=args.max_duration,
            )

            self.logger.info(f"视频处理完成:")
            self.logger.info(f"  处理帧数: {stats.processed_frames}")
            self.logger.info(f"  总检测数: {stats.total_detections}")
            self.logger.info(f"  平均FPS: {stats.average_fps:.2f}")
            self.logger.info(f"  处理时间: {stats.processing_time:.2f}秒")

            return True

        except Exception as e:
            self.logger.error(f"视频处理失败: {e}")
            return False

        finally:
            self.blindstar.cleanup()

    def run_image(self, args):
        """处理单张图像"""
        self.logger.info(f"正在处理图像: {args.source}")

        # 初始化BlindStar
        self.blindstar = BlindStar(
            yolo_model=args.model,
            midas_model=args.midas_model,
            confidence_threshold=args.confidence,
            enable_distance=True,
            device=args.device
        )

        if not self.blindstar.initialize():
            self.logger.error("BlindStar初始化失败")
            return False

        try:
            # 确保logs文件夹存在
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # 生成输出文件名
            source_name = Path(args.source).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"{source_name}_detected_{timestamp}.jpg"
            output_path = logs_dir / output_filename
            
            # 处理图像
            result = self.blindstar.detect_image(
                image_path=args.source,
                save_result=True,
                output_path=str(output_path),
            )

            self.logger.info(f"图像处理完成:")
            self.logger.info(f"  检测数: {len(result['detections'])}")
            self.logger.info(f"  处理时间: {result['processing_time']:.3f}秒")

            # 打印检测详情
            for i, detection in enumerate(result['detections']):
                self.logger.info(f"  {i + 1}. {detection['class_name']} "
                                 f"(confidence: {detection['confidence']:.2f})")
                if detection.get('distance_meters'):
                    self.logger.info(f"      距离: {detection['distance_meters']:.1f}米")

            return True

        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return False

        finally:
            self.blindstar.cleanup()

    def run(self):
        """主运行方法"""
        args = self.parse_arguments()

        # 设置日志
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(log_level=log_level)  # 修复这里

        self.logger.info("BlindStar - 智能视觉辅助系统（带语音）")
        self.logger.info(f"模型: {args.model}, 置信度: {args.confidence}")

        # 确定源类型并运行相应方法
        source = args.source

        if source.isdigit():
            # 摄像头
            return self.run_camera(args)
        elif Path(source).is_file():
            # 检查是否为视频或图像
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            if Path(source).suffix.lower() in video_extensions:
                return self.run_video(args)
            else:
                return self.run_image(args)
        else:
            self.logger.error(f"无效的源: {source}")
            return False


def main():
    """主入口点"""
    cli = BlindStarCLI()

    try:
        success = cli.run()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n用户中断")
        return 0
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main())