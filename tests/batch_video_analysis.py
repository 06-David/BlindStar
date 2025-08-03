#!/usr/bin/env python3
r"""
批量视频帧分析工具
从本地文件夹读取视频文件并生成帧分析数据

默认配置:
- 输入目录: C:\MyCode\AI\BlindStar\input_videos
- 输出目录: C:\MyCode\AI\BlindStar\analyze_videos
- 分析日志目录: C:\MyCode\AI\BlindStar\analyze_videos

使用方法:
1. 直接运行（使用默认路径）:
   python batch_video_analysis.py

2. 指定输入目录:
   python batch_video_analysis.py "your_input_folder"

3. 完全自定义:
   python batch_video_analysis.py "input" --output-dir "output" --analysis-dir "logs"
"""

import os
import sys
import argparse
import logging

# Add project root to Python path to allow running from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pathlib import Path
from typing import List, Optional
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchVideoAnalyzer:
    """
    批量视频分析器
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str = "batch_analysis_output",
                 analysis_log_dir: str = "logs/batch_video_analysis",
                 model_weights: str | None = None,
                 data_yaml: str | None = None,
                 conf: float = 0.25):
        """
        初始化批量视频分析器
        
        Args:
            input_dir: 输入视频文件夹路径
            output_dir: 输出视频文件夹路径
            analysis_log_dir: 帧分析日志文件夹路径
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.analysis_log_dir = Path(analysis_log_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的视频格式
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        # 参数
        self.model_weights = model_weights
        self.data_yaml = data_yaml
        self.confidence_threshold = conf

        # 初始化组件
        self.detector = None
        self.distance_calc = None
        self.speed_calc = None
        self.video_processor = None
        
        logger.info(f"批量视频分析器初始化完成")
        logger.info(f"  输入目录: {self.input_dir}")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info(f"  分析日志目录: {self.analysis_log_dir}")
    
    def initialize_components(self):
        """初始化检测和分析组件"""
        logger.info("初始化检测和分析组件...")
        
        try:
            # 初始化检测器
            from core.detector import YOLOv8Detector
            self.detector = YOLOv8Detector(
                model_variant=self.model_weights or 'small',
                confidence_threshold=self.confidence_threshold,
                data_yaml=self.data_yaml)
            logger.info("✅ YOLO检测器初始化成功")
            
            # 初始化距离计算器
            from core.distance import DistanceMeasurement
            self.distance_calc = DistanceMeasurement()
            logger.info("✅ 距离计算器初始化成功")
            
            # 初始化速度计算器
            try:
                from core.speed_measurement import OpticalFlowSpeedMeasurement
                self.speed_calc = OpticalFlowSpeedMeasurement()
                logger.info("✅ 速度计算器初始化成功")
            except Exception as e:
                logger.warning(f"⚠️  速度计算器初始化失败: {e}")
                self.speed_calc = None
            
            # 初始化视频处理器
            from core.video_processor import VideoProcessor
            self.video_processor = VideoProcessor(
                detector=self.detector,
                distance_calculator=self.distance_calc,
                speed_calculator=self.speed_calc,
                frame_skip=1,
                output_fps=30
            )
            logger.info("✅ 视频处理器初始化成功")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            return False
    
    def find_video_files(self) -> List[Path]:
        """查找输入目录中的所有视频文件"""
        logger.info(f"搜索视频文件: {self.input_dir}")
        
        video_files = []
        
        if not self.input_dir.exists():
            logger.error(f"❌ 输入目录不存在: {self.input_dir}")
            return video_files
        
        # 递归搜索视频文件
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                video_files.append(file_path)
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        for video_file in video_files:
            logger.info(f"  - {video_file.name}")
        
        return video_files
    
    def process_single_video(self, 
                           video_path: Path, 
                           max_duration: Optional[float] = None) -> bool:
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            max_duration: 最大处理时长（秒）
            
        Returns:
            bool: 处理是否成功
        """
        if not self.video_processor:
            logger.error("❌ Video processor not initialized. Cannot process video.")
            return False

        logger.info(f"开始处理视频: {video_path.name}")
        
        try:
            # 生成输出文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"analyzed_{timestamp}_{video_path.stem}.mp4"
            output_path = self.output_dir / output_filename
            
            # 处理视频（使用帧分析功能）
            start_time = time.time()
            stats = self.video_processor.process_video_file_with_frame_analysis(
                str(video_path),
                str(output_path),
                max_duration=max_duration,
                frame_analysis_config={
                    "log_dir": str(self.analysis_log_dir),
                    "enable_json_log": True,
                    "enable_csv_log": True,
                    "enable_detailed_log": True,
                    "log_level": "INFO"
                }
            )
            processing_time = time.time() - start_time
            
            # 输出处理结果
            logger.info(f"✅ 视频处理完成: {video_path.name}")
            logger.info(f"   输出文件: {output_filename}")
            logger.info(f"   处理帧数: {stats.processed_frames}")
            logger.info(f"   检测总数: {stats.total_detections}")
            logger.info(f"   处理时间: {processing_time:.2f}s")
            logger.info(f"   平均FPS: {stats.average_fps:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 处理视频失败 {video_path.name}: {e}")
            return False
    
    def process_all_videos(self, max_duration: Optional[float] = None) -> dict:
        """
        处理所有视频文件
        
        Args:
            max_duration: 每个视频的最大处理时长（秒）
            
        Returns:
            dict: 处理结果统计
        """
        logger.info("开始批量处理视频...")

        video_files = self.find_video_files()
        if not video_files:
            logger.warning("⚠️  没有找到视频文件")
            return {"total": 0, "success": 0, "failed": 0}

        # 处理统计
        total_files = len(video_files)
        success_count = 0
        failed_count = 0
        
        # 逐个处理视频
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"\n--- 处理进度: {i}/{total_files} ---")
            
            if self.process_single_video(video_path, max_duration):
                success_count += 1
            else:
                failed_count += 1
        
        # 输出总结
        logger.info("\n" + "=" * 60)
        logger.info("批量处理完成!")
        logger.info(f"  总文件数: {total_files}")
        logger.info(f"  成功处理: {success_count}")
        logger.info(f"  处理失败: {failed_count}")
        logger.info(f"  成功率: {success_count/total_files*100:.1f}%")
        
        # 显示生成的分析文件
        self.show_analysis_files()
        
        return {
            "total": total_files,
            "success": success_count,
            "failed": failed_count
        }
    
    def show_analysis_files(self):
        """显示生成的帧分析文件"""
        logger.info("\n生成的帧分析文件:")
        
        if not self.analysis_log_dir.exists():
            logger.warning("⚠️  分析日志目录不存在")
            return
        
        # 查找分析文件
        json_files = list(self.analysis_log_dir.glob("frame_analysis_*.json"))
        csv_files = list(self.analysis_log_dir.glob("frame_analysis_*.csv"))
        log_files = list(self.analysis_log_dir.glob("detailed_analysis_*.log"))
        
        logger.info(f"  JSON文件: {len(json_files)}")
        logger.info(f"  CSV文件: {len(csv_files)}")
        logger.info(f"  详细日志: {len(log_files)}")
        
        # 显示最新的几个文件
        if json_files:
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"  最新JSON: {latest_json.name}")
        
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"  最新CSV: {latest_csv.name}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量视频帧分析工具')
    parser.add_argument('input_dir', nargs='?',
                       default=r'C:\MyCode\AI\BlindStar\input_videos',
                       help='输入视频文件夹路径 (默认: C:\\MyCode\\AI\\BlindStar\\input_videos)')
    parser.add_argument('--output-dir', default=r'C:\MyCode\AI\BlindStar\analyze_videos',
                       help='输出视频文件夹路径 (默认: C:\\MyCode\\AI\\BlindStar\\analyze_videos)')
    parser.add_argument('--analysis-dir', default=r'C:\MyCode\AI\BlindStar\analyze_videos',
                       help='帧分析日志文件夹路径 (默认: C:\\MyCode\\AI\\BlindStar\\analyze_videos)')
    parser.add_argument('--max-duration', type=float, default=None,
                       help='每个视频的最大处理时长（秒）')
    parser.add_argument('--weights', default=None,
                       help='自定义权重文件路径（默认: 官方yolov8s.pt）')
    parser.add_argument('--data', dest='data_yaml', default=None,
                       help='自定义数据集 YAML（含 names）文件路径')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='检测置信度阈值 (默认0.25)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志')

    args = parser.parse_args()

    # 显示启动信息
    logger.info("🚀 批量视频帧分析工具启动")
    logger.info(f"📁 输入目录: {args.input_dir}")
    logger.info(f"📁 输出目录: {args.output_dir}")
    logger.info(f"📁 分析日志目录: {args.analysis_dir}")

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 详细日志模式已启用")

    # 检查输入目录
    if not Path(args.input_dir).exists():
        logger.error(f"❌ 输入目录不存在: {args.input_dir}")
        logger.info(f"💡 请确保目录存在或创建目录: {args.input_dir}")
        logger.info(f"💡 您可以手动创建目录并放入视频文件，然后重新运行程序")
        return 1

    # 创建批量分析器
    analyzer = BatchVideoAnalyzer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        analysis_log_dir=args.analysis_dir,
        model_weights=args.weights,
        data_yaml=args.data_yaml,
        conf=args.conf
    )
    
    # 初始化组件
    if not analyzer.initialize_components():
        logger.error("❌ 组件初始化失败")
        return 1
    
    # 处理所有视频
    results = analyzer.process_all_videos(max_duration=args.max_duration)
    
    # 返回结果
    if results["failed"] == 0:
        logger.info("🎉 所有视频处理成功!")
        return 0
    else:
        logger.warning(f"⚠️  {results['failed']} 个视频处理失败")
        return 1

if __name__ == "__main__":
    exit(main())
