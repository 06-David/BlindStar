#!/usr/bin/env python3
"""
单视频帧分析工具
处理单个视频文件并生成帧分析数据
"""

import argparse
import logging
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_video(video_path: str, 
                 output_path: str = None,
                 analysis_dir: str = "logs/single_video_analysis",
                 max_duration: float = None) -> bool:
    """
    分析单个视频文件
    
    Args:
        video_path: 输入视频文件路径
        output_path: 输出视频文件路径（可选）
        analysis_dir: 帧分析日志目录
        max_duration: 最大处理时长（秒）
        
    Returns:
        bool: 处理是否成功
    """
    video_path = Path(video_path)
    
    # 检查输入文件
    if not video_path.exists():
        logger.error(f"❌ 视频文件不存在: {video_path}")
        return False
    
    # 生成输出路径
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"analyzed_{timestamp}_{video_path.stem}.mp4"
    
    output_path = Path(output_path)
    
    # 创建分析日志目录
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始分析视频: {video_path.name}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"分析日志目录: {analysis_dir}")
    
    try:
        # 初始化组件
        logger.info("初始化检测组件...")
        
        from core.detector import YOLOv8Detector
        from core.distance import DistanceMeasurement
        from core.video_processor import VideoProcessor
        
        detector = YOLOv8Detector()
        distance_calc = DistanceMeasurement()
        
        # 尝试初始化速度计算器
        speed_calc = None
        try:
            from core.speed_measurement import OpticalFlowSpeedMeasurement
            speed_calc = OpticalFlowSpeedMeasurement()
            logger.info("✅ 速度计算器已启用")
        except Exception as e:
            logger.warning(f"⚠️  速度计算器未启用: {e}")
        
        # 创建视频处理器
        processor = VideoProcessor(
            detector=detector,
            distance_calculator=distance_calc,
            speed_calculator=speed_calc,
            frame_skip=1,
            output_fps=30
        )
        
        logger.info("✅ 所有组件初始化完成")
        
        # 处理视频（使用帧分析功能）
        logger.info("开始处理视频...")
        start_time = time.time()

        stats = processor.process_video_file_with_frame_analysis(
            str(video_path),
            str(output_path),
            max_duration=max_duration,
            frame_analysis_config={
                "log_dir": str(analysis_dir),
                "enable_json_log": True,
                "enable_csv_log": True,
                "enable_detailed_log": True,
                "log_level": "INFO"
            }
        )
        
        processing_time = time.time() - start_time
        
        # 输出结果
        logger.info("🎉 视频分析完成!")
        logger.info(f"处理统计:")
        logger.info(f"  总帧数: {stats.total_frames}")
        logger.info(f"  处理帧数: {stats.processed_frames}")
        logger.info(f"  检测总数: {stats.total_detections}")
        logger.info(f"  处理时间: {processing_time:.2f}s")
        logger.info(f"  平均FPS: {stats.average_fps:.2f}")
        
        # 显示生成的分析文件
        show_generated_files(analysis_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 视频分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_generated_files(analysis_dir: Path):
    """显示生成的分析文件"""
    logger.info("\n生成的帧分析文件:")
    
    # 查找分析文件
    json_files = list(analysis_dir.glob("frame_analysis_*.json"))
    csv_files = list(analysis_dir.glob("frame_analysis_*.csv"))
    log_files = list(analysis_dir.glob("detailed_analysis_*.log"))
    
    logger.info(f"  📄 JSON文件: {len(json_files)}")
    for json_file in json_files:
        file_size = json_file.stat().st_size
        logger.info(f"    - {json_file.name} ({file_size:,} bytes)")
    
    logger.info(f"  📊 CSV文件: {len(csv_files)}")
    for csv_file in csv_files:
        # 统计CSV行数
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            data_rows = len(lines) - 1  # 减去头部行
            logger.info(f"    - {csv_file.name} ({data_rows} 行数据)")
        except:
            logger.info(f"    - {csv_file.name}")
    
    logger.info(f"  📝 详细日志: {len(log_files)}")
    for log_file in log_files:
        logger.info(f"    - {log_file.name}")
    
    # 如果有数据，显示一些统计信息
    if json_files:
        try:
            import json
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary = data.get('session_summary', {})
            logger.info(f"\n📈 分析结果摘要:")
            logger.info(f"  处理帧数: {summary.get('frames_processed', 0)}")
            logger.info(f"  检测物体总数: {summary.get('total_objects_detected', 0)}")
            logger.info(f"  平均每帧物体数: {summary.get('average_objects_per_frame', 0):.2f}")
            logger.info(f"  处理FPS: {summary.get('processing_fps', 0):.2f}")
            
            # 显示检测到的物体类型
            frames = data.get('frames', [])
            object_types = set()
            for frame in frames:
                for obj in frame.get('objects', []):
                    object_types.add(obj.get('class_name'))
            
            if object_types:
                logger.info(f"  检测到的物体类型: {', '.join(sorted(object_types))}")
            
        except Exception as e:
            logger.warning(f"⚠️  无法读取分析结果: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='单视频帧分析工具')
    parser.add_argument('video_path', help='输入视频文件路径')
    parser.add_argument('--output', '-o', help='输出视频文件路径')
    parser.add_argument('--analysis-dir', default='logs/single_video_analysis',
                       help='帧分析日志目录 (默认: logs/single_video_analysis)')
    parser.add_argument('--max-duration', type=float, 
                       help='最大处理时长（秒）')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 分析视频
    success = analyze_video(
        video_path=args.video_path,
        output_path=args.output,
        analysis_dir=args.analysis_dir,
        max_duration=args.max_duration
    )
    
    if success:
        logger.info("✅ 视频分析成功完成!")
        return 0
    else:
        logger.error("❌ 视频分析失败!")
        return 1

if __name__ == "__main__":
    exit(main())
