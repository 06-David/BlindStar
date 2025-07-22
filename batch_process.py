#!/usr/bin/env python3
"""
BlindStar批量处理工具
提供视频和图像的批量处理功能
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.batch_processor import (
    BatchVideoProcessor, 
    BatchImageProcessor, 
    BatchProcessingConfig,
    BatchProcessingResult
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_config_from_args(args) -> BatchProcessingConfig:
    """从命令行参数创建批量处理配置"""
    return BatchProcessingConfig(
        yolo_model=args.yolo_model,
        midas_model=args.midas_model,
        device=args.device,
        confidence_threshold=args.confidence,
        max_frames_per_video=args.max_frames,
        enable_distance=not args.no_distance,
        output_base_dir=args.output_dir,
        parallel_processing=args.parallel,
        max_workers=args.workers,
        generate_output_video=getattr(args, 'generate_video', True)
    )


def progress_callback(current: int, total: int, filename: str):
    """进度回调函数"""
    progress = (current / total) * 100
    print(f"进度: {progress:.1f}% ({current}/{total}) - 正在处理: {filename}")


def process_videos(args) -> int:
    """处理视频批量任务"""
    logger.info(f"开始批量视频处理: {args.input}")
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 创建批量视频处理器
    processor = BatchVideoProcessor(config)
    
    try:
        # 执行批量处理
        result = processor.process_directory(
            directory=args.input,
            recursive=args.recursive,
            progress_callback=progress_callback if args.show_progress else None
        )
        
        # 保存结果报告
        if args.save_report:
            save_batch_report(result, args.output_dir, "video_batch_report")
        
        return 0 if result.failed_files == 0 else 1
    
    except Exception as e:
        logger.error(f"批量视频处理失败: {e}")
        return 1


def process_images(args) -> int:
    """处理图像批量任务"""
    logger.info(f"开始批量图像处理: {args.input}")
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 创建批量图像处理器
    processor = BatchImageProcessor(config)
    
    try:
        # 执行批量处理
        result = processor.process_directory(
            directory=args.input,
            recursive=args.recursive,
            save_results=args.save_results,
            progress_callback=progress_callback if args.show_progress else None
        )
        
        # 保存结果报告
        if args.save_report:
            save_batch_report(result, args.output_dir, "image_batch_report")
        
        return 0 if result.failed_files == 0 else 1
    
    except Exception as e:
        logger.error(f"批量图像处理失败: {e}")
        return 1


def save_batch_report(result: BatchProcessingResult, 
                     output_dir: str, 
                     report_name: str):
    """保存批量处理报告"""
    import json
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建报告数据
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": result.total_files,
            "successful_files": result.successful_files,
            "failed_files": result.failed_files,
            "success_rate": (result.successful_files / result.total_files * 100) if result.total_files > 0 else 0,
            "total_processing_time": result.total_processing_time
        },
        "statistics": result.summary_stats,
        "results": result.results,
        "errors": result.error_files
    }
    
    # 保存JSON报告
    json_file = output_path / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"批量处理报告已保存: {json_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="BlindStar批量处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量处理视频
  python batch_process.py videos data/videos --type video
  
  # 批量处理图像并保存结果
  python batch_process.py images data/images --type image --save-results
  
  # 使用高精度模型处理视频
  python batch_process.py videos data/videos --yolo-model large --midas-model DPT_Large
  
  # 限制每个视频处理帧数（用于测试）
  python batch_process.py videos data/videos --max-frames 1000
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='处理类型')
    
    # 视频处理子命令
    video_parser = subparsers.add_parser('videos', help='批量处理视频')
    video_parser.add_argument('input', help='输入目录路径')
    
    # 图像处理子命令
    image_parser = subparsers.add_parser('images', help='批量处理图像')
    image_parser.add_argument('input', help='输入目录路径')
    image_parser.add_argument('--save-results', action='store_true',
                             help='保存标注结果图像')
    
    # 共同参数
    for subparser in [video_parser, image_parser]:
        # 模型配置
        subparser.add_argument('--yolo-model', default='small',
                              choices=['nano', 'small', 'medium', 'large', 'xlarge'],
                              help='YOLO模型变体')
        subparser.add_argument('--midas-model', default='MiDaS_small',
                              choices=['MiDaS_small', 'MiDaS', 'DPT_Large', 'DPT_Hybrid'],
                              help='MiDaS模型变体')
        subparser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                              help='使用的设备')
        subparser.add_argument('--confidence', type=float, default=0.6,
                              help='检测置信度阈值')
        
        # 处理配置
        subparser.add_argument('--no-distance', action='store_true',
                              help='禁用距离测量')
        subparser.add_argument('--recursive', action='store_true', default=True,
                              help='递归搜索子目录')
        subparser.add_argument('--output-dir', default='logs/batch_processing',
                              help='输出目录')
        
        # 进度和报告
        subparser.add_argument('--show-progress', action='store_true',
                              help='显示处理进度')
        subparser.add_argument('--save-report', action='store_true',
                              help='保存处理报告')
        
        # 并行处理（预留）
        subparser.add_argument('--parallel', action='store_true',
                              help='启用并行处理（实验性）')
        subparser.add_argument('--workers', type=int, default=1,
                              help='并行工作线程数')
    
    # 视频特有参数
    video_parser.add_argument('--max-frames', type=int, default=None,
                             help='每个视频最大处理帧数（默认处理全部）')
    video_parser.add_argument('--no-video-output', dest='generate_video',
                             action='store_false', default=True,
                             help='不生成标注视频输出')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 验证输入目录
    if not Path(args.input).exists():
        logger.error(f"输入目录不存在: {args.input}")
        return 1
    
    # 执行相应的处理
    try:
        if args.command == 'videos':
            return process_videos(args)
        elif args.command == 'images':
            return process_images(args)
        else:
            logger.error(f"未知命令: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        return 1
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
