#!/usr/bin/env python3
"""
BlindStar 实时视频分析统一启动器
整合了原有的三个启动脚本功能：
- realtime_analysis.py: 直接启动分析器
- start_realtime_analysis.py: 通过测试脚本启动
- realtime_launcher.py: 统一启动器功能

支持多种启动模式和完整的参数配置
"""

import sys
import os
import logging
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger("realtime_launcher")


def setup_logging(source_path: str) -> str:
    """设置日志配置"""
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = Path(source_path).stem if not source_path.isdigit() else f"摄像头{source_path}"
    log_filename = f"实时视频分析_{file_name}_{timestamp}.log"
    log_path = logs_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding='utf-8')
        ],
        force=True
    )
    
    return str(log_path)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realtime_launcher.py",
        description="BlindStar 实时视频决策分析系统统一启动器"
    )

    # 基础参数
    parser.add_argument('--source', '-s', type=str, default='0',
                       help='视频源：摄像头ID(如0,1)或视频文件路径')
    parser.add_argument('--model', '-m', type=str, default='small',
                       choices=['nano', 'small', 'medium', 'large', 'n', 's', 'm', 'l', 'x', 'tiny'],
                       help='YOLO模型大小')
    parser.add_argument('--fps', '-f', type=float, default=15.0,
                       help='最大处理帧率')
    parser.add_argument('--confidence', '-c', type=float, default=0.6,
                       help='检测置信度阈值')
    
    # 功能开关
    parser.add_argument('--no-depth', action='store_true',
                       help='禁用深度估计（提高性能）')
    parser.add_argument('--no-tts', action='store_true',
                       help='禁用语音播报')
    
    # 模块化启动参数（兼容start_realtime_analysis.py）
    parser.add_argument('--module', type=str, 
                       choices=['detection', 'depth', 'decision', 'tts', 'navigation', 'all'],
                       help='启动指定模块（兼容模式，使用测试脚本）')
    
    # 启动模式选择
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'direct', 'test'],
                       help='启动模式：auto(自动选择), direct(直接启动), test(测试脚本)')
    
    # 导航集成参数
    parser.add_argument('--enable-navigation', action='store_true',
                       help='启用导航功能集成')
    parser.add_argument('--nav-mode', type=str, default='assist',
                       choices=['assist', 'guide', 'full'],
                       help='导航模式：assist(辅助), guide(引导), full(完全导航)')

    return parser


def validate_args(args: argparse.Namespace) -> bool:
    # model 校验
    valid_models = {"n", "s", "m", "l", "x", "nano", "tiny", "small", "medium", "large"}
    if args.model.lower() not in valid_models:
        logger.error(f"无效的模型: {args.model}")
        return False
    
    # test 模式下 module 可选校验
    if args.mode == "test" and args.module is not None:
        valid_modules = {"detection", "depth", "tts", "navigation", "decision", "all"}
        if args.module.lower() not in valid_modules:
            logger.error(f"无效的测试模块: {args.module}")
            return False
    return True

def launch_with_test_script(args):
    """使用测试脚本启动（兼容旧版本start_realtime_analysis.py功能）"""
    test_script = project_root / "tests" / "test_realtime_analysis.py"
    
    if not test_script.exists():
        print("❌ 错误：找不到测试脚本")
        print(f"预期路径: {test_script}")
        return False
    
    # 构建命令行参数
    cmd = [sys.executable, str(test_script)]
    
    # 转换参数格式
    if hasattr(args, 'source') and args.source:
        cmd.extend(['--source', args.source])
    if hasattr(args, 'module') and args.module:
        cmd.extend(['--module', args.module])
    if hasattr(args, 'model') and args.model:
        cmd.extend(['--model', args.model])
    if hasattr(args, 'fps') and args.fps:
        cmd.extend(['--fps', str(args.fps)])
    if hasattr(args, 'confidence') and args.confidence:
        cmd.extend(['--confidence', str(args.confidence)])
    if hasattr(args, 'no_depth') and args.no_depth:
        cmd.append('--no-depth')
    if hasattr(args, 'no_tts') and args.no_tts:
        cmd.append('--no-tts')
    
    print("🚀 启动BlindStar实时视频分析系统...")
    print(f"执行命令: {' '.join(cmd)}")
    print("="*60)
    
    try:
        result = subprocess.run(cmd, cwd=str(project_root))
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n用户中断分析")
        return True
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False


def launch_direct_analysis(args):
    """直接启动分析器（原realtime_analysis.py功能）"""
    from core.realtime_video_analyzer import RealtimeVideoAnalyzer
    
    # 设置日志
    log_path = setup_logging(args.source)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 启动BlindStar实时视频分析系统")
    logger.info(f"日志文件: {log_path}")
    logger.info(f"视频源: {args.source}")
    logger.info(f"YOLO模型: {args.model}")
    logger.info(f"最大FPS: {args.fps}")
    logger.info(f"深度估计: {'禁用' if args.no_depth else '启用'}")
    logger.info(f"语音播报: {'禁用' if args.no_tts else '启用'}")
    
    try:
        # 创建分析器
        analyzer = RealtimeVideoAnalyzer(
            yolo_model=args.model,
            confidence_threshold=args.confidence,
            enable_depth=not args.no_depth,
            enable_tts=not args.no_tts,
            max_fps=args.fps,
            enable_navigation=args.enable_navigation,
            nav_mode=args.nav_mode
        )
        
        print("\n" + "="*60)
        print("🎯 BlindStar 实时视频决策分析系统")
        print("="*60)
        print("📹 视频源:", args.source)
        print("🤖 YOLO模型:", args.model)
        print("⚡ 最大FPS:", args.fps)
        print("🔍 深度估计:", "✅启用" if not args.no_depth else "❌禁用")
        print("🔊 语音播报:", "✅启用" if not args.no_tts else "❌禁用")
        if args.enable_navigation:
            print("🧭 导航功能:", f"✅启用 ({args.nav_mode})")
        print("="*60)
        print("💡 按 'q' 键退出分析")
        print("="*60)
        
        # 启动分析
        success = analyzer.start_analysis(args.source, display_results=True)
        
        # 显示统计信息
        stats = analyzer.get_statistics()
        print("\n" + "="*60)
        print("📊 分析完成统计")
        print("="*60)
        print(f"总帧数: {stats['total_frames']}")
        print(f"处理帧数: {stats['processed_frames']}")
        print(f"丢帧数: {stats['dropped_frames']}")
        print(f"丢帧率: {stats['drop_rate']:.2%}")
        print(f"平均FPS: {stats['average_fps']:.1f}")
        print(f"平均处理时间: {stats['average_processing_time']*1000:.1f}ms")
        print(f"检测总数: {stats['detection_count']}")
        print(f"危险总数: {stats['hazard_count']}")
        print(f"决策总数: {stats['decision_count']}")
        print("="*60)
        
        if success:
            logger.info("✅ 实时分析成功完成")
        else:
            logger.error("❌ 实时分析出现错误")
            
        return success
        
    except KeyboardInterrupt:
        logger.info("用户中断分析")
        return True
    except Exception as e:
        logger.error(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_direct(args: argparse.Namespace) -> int:
    """直接启动实时分析（简化版，用于测试）"""
    try:
        success = launch_direct_analysis(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("收到中断信号，退出")
        return 0
    except Exception as e:
        logger.error(f"启动失败: {e}")
        time.sleep(1.0)
        return 1


def run_test(args: argparse.Namespace) -> int:
    """测试模式：不依赖真实视频源，快速自检占位"""
    logger.info(f"测试模式启动，模块: {args.module or 'all'}")
    # 简单保持进程运行，等待外部终止
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def show_usage_examples():
    """显示使用示例"""
    print("💡 使用示例:")
    print()
    print("📹 基础使用:")
    print("  python realtime_launcher.py --source data/1.mp4")
    print("  python realtime_launcher.py --source 0")
    print()
    print("⚡ 性能优化:")
    print("  python realtime_launcher.py --source 0 --no-depth --fps 30")
    print("  python realtime_launcher.py --source 0 --model nano --fps 25")
    print()
    print("🔇 静默模式:")
    print("  python realtime_launcher.py --source 0 --no-tts")
    print()
    print("🧩 模块化启动（兼容模式）:")
    print("  python realtime_launcher.py --source 0 --module detection")
    print("  python realtime_launcher.py --source 0 --module depth")
    print("  python realtime_launcher.py --source 0 --module decision")
    print("  python realtime_launcher.py --source 0 --module tts")
    print("  python realtime_launcher.py --source 0 --module all")
    print()
    print("🎯 启动模式:")
    print("  python realtime_launcher.py --source 0 --mode direct")
    print("  python realtime_launcher.py --source 0 --mode test --module all")
    print()
    print("🧭 导航模式:")
    print("  python realtime_launcher.py --source 0 --enable-navigation")
    print("  python realtime_launcher.py --source 0 --enable-navigation --nav-mode guide")
    print()
    print("🔧 模块说明:")
    print("  detection: 仅YOLO检测，最高性能")
    print("  depth:     检测 + 深度估计")
    print("  decision:  检测 + 深度 + 智能决策")
    print("  tts:       完整系统 + 语音播报")
    print("  navigation: 导航功能测试")
    print("  all:       完整系统（默认）")

def main(argv=None) -> int:
    """主函数"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not validate_args(args):
        return 2
    
    # 显示启动信息
    print("BlindStar 实时视频分析系统统一启动器 v3.0")
    print("="*60)
    print("📋 启动参数:")
    print(f"  视频源: {args.source}")
    print(f"  YOLO模型: {args.model}")
    print(f"  最大FPS: {args.fps}")
    print(f"  置信度: {args.confidence}")
    print(f"  深度估计: {'❌禁用' if args.no_depth else '✅启用'}")
    print(f"  语音播报: {'❌禁用' if args.no_tts else '✅启用'}")
    print(f"  启动模式: {args.mode}")
    if args.module:
        print(f"  模块选择: {args.module}")
    if args.enable_navigation:
        print(f"  导航功能: ✅启用 ({args.nav_mode})")
    print("="*60)
    
    # 根据模式选择启动方式
    if args.mode == 'test' or (args.mode == 'auto' and args.module):
        # 使用测试脚本启动（兼容start_realtime_analysis.py）
        print("🔄 使用测试脚本模式启动（兼容模式）")
        success = launch_with_test_script(args)
        return 0 if success else 1
    elif args.mode == 'direct' or args.mode == 'auto':
        # 直接启动分析器（realtime_analysis.py功能）
        print("🚀 使用直接模式启动")
        return run_direct(args)
    else:
        print("❌ 未知的启动模式")
        return 2


if __name__ == "__main__":
    print("🎯 BlindStar 实时视频分析系统统一启动器")
    print("整合了原有三个启动脚本的所有功能")
    print("="*60)
    
    # 如果没有参数，显示使用示例
    if len(sys.argv) == 1:
        show_usage_examples()
        print("="*60)
        print("💡 添加 --help 查看完整参数说明")
        print("💡 添加任意参数开始使用，例如: --source 0")
        sys.exit(0)
    
    success = main()
    sys.exit(success)
