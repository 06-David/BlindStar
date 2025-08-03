#!/usr/bin/env python3
"""
批量处理模块
提供视频和图像的批量处理功能
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import time

from .advanced_frame_analyzer import RobustVideoAnalyzer, VideoAnalysisStats

# 导入配置
try:
    from ..config import VideoConfig, ImageConfig
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import VideoConfig, ImageConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """批量处理配置"""
    yolo_model: str = "small"
    midas_model: str = "MiDaS_small"
    device: str = "cuda"
    confidence_threshold: float = 0.6
    max_frames_per_video: Optional[int] = None
    enable_distance: bool = True
    output_base_dir: str = "logs/batch_processing"
    parallel_processing: bool = False
    max_workers: int = 1
    # 视频输出配置
    generate_output_video: bool = True
    output_video_quality: str = "high"  # low, medium, high


@dataclass
class BatchProcessingResult:
    """批量处理结果"""
    total_files: int
    successful_files: int
    failed_files: int
    total_processing_time: float
    results: List[Dict[str, Any]]
    error_files: List[Dict[str, str]]
    summary_stats: Dict[str, Any]


class BatchVideoProcessor:
    """批量视频处理器"""
    
    def __init__(self, config: BatchProcessingConfig = None):
        """
        初始化批量视频处理器
        
        Args:
            config: 批量处理配置
        """
        self.config = config or BatchProcessingConfig()
        self.supported_formats = VideoConfig.SUPPORTED_FORMATS
        
        logger.info(f"批量视频处理器初始化完成")
        logger.info(f"  配置: {self.config}")
    
    def find_video_files(self, 
                        directory: Union[str, Path], 
                        recursive: bool = True) -> List[Path]:
        """
        查找目录下的视频文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归搜索子目录
            
        Returns:
            视频文件路径列表
        """
        video_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"目录不存在: {directory}")
            return video_files
        
        logger.info(f"搜索视频文件: {directory} (递归: {recursive})")
        
        # 选择搜索方法
        search_pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(search_pattern):
            if file_path.is_file():
                extension = file_path.suffix.lower().lstrip('.')
                if extension in self.supported_formats:
                    video_files.append(file_path)
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        for video_file in video_files:
            logger.info(f"  - {video_file.name}")
        
        return sorted(video_files)
    
    def process_single_video(self, 
                           video_path: Path,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            progress_callback: 进度回调函数
            
        Returns:
            处理结果字典
        """
        logger.info(f"开始处理视频: {video_path.name}")
        
        try:
            # 创建视频分析器
            analyzer = RobustVideoAnalyzer(
                yolo_model=self.config.yolo_model,
                midas_model=self.config.midas_model,
                device=self.config.device,
                confidence=self.config.confidence_threshold
            )
            
            # 分析视频
            stats = analyzer.analyze_video(
                video_path=str(video_path),
                max_frames=self.config.max_frames_per_video,
                output_video=self.config.generate_output_video
            )
            
            # 构建结果
            result = {
                "status": "success",
                "video_path": str(video_path),
                "video_name": video_path.name,
                "stats": stats,
                "processing_time": stats.processing_time_seconds,
                "total_frames": stats.total_frames,
                "processed_frames": stats.processed_frames,
                "total_detections": stats.total_detections,
                "unique_classes": stats.unique_classes,
                "output_directory": stats.output_directory
            }
            
            logger.info(f"✅ {video_path.name} 处理完成")
            logger.info(f"   帧数: {stats.processed_frames}/{stats.total_frames}")
            logger.info(f"   检测: {stats.total_detections} 个物体")
            logger.info(f"   类别: {len(stats.unique_classes)} 种")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ {video_path.name} 处理失败: {e}")
            
            return {
                "status": "failed",
                "video_path": str(video_path),
                "video_name": video_path.name,
                "error": str(e),
                "processing_time": 0
            }
    
    def process_directory(self, 
                         directory: Union[str, Path],
                         recursive: bool = True,
                         progress_callback: Optional[Callable] = None) -> BatchProcessingResult:
        """
        批量处理目录下的所有视频
        
        Args:
            directory: 目录路径
            recursive: 是否递归搜索
            progress_callback: 进度回调函数
            
        Returns:
            批量处理结果
        """
        start_time = time.time()
        
        logger.info(f"开始批量视频处理: {directory}")
        
        # 查找视频文件
        video_files = self.find_video_files(directory, recursive)
        
        if not video_files:
            logger.warning(f"未找到任何视频文件")
            return BatchProcessingResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                total_processing_time=0,
                results=[],
                error_files=[],
                summary_stats={}
            )
        
        # 处理每个视频
        results = []
        error_files = []
        successful_count = 0
        
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"处理视频 {i}/{len(video_files)}: {video_file.name}")
            logger.info(f"{'='*80}")
            
            # 进度回调
            if progress_callback:
                progress_callback(i, len(video_files), video_file.name)
            
            # 处理视频
            result = self.process_single_video(video_file)
            results.append(result)
            
            if result["status"] == "success":
                successful_count += 1
            else:
                error_files.append({
                    "file": video_file.name,
                    "error": result.get("error", "未知错误")
                })
        
        # 计算总体统计
        total_time = time.time() - start_time
        summary_stats = self._calculate_summary_stats(results)
        
        # 创建批量处理结果
        batch_result = BatchProcessingResult(
            total_files=len(video_files),
            successful_files=successful_count,
            failed_files=len(video_files) - successful_count,
            total_processing_time=total_time,
            results=results,
            error_files=error_files,
            summary_stats=summary_stats
        )
        
        # 输出总结
        self._log_batch_summary(batch_result)
        
        return batch_result
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算总体统计信息"""
        successful_results = [r for r in results if r["status"] == "success"]
        
        if not successful_results:
            return {}
        
        total_frames = sum(r.get("total_frames", 0) for r in successful_results)
        processed_frames = sum(r.get("processed_frames", 0) for r in successful_results)
        total_detections = sum(r.get("total_detections", 0) for r in successful_results)
        
        # 收集所有检测类别
        all_classes = set()
        for result in successful_results:
            if "unique_classes" in result:
                all_classes.update(result["unique_classes"])
        
        # 计算平均处理时间
        processing_times = [r.get("processing_time", 0) for r in successful_results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "total_detections": total_detections,
            "unique_classes": sorted(list(all_classes)),
            "unique_class_count": len(all_classes),
            "average_processing_time": avg_processing_time,
            "completion_rate": (processed_frames / total_frames * 100) if total_frames > 0 else 0
        }
    
    def _log_batch_summary(self, result: BatchProcessingResult):
        """输出批量处理总结"""
        logger.info(f"\n{'='*80}")
        logger.info("批量处理完成总结")
        logger.info(f"{'='*80}")
        logger.info(f"总视频数: {result.total_files}")
        logger.info(f"成功处理: {result.successful_files}")
        logger.info(f"失败数量: {result.failed_files}")
        logger.info(f"成功率: {(result.successful_files/result.total_files)*100:.1f}%")
        logger.info(f"总处理时间: {result.total_processing_time:.2f}秒")
        
        if result.summary_stats:
            stats = result.summary_stats
            logger.info(f"总处理帧数: {stats.get('processed_frames', 0)}")
            logger.info(f"总检测数: {stats.get('total_detections', 0)}")
            logger.info(f"检测类别数: {stats.get('unique_class_count', 0)}")
            logger.info(f"完成率: {stats.get('completion_rate', 0):.1f}%")
            
            if stats.get('unique_classes'):
                logger.info(f"检测类别: {', '.join(stats['unique_classes'][:10])}")
                if len(stats['unique_classes']) > 10:
                    logger.info(f"  ... 还有 {len(stats['unique_classes']) - 10} 个类别")
        
        if result.error_files:
            logger.info(f"\n失败文件:")
            for error_file in result.error_files:
                logger.info(f"  - {error_file['file']}: {error_file['error']}")


class BatchImageProcessor:
    """批量图像处理器"""
    
    def __init__(self, config: BatchProcessingConfig = None):
        """
        初始化批量图像处理器
        
        Args:
            config: 批量处理配置
        """
        self.config = config or BatchProcessingConfig()
        self.supported_formats = ImageConfig.SUPPORTED_FORMATS
        
        logger.info(f"批量图像处理器初始化完成")
    
    def find_image_files(self, 
                        directory: Union[str, Path], 
                        recursive: bool = True) -> List[Path]:
        """
        查找目录下的图像文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归搜索子目录
            
        Returns:
            图像文件路径列表
        """
        image_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"目录不存在: {directory}")
            return image_files
        
        logger.info(f"搜索图像文件: {directory} (递归: {recursive})")
        
        # 选择搜索方法
        search_pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(search_pattern):
            if file_path.is_file():
                extension = file_path.suffix.lower().lstrip('.')
                if extension in self.supported_formats:
                    image_files.append(file_path)
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        return sorted(image_files)
    
    def process_directory(self, 
                         directory: Union[str, Path],
                         recursive: bool = True,
                         save_results: bool = True,
                         progress_callback: Optional[Callable] = None) -> BatchProcessingResult:
        """
        批量处理目录下的所有图像
        
        Args:
            directory: 目录路径
            recursive: 是否递归搜索
            save_results: 是否保存标注结果
            progress_callback: 进度回调函数
            
        Returns:
            批量处理结果
        """
        start_time = time.time()
        
        logger.info(f"开始批量图像处理: {directory}")
        
        # 查找图像文件
        image_files = self.find_image_files(directory, recursive)
        
        if not image_files:
            logger.warning(f"未找到任何图像文件")
            return BatchProcessingResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                total_processing_time=0,
                results=[],
                error_files=[],
                summary_stats={}
            )
        
        # 初始化BlindStar
        from . import BlindStar
        
        blindstar = BlindStar(
            yolo_model=self.config.yolo_model,
            midas_model=self.config.midas_model,
            confidence_threshold=self.config.confidence_threshold,
            enable_distance=self.config.enable_distance,
            device=self.config.device
        )
        
        if not blindstar.initialize():
            raise RuntimeError("BlindStar初始化失败")
        
        try:
            # 处理每个图像
            results = []
            error_files = []
            successful_count = 0
            
            for i, image_file in enumerate(image_files, 1):
                logger.info(f"处理图像 {i}/{len(image_files)}: {image_file.name}")
                
                # 进度回调
                if progress_callback:
                    progress_callback(i, len(image_files), image_file.name)
                
                try:
                    # 处理图像
                    output_path = None
                    if save_results:
                        output_dir = Path(self.config.output_base_dir) / "images"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = str(output_dir / f"detected_{image_file.stem}.jpg")
                    
                    result = blindstar.detect_image(
                        image_path=str(image_file),
                        save_result=save_results,
                        output_path=output_path
                    )
                    
                    # 构建结果
                    image_result = {
                        "status": "success",
                        "image_path": str(image_file),
                        "image_name": image_file.name,
                        "total_detections": result["total_objects"],
                        "processing_time": result["processing_time"],
                        "detections": result["detections"],
                        "output_path": output_path
                    }
                    
                    results.append(image_result)
                    successful_count += 1
                    
                    logger.info(f"✅ {image_file.name}: {result['total_objects']} 个物体")
                
                except Exception as e:
                    logger.error(f"❌ {image_file.name} 处理失败: {e}")
                    
                    error_result = {
                        "status": "failed",
                        "image_path": str(image_file),
                        "image_name": image_file.name,
                        "error": str(e),
                        "processing_time": 0
                    }
                    
                    results.append(error_result)
                    error_files.append({
                        "file": image_file.name,
                        "error": str(e)
                    })
            
            # 计算总体统计
            total_time = time.time() - start_time
            summary_stats = self._calculate_image_summary_stats(results)
            
            # 创建批量处理结果
            batch_result = BatchProcessingResult(
                total_files=len(image_files),
                successful_files=successful_count,
                failed_files=len(image_files) - successful_count,
                total_processing_time=total_time,
                results=results,
                error_files=error_files,
                summary_stats=summary_stats
            )
            
            # 输出总结
            self._log_image_batch_summary(batch_result)
            
            return batch_result
        
        finally:
            blindstar.cleanup()
    
    def _calculate_image_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算图像批量处理统计信息"""
        successful_results = [r for r in results if r["status"] == "success"]
        
        if not successful_results:
            return {}
        
        total_detections = sum(r.get("total_detections", 0) for r in successful_results)
        
        # 收集所有检测类别
        all_classes = set()
        for result in successful_results:
            if "detections" in result:
                for detection in result["detections"]:
                    all_classes.add(detection.get("class_name", "unknown"))
        
        # 计算平均处理时间
        processing_times = [r.get("processing_time", 0) for r in successful_results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_detections": total_detections,
            "unique_classes": sorted(list(all_classes)),
            "unique_class_count": len(all_classes),
            "average_processing_time": avg_processing_time,
            "average_detections_per_image": total_detections / len(successful_results) if successful_results else 0
        }
    
    def _log_image_batch_summary(self, result: BatchProcessingResult):
        """输出图像批量处理总结"""
        logger.info(f"\n{'='*80}")
        logger.info("批量图像处理完成总结")
        logger.info(f"{'='*80}")
        logger.info(f"总图像数: {result.total_files}")
        logger.info(f"成功处理: {result.successful_files}")
        logger.info(f"失败数量: {result.failed_files}")
        logger.info(f"成功率: {(result.successful_files/result.total_files)*100:.1f}%")
        logger.info(f"总处理时间: {result.total_processing_time:.2f}秒")
        
        if result.summary_stats:
            stats = result.summary_stats
            logger.info(f"总检测数: {stats.get('total_detections', 0)}")
            logger.info(f"平均每图检测: {stats.get('average_detections_per_image', 0):.1f} 个物体")
            logger.info(f"检测类别数: {stats.get('unique_class_count', 0)}")
            logger.info(f"平均处理时间: {stats.get('average_processing_time', 0):.3f}秒/图")
            
            if stats.get('unique_classes'):
                logger.info(f"检测类别: {', '.join(stats['unique_classes'])}")
        
        if result.error_files:
            logger.info(f"\n失败文件:")
            for error_file in result.error_files:
                logger.info(f"  - {error_file['file']}: {error_file['error']}")
