#!/usr/bin/env python3
"""
决策引擎模块 - BlindStar决策系统主控制器
整合风险评估和动作规划，为视障辅助系统提供统一的决策接口
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
import numpy as np

from .risk_assessor import RiskAssessor, ObjectRisk, RiskLevel
from .action_planner import ActionPlanner, ActionPlan, ActionType

logger = logging.getLogger(__name__)


class DecisionContext:
    """决策上下文 - 标准化的感知输入数据"""
    def __init__(self, frame_id: int, timestamp: float, objects: List[Dict[str, Any]],
                 camera_pose: Optional[List[float]] = None,
                 frame_info: Optional[Dict[str, Any]] = None):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.objects = objects  # 检测物体列表
        self.camera_pose = camera_pose or [0, 0, 0]
        self.frame_info = frame_info or {}


class DecisionOutput:
    """决策输出 - 标准化的决策结果"""
    def __init__(self, frame_id: int, action_plan: ActionPlan, 
                 risks: List[ObjectRisk], processing_time: float):
        self.frame_id = frame_id
        self.action = action_plan.action.value
        self.priority = action_plan.priority
        self.speech = action_plan.speech_text
        self.vibration_pattern = action_plan.vibration_pattern
        self.target_bearing = action_plan.target_bearing
        self.confidence = action_plan.confidence
        self.risks = risks
        self.processing_time = processing_time
        
        # 统计信息
        self.total_objects = len(risks)
        self.high_risk_objects = len([r for r in risks if r.risk_level.value >= 2])
        self.max_risk_level = max([r.risk_level.value for r in risks]) if risks else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "frame_id": self.frame_id,
            "action": self.action,
            "priority": self.priority,
            "speech": self.speech,
            "vibration_pattern": self.vibration_pattern,
            "target_bearing": self.target_bearing,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "statistics": {
                "total_objects": self.total_objects,
                "high_risk_objects": self.high_risk_objects,
                "max_risk_level": self.max_risk_level
            },
            "risks": [
                {
                    "class_id": risk.class_id,
                    "class_name": risk.class_name,
                    "risk_level": risk.risk_level.value,
                    "risk_score": risk.risk_score,
                    "distance_risk": risk.distance_risk,
                    "velocity_risk": risk.velocity_risk,
                    "time_to_collide": risk.time_to_collide
                }
                for risk in self.risks
            ]
        }


class DecisionEngine:
    """决策引擎 - BlindStar决策系统的核心控制器"""
    
    def __init__(self, enable_logging: bool = True):
        """
        初始化决策引擎
        
        Args:
            enable_logging: 是否启用详细日志记录
        """
        self.enable_logging = enable_logging
        
        # 初始化子模块
        self.risk_assessor = RiskAssessor()
        self.action_planner = ActionPlanner()
        
        # 决策统计
        self.decision_count = 0
        self.total_processing_time = 0.0
        
        # 状态记忆
        self.last_decision_output = None
        self.emergency_state = False  # 紧急状态标志
        self.emergency_start_time = None
        
        logger.info("✅ 决策引擎初始化完成")
    
    def _validate_input_data(self, context: DecisionContext) -> bool:
        """验证输入数据的完整性"""
        if not isinstance(context.objects, list):
            logger.error("输入数据格式错误：objects必须是列表")
            return False
        
        # 验证每个检测物体的必要字段
        required_fields = ['class_id', 'class_name', 'confidence', 'bbox']
        for i, obj in enumerate(context.objects):
            for field in required_fields:
                if field not in obj:
                    logger.warning(f"物体{i}缺少必要字段: {field}")
                    return False
        
        return True
    
    def _enrich_detection_data(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """丰富检测数据，添加缺失字段的默认值"""
        enriched_objects = []
        
        for obj in objects:
            enriched_obj = obj.copy()
            
            # 添加缺失字段的默认值
            if 'distance_m' not in enriched_obj:
                enriched_obj['distance_m'] = None
            if 'velocity_mps' not in enriched_obj:
                enriched_obj['velocity_mps'] = None
            if 'height_m' not in enriched_obj:
                enriched_obj['height_m'] = None
            if 'time_to_collide' not in enriched_obj:
                enriched_obj['time_to_collide'] = None
            if 'is_static' not in enriched_obj:
                enriched_obj['is_static'] = True
            
            enriched_objects.append(enriched_obj)
        
        return enriched_objects
    
    def _check_emergency_conditions(self, risks: List[ObjectRisk]) -> bool:
        """检查是否存在紧急情况"""
        for risk in risks:
            # L3级别风险
            if risk.risk_level == RiskLevel.L3:
                return True
            
            # 即将碰撞的高风险物体
            if (risk.risk_level == RiskLevel.L2 and 
                risk.time_to_collide is not None and 
                risk.time_to_collide < 2.0):
                return True
            
            # 极近距离的危险物体
            if (risk.risk_level.value >= 1 and 
                risk.distance_risk > 0.9):
                return True
        
        return False
    
    def _update_emergency_state(self, is_emergency: bool):
        """更新紧急状态"""
        if is_emergency and not self.emergency_state:
            self.emergency_state = True
            self.emergency_start_time = time.time()
            logger.warning("⚠️ 进入紧急状态")
        elif not is_emergency and self.emergency_state:
            emergency_duration = time.time() - (self.emergency_start_time or 0)
            self.emergency_state = False
            self.emergency_start_time = None
            logger.info(f"✅ 退出紧急状态，持续时间: {emergency_duration:.2f}秒")
    
    def make_decision(self, context: DecisionContext) -> DecisionOutput:
        """
        进行决策分析
        
        Args:
            context: 决策上下文，包含感知数据
            
        Returns:
            DecisionOutput: 决策结果
        """
        start_time = time.time()
        
        # 验证输入数据
        if not self._validate_input_data(context):
            # 返回默认安全决策
            safe_plan = ActionPlan(
                action=ActionType.STOP,
                priority=5,
                speech_text="数据异常，请停止前进",
                confidence=0.5
            )
            return DecisionOutput(
                frame_id=context.frame_id,
                action_plan=safe_plan,
                risks=[],
                processing_time=time.time() - start_time
            )
        
        # 丰富检测数据
        enriched_objects = self._enrich_detection_data(context.objects)
        
        # 获取深度图数据（如果有的话）
        depth_map = context.frame_info.get('depth_map') if context.frame_info else None
        
        # 风险评估（包括深度危险检测）
        risks = self.risk_assessor.assess_frame_risks(
            enriched_objects, depth_map, context.frame_info
        )
        
        # 检查紧急状态
        is_emergency = self._check_emergency_conditions(risks)
        self._update_emergency_state(is_emergency)
        
        # 动作规划
        action_plan = self.action_planner.plan_action(risks, context.frame_info)
        
        # 紧急状态下强制停止
        if self.emergency_state and action_plan.action != ActionType.STOP:
            action_plan = ActionPlan(
                action=ActionType.STOP,
                priority=5,
                speech_text="紧急情况，请立即停止",
                vibration_pattern="emergency",
                confidence=1.0
            )
        
        # 创建决策输出
        processing_time = time.time() - start_time
        decision_output = DecisionOutput(
            frame_id=context.frame_id,
            action_plan=action_plan,
            risks=risks,
            processing_time=processing_time
        )
        
        # 更新统计信息
        self.decision_count += 1
        self.total_processing_time += processing_time
        self.last_decision_output = decision_output
        
        # 记录决策日志
        if self.enable_logging:
            self._log_decision(decision_output)
        
        return decision_output
    
    def _log_decision(self, output: DecisionOutput):
        """记录决策日志"""
        risk_summary = f"风险物体: {output.total_objects}, 高危: {output.high_risk_objects}"
        logger.info(f"决策#{output.frame_id}: {output.action} | {risk_summary} | "
                   f"处理时间: {output.processing_time*1000:.1f}ms")
        
        if output.high_risk_objects > 0:
            logger.warning(f"语音播报: {output.speech}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取决策引擎统计信息"""
        avg_processing_time = (self.total_processing_time / self.decision_count 
                              if self.decision_count > 0 else 0)
        
        return {
            "total_decisions": self.decision_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "emergency_state": self.emergency_state,
            "last_decision": self.last_decision_output.to_dict() if self.last_decision_output else None
        }
    
    def reset_state(self):
        """重置决策引擎状态"""
        self.action_planner.reset_history()
        self.emergency_state = False
        self.emergency_start_time = None
        self.last_decision_output = None
        logger.info("决策引擎状态已重置")
    
    def update_risk_parameters(self, **kwargs):
        """更新风险评估参数"""
        if hasattr(self.risk_assessor, 'safe_distance_threshold'):
            for key, value in kwargs.items():
                if hasattr(self.risk_assessor, key):
                    setattr(self.risk_assessor, key, value)
                    logger.info(f"风险参数已更新: {key} = {value}")
    
    def update_action_parameters(self, **kwargs):
        """更新动作规划参数"""
        if hasattr(self.action_planner, 'safe_side_distance'):
            for key, value in kwargs.items():
                if hasattr(self.action_planner, key):
                    setattr(self.action_planner, key, value)
                    logger.info(f"动作参数已更新: {key} = {value}")


# 便捷函数
def create_decision_context(frame_id: int, detections: List[Dict[str, Any]], 
                          frame_info: Optional[Dict[str, Any]] = None) -> DecisionContext:
    """创建决策上下文的便捷函数"""
    return DecisionContext(
        frame_id=frame_id,
        timestamp=time.time(),
        objects=detections,
        frame_info=frame_info
    )


def make_quick_decision(detections: List[Dict[str, Any]], 
                       frame_id: int = 0) -> DecisionOutput:
    """快速决策的便捷函数"""
    engine = DecisionEngine(enable_logging=False)
    context = create_decision_context(frame_id, detections)
    return engine.make_decision(context)
