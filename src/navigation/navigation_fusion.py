#!/usr/bin/env python3
"""
BlindStar 导航决策融合器
融合视觉决策和导航决策，生成统一的行动指令
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

from .navigation_context import NavigationContext, NavigationMode, NavigationState
from .action_planner import ActionPlan, ActionType
from .risk_assessor import ObjectRisk, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class NavigationDecision:
    """导航决策"""
    # 全局导航指令
    global_instruction: Optional[str] = None  # "直行100米后右转"
    direction_angle: Optional[float] = None   # 目标方向角度
    distance_to_turn: Optional[float] = None  # 距离转弯点距离
    
    # 局部避障指令
    local_hazards: List[ObjectRisk] = None    # 当前检测到的风险
    avoidance_instruction: Optional[str] = None  # "左侧有障碍物，请右侧通行"
    
    # 融合决策
    priority_level: int = 1                   # 优先级：1-全局导航，2-局部避障，3-紧急避险
    final_instruction: str = ""               # 最终播报指令
    action_type: str = "navigate"             # 动作类型："navigate", "avoid", "emergency"
    confidence: float = 1.0                   # 决策置信度
    timestamp: float = 0.0                    # 决策时间戳
    
    def __post_init__(self):
        if self.local_hazards is None:
            self.local_hazards = []
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class DecisionPriority(Enum):
    """决策优先级"""
    NAVIGATION = 1      # 全局导航
    AVOIDANCE = 2       # 局部避障
    EMERGENCY = 3       # 紧急避险


class NavigationDecisionFusion:
    """导航决策融合器"""
    
    def __init__(self, vision_engine=None, navigation_context=None):
        """
        初始化导航决策融合器
        
        Args:
            vision_engine: 视觉决策引擎
            navigation_context: 导航上下文管理器
        """
        self.vision_engine = vision_engine
        self.navigation_context = navigation_context
        self.logger = logging.getLogger(__name__)
        
        # 融合配置
        self.config = {
            'vision_weight': 0.7,        # 视觉决策权重
            'navigation_weight': 0.3,    # 导航决策权重
            'emergency_override': True,  # 紧急情况覆盖
            'instruction_interval': 5.0, # 指令播报间隔(秒)
            'hazard_priority': True      # 危险提醒优先级
        }
        
        # 状态跟踪
        self.last_instruction_time = 0.0
        self.last_decision = None
        self.decision_history = []
        self.max_history_size = 50
        
        self.logger.info("[融合] 导航决策融合器初始化完成")
    
    def make_fused_decision(self, frame_data: Dict[str, Any]) -> NavigationDecision:
        """
        融合视觉决策和导航决策
        
        Args:
            frame_data: 当前帧数据，包含检测结果、深度信息等
            
        Returns:
            NavigationDecision: 融合后的导航决策
        """
        try:
            # 获取视觉决策
            vision_decision = None
            if self.vision_engine:
                vision_decision = self.vision_engine.make_decision(frame_data)
            
            # 获取导航上下文
            nav_context = None
            if self.navigation_context:
                nav_context = self.navigation_context.get_context()
            
            # 融合决策
            fused_decision = self.fuse_decisions(vision_decision, nav_context)
            
            # 更新历史记录
            self._update_decision_history(fused_decision)
            
            return fused_decision
            
        except Exception as e:
            self.logger.error(f"[融合] 决策融合失败: {e}")
            # 返回默认决策
            return NavigationDecision(
                final_instruction="系统暂时不可用",
                action_type="error",
                priority_level=3
            )
    
    def fuse_decisions(self, vision_decision: Optional[ActionPlan], 
                      nav_context: Optional[NavigationContext]) -> NavigationDecision:
        """
        融合视觉决策和导航决策
        
        Args:
            vision_decision: 视觉决策结果
            nav_context: 导航上下文
            
        Returns:
            NavigationDecision: 融合决策
        """
        decision = NavigationDecision()
        
        # 提取视觉决策信息
        if vision_decision:
            decision.local_hazards = getattr(vision_decision, 'risks', [])
            decision.avoidance_instruction = getattr(vision_decision, 'instruction', None)
        
        # 提取导航决策信息
        if nav_context and nav_context.navigation_state != NavigationState.IDLE:
            if self.navigation_context:
                decision.global_instruction = self.navigation_context.get_current_instruction()
        
        # 确定优先级和最终指令
        decision.priority_level, decision.final_instruction, decision.action_type = \
            self.prioritize_instructions(
                decision.global_instruction,
                decision.avoidance_instruction,
                decision.local_hazards
            )
        
        # 计算置信度
        decision.confidence = self._calculate_confidence(vision_decision, nav_context)
        
        return decision
    
    def prioritize_instructions(self, global_instruction: Optional[str],
                              local_instruction: Optional[str],
                              hazards: List[ObjectRisk]) -> tuple:
        """
        指令优先级处理
        
        Args:
            global_instruction: 全局导航指令
            local_instruction: 局部避障指令
            hazards: 检测到的风险列表
            
        Returns:
            tuple: (优先级, 最终指令, 动作类型)
        """
        # 检查紧急情况
        emergency_risks = [risk for risk in hazards if risk.risk_level == RiskLevel.L3]
        if emergency_risks and self.config['emergency_override']:
            emergency_instruction = self._generate_emergency_instruction(emergency_risks)
            return DecisionPriority.EMERGENCY.value, emergency_instruction, "emergency"
        
        # 检查高风险情况
        high_risks = [risk for risk in hazards if risk.risk_level in [RiskLevel.L2, RiskLevel.L3]]
        if high_risks and self.config['hazard_priority']:
            if local_instruction:
                return DecisionPriority.AVOIDANCE.value, local_instruction, "avoid"
            else:
                avoidance_instruction = self._generate_avoidance_instruction(high_risks)
                return DecisionPriority.AVOIDANCE.value, avoidance_instruction, "avoid"
        
        # 检查是否需要播报导航指令
        current_time = time.time()
        if (global_instruction and 
            current_time - self.last_instruction_time > self.config['instruction_interval']):
            self.last_instruction_time = current_time
            return DecisionPriority.NAVIGATION.value, global_instruction, "navigate"
        
        # 如果有局部避障指令但不是高风险
        if local_instruction:
            return DecisionPriority.AVOIDANCE.value, local_instruction, "avoid"
        
        # 默认情况
        return DecisionPriority.NAVIGATION.value, "继续前进", "navigate"
    
    def _generate_emergency_instruction(self, emergency_risks: List[ObjectRisk]) -> str:
        """生成紧急避险指令"""
        if not emergency_risks:
            return "立即停止"
        
        risk_descriptions = []
        for risk in emergency_risks[:2]:  # 最多报告2个最危险的情况
            if hasattr(risk, 'object_name') and hasattr(risk, 'distance'):
                risk_descriptions.append(f"{risk.object_name}距离{risk.distance:.1f}米")
        
        if risk_descriptions:
            return f"危险！前方有{', '.join(risk_descriptions)}，立即停止"
        else:
            return "危险！立即停止"
    
    def _generate_avoidance_instruction(self, high_risks: List[ObjectRisk]) -> str:
        """生成避障指令"""
        if not high_risks:
            return "注意避让"
        
        # 简化的避障指令生成
        risk = high_risks[0]  # 取第一个高风险对象
        if hasattr(risk, 'object_name'):
            return f"注意避让前方{risk.object_name}"
        else:
            return "注意前方障碍物"
    
    def _calculate_confidence(self, vision_decision: Optional[ActionPlan], 
                            nav_context: Optional[NavigationContext]) -> float:
        """计算决策置信度"""
        confidence = 0.5  # 基础置信度
        
        # 视觉决策置信度
        if vision_decision and hasattr(vision_decision, 'confidence'):
            confidence += vision_decision.confidence * self.config['vision_weight']
        
        # 导航决策置信度
        if nav_context and nav_context.current_location:
            # 如果有GPS位置信息，增加置信度
            if nav_context.current_location.accuracy and nav_context.current_location.accuracy < 10:
                confidence += 0.3 * self.config['navigation_weight']
            else:
                confidence += 0.1 * self.config['navigation_weight']
        
        return min(confidence, 1.0)
    
    def _update_decision_history(self, decision: NavigationDecision):
        """更新决策历史"""
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history_size:
            self.decision_history.pop(0)
        
        self.last_decision = decision
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        if not self.decision_history:
            return {}
        
        total_decisions = len(self.decision_history)
        emergency_count = sum(1 for d in self.decision_history if d.action_type == "emergency")
        avoidance_count = sum(1 for d in self.decision_history if d.action_type == "avoid")
        navigation_count = sum(1 for d in self.decision_history if d.action_type == "navigate")
        
        avg_confidence = sum(d.confidence for d in self.decision_history) / total_decisions
        
        return {
            'total_decisions': total_decisions,
            'emergency_decisions': emergency_count,
            'avoidance_decisions': avoidance_count,
            'navigation_decisions': navigation_count,
            'average_confidence': avg_confidence,
            'last_decision_time': self.last_decision.timestamp if self.last_decision else 0
        }
    
    def update_config(self, config_updates: Dict[str, Any]):
        """更新融合配置"""
        self.config.update(config_updates)
        self.logger.info(f"[融合] 配置已更新: {config_updates}")
