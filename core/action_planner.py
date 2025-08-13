#!/usr/bin/env python3
"""
动作规划模块 - BlindStar决策系统核心组件
根据风险评估结果生成具体的动作建议和语音指令
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from .risk_assessor import ObjectRisk, RiskLevel

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """动作类型枚举"""
    CONTINUE = "CONTINUE"           # 继续前进
    FOLLOW_GUIDE = "FOLLOW_GUIDE"   # 跟随指引
    SIDE_STEP = "SIDE_STEP"         # 侧向避让
    BRAKE_AND_WARN = "BRAKE_AND_WARN"  # 刹车预警
    STOP = "STOP"                   # 立即停止


class ActionPlan:
    """动作规划结果"""
    def __init__(self,
                 action: ActionType,
                 priority: int,
                 speech_text: str,
                 target_bearing: Optional[float] = None,
                 vibration_pattern: str = "none",
                 confidence: float = 1.0):
        self.action = action
        self.priority = priority  # 优先级 1-5，5最高
        self.speech_text = speech_text
        self.target_bearing = target_bearing  # 目标方位角度，左负右正
        self.vibration_pattern = vibration_pattern  # 振动模式
        self.confidence = confidence  # 决策置信度 0-1


class ActionPlanner:
    """动作规划器 - 根据风险评估生成动作建议"""
    
    def __init__(self):
        """初始化动作规划器"""
        # 动作规划参数
        self.safe_side_distance = 0.8  # 侧向安全距离(米)
        self.emergency_distance = 1.0   # 紧急停止距离(米)
        self.warning_distance = 2.0     # 预警距离(米)
        
        # 语音模板
        self.speech_templates = self._init_speech_templates()
        
        # 动作历史记录(用于平滑决策)
        self.action_history = []
        self.max_history_length = 5
        
        logger.info("✅ 动作规划器初始化完成")
    
    def _init_speech_templates(self) -> Dict[str, Dict[str, str]]:
        """初始化语音播报模板"""
        return {
            # L0级别 - 提示指引
            "L0": {
                "default": "前方有{object}，请注意",
                "guide": "前方{distance}米处有{object}，建议跟随",
                "multiple": "前方有多个指引标识"
            },
            
            # L1级别 - 避让绕行
            "L1": {
                "close": "注意，前方{distance}米处有{object}",
                "side": "前方{distance}米处有{object}，建议{direction}侧避让",
                "multiple": "前方有多个障碍物，请小心通行",
                "moving": "注意，{object}正在移动，距离{distance}米",
                "moderate_speed": "注意，{object}正在移动，距离{distance}米，请小心避让"
            },
            
            # L2级别 - 高危预警
            "L2": {
                "approaching": "警告！{object}正在靠近，距离{distance}米",
                "close": "危险！前方{distance}米处有{object}，请立即避让",
                "fast": "高速{object}接近，距离{distance}米，请注意安全",
                "very_fast": "极高速{object}！距离{distance}米，立即避让！",
                "collision": "碰撞预警！{object}将在{time}秒后到达",
                "speed_danger": "危险！高速移动的{object}，距离{distance}米"
            },
            
            # L3级别 - 强制停止
            "L3": {
                "stop": "停止！前方是{object}",
                "red_light": "红灯，请停止前进",
                "barrier": "前方有障碍，禁止通行",
                "danger": "危险区域，请立即停止"
            }
        }
    
    def _get_direction_text(self, bearing: float) -> str:
        """根据方位角获取方向文本"""
        if bearing < -45:
            return "左前方"
        elif bearing < -15:
            return "左侧"
        elif bearing > 45:
            return "右前方"
        elif bearing > 15:
            return "右侧"
        else:
            return "正前方"
    
    def _analyze_spatial_occupancy(self, target_bbox: List[float], 
                                 all_risks: List[ObjectRisk],
                                 frame_width: int, frame_height: int,
                                 detections: List[Dict[str, Any]] = None) -> Tuple[float, float]:
        """分析左右空间的占用情况，返回左侧和右侧的空间得分"""
        left_occupancy = 0.0
        right_occupancy = 0.0
        
        frame_center_x = frame_width / 2
        target_center_x = (target_bbox[0] + target_bbox[2]) / 2
        
        # 分析其他物体对左右空间的占用
        if detections:
            for detection in detections:
                detection_bbox = detection.get('bbox')
                if detection_bbox is None:
                    continue
                
                # 跳过目标物体本身
                detection_center_x = (detection_bbox[0] + detection_bbox[2]) / 2
                if abs(detection_center_x - target_center_x) < 20:  # 可能是同一个物体
                    continue
                
                # 计算物体对左右空间的影响
                object_width = detection_bbox[2] - detection_bbox[0]
                distance_m = detection.get('distance_m', 5.0)  # 默认5米
                
                # 距离越近影响越大
                distance_factor = max(0.1, min(1.0, 3.0 / max(0.5, distance_m)))
                
                # 物体越大影响越大
                size_factor = min(1.0, object_width / (frame_width * 0.3))
                
                # 风险等级影响
                risk_factor = 0.2
                for risk in all_risks:
                    if (risk.class_id == detection.get('class_id') and 
                        risk.class_name == detection.get('class_name')):
                        risk_factor = (risk.risk_level.value + 1) * 0.2
                        break
                
                occupancy_impact = distance_factor * size_factor * risk_factor
                
                # 根据物体位置分配到左右空间
                if detection_center_x < frame_center_x:
                    left_occupancy += occupancy_impact
                else:
                    right_occupancy += occupancy_impact
        
        # 考虑目标物体本身的位置
        if target_center_x < frame_center_x:
            # 目标在左侧，左侧空间受限
            left_occupancy += 0.4
        else:
            # 目标在右侧，右侧空间受限
            right_occupancy += 0.4
        
        # 计算空间得分（越高表示越适合避让到该侧）
        left_space_score = max(0.0, 1.0 - left_occupancy)
        right_space_score = max(0.0, 1.0 - right_occupancy)
        
        return left_space_score, right_space_score
    
    def _determine_optimal_avoidance(self, relative_x: float, relative_y: float,
                                   object_width: float, object_height: float,
                                   left_space_score: float, right_space_score: float,
                                   frame_width: int) -> Tuple[float, str]:
        """确定最优避让方向"""
        
        # 基础避让角度
        base_angle = 45.0
        
        # 根据物体位置调整避让策略
        if abs(relative_x) < 0.2:  # 物体在中央
            # 选择空间更大的一侧
            if left_space_score > right_space_score:
                angle = -base_angle
                direction = "左侧"
            else:
                angle = base_angle  
                direction = "右侧"
        elif relative_x < 0:  # 物体在左侧
            # 优先向右避让，但考虑右侧空间
            if right_space_score > 0.3:
                angle = base_angle
                direction = "右侧"
            else:
                # 右侧空间不足，尝试左后方
                angle = -base_angle * 1.5
                direction = "左后方"
        else:  # 物体在右侧
            # 优先向左避让，但考虑左侧空间
            if left_space_score > 0.3:
                angle = -base_angle
                direction = "左侧"
            else:
                # 左侧空间不足，尝试右后方
                angle = base_angle * 1.5
                direction = "右后方"
        
        # 根据物体大小调整避让角度
        size_factor = min(2.0, (object_width + object_height) / (frame_width * 0.5))
        angle *= size_factor
        
        # 限制角度范围
        angle = max(-90.0, min(90.0, angle))
        
        return angle, direction
    
    def _calculate_avoidance_bearing(self, target_risk: ObjectRisk, 
                                   all_risks: List[ObjectRisk],
                                   frame_info: Dict[str, Any]) -> Tuple[float, str]:
        """计算避让方向的方位角和详细说明"""
        frame_width = frame_info.get('frame_width', 640)
        frame_height = frame_info.get('frame_height', 480)
        
        # 从frame_info中获取检测数据
        detections = frame_info.get('detections', [])
        target_bbox = None
        
        # 查找目标物体的bbox
        for detection in detections:
            if (detection.get('class_id') == target_risk.class_id and 
                detection.get('class_name') == target_risk.class_name):
                target_bbox = detection.get('bbox')
                break
        
        # 如果没有找到bbox，尝试从风险对象中获取（如果有的话）
        if target_bbox is None and hasattr(target_risk, 'bbox'):
            target_bbox = target_risk.bbox
        
        # 如果仍然没有bbox，使用默认策略
        if target_bbox is None:
            logger.warning(f"无法获取{target_risk.class_name}的bbox信息，使用默认避让策略")
            return -45.0, "左侧"  # 默认向左避让
        
        # 分析空间占用情况
        left_space_score, right_space_score = self._analyze_spatial_occupancy(
            target_bbox, all_risks, frame_width, frame_height, detections
        )
        
        # 计算物体位置和大小
        center_x = (target_bbox[0] + target_bbox[2]) / 2
        center_y = (target_bbox[1] + target_bbox[3]) / 2
        object_width = target_bbox[2] - target_bbox[0]
        object_height = target_bbox[3] - target_bbox[1]
        
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        # 计算相对位置
        relative_x = (center_x - frame_center_x) / frame_center_x  # -1到1
        relative_y = (center_y - frame_center_y) / frame_center_y  # -1到1
        
        # 智能避让方向决策
        avoidance_angle, direction_text = self._determine_optimal_avoidance(
            relative_x, relative_y, object_width, object_height,
            left_space_score, right_space_score, frame_width
        )
        
        return avoidance_angle, direction_text
    
    def _generate_speech_text(self, highest_risk: ObjectRisk, 
                            all_risks: List[ObjectRisk],
                            target_bearing: Optional[float] = None,
                            avoidance_direction: Optional[str] = None) -> str:
        """生成语音播报文本"""
        risk_level = highest_risk.risk_level
        class_name = highest_risk.class_name
        
        # 获取距离信息
        distance_text = "未知"
        if hasattr(highest_risk, 'distance_m') and highest_risk.distance_m is not None:
            distance_text = f"{highest_risk.distance_m:.1f}"
        
        # 获取方向文本
        direction_text = ""
        if avoidance_direction is not None:
            direction_text = avoidance_direction
        elif target_bearing is not None:
            direction_text = self._get_direction_text(target_bearing)
        
        # 检查是否为深度危险
        if highest_risk.is_depth_hazard and highest_risk.depth_hazard:
            return highest_risk.depth_hazard.description
        
        # 根据风险等级选择模板
        templates = self.speech_templates.get(f"L{risk_level.value}", {})
        
        if risk_level == RiskLevel.L3:
            if "red" in class_name.lower() or "traffic" in class_name.lower():
                return templates.get("red_light", "前方红灯，请停止")
            elif "stop" in class_name.lower():
                template = templates.get("stop", "停止！前方是{object}")
                return template.format(object=class_name)
            else:
                return templates.get("danger", f"危险！前方有{class_name}，请立即停止")
        
        elif risk_level == RiskLevel.L2:
            if highest_risk.time_to_collide and highest_risk.time_to_collide < 3:
                template = templates.get("collision", "碰撞预警！{object}将在{time}秒后到达")
                return template.format(object=class_name, time=f"{highest_risk.time_to_collide:.1f}")
            elif highest_risk.velocity_risk >= 0.9:  # 极高速度
                template = templates.get("very_fast", "极高速{object}！距离{distance}米，立即避让！")
                return template.format(object=class_name, distance=distance_text)
            elif highest_risk.velocity_risk >= 0.7:  # 高速度
                template = templates.get("fast", "高速{object}接近，距离{distance}米，请注意安全")
                return template.format(object=class_name, distance=distance_text)
            elif highest_risk.velocity_risk >= 0.5:  # 中高速度
                template = templates.get("speed_danger", "危险！高速移动的{object}，距离{distance}米")
                return template.format(object=class_name, distance=distance_text)
            else:
                template = templates.get("close", "危险！前方{distance}米处有{object}，请立即避让")
                return template.format(object=class_name, distance=distance_text)
        
        elif risk_level == RiskLevel.L1:
            if len(all_risks) > 3:
                return templates.get("multiple", "前方有多个障碍物，请小心通行")
            elif highest_risk.velocity_risk >= 0.5:  # 中等速度
                template = templates.get("moderate_speed", "注意，{object}正在移动，距离{distance}米，请小心避让")
                return template.format(object=class_name, distance=distance_text)
            elif highest_risk.velocity_risk >= 0.3:  # 低速移动
                template = templates.get("moving", "注意，{object}正在移动，距离{distance}米")
                return template.format(object=class_name, distance=distance_text)
            elif target_bearing is not None:
                template = templates.get("side", "前方{distance}米处有{object}，建议{direction}侧避让")
                return template.format(object=class_name, distance=distance_text, direction=direction_text)
            else:
                template = templates.get("close", "注意，前方{distance}米处有{object}")
                return template.format(object=class_name, distance=distance_text)
        
        else:  # L0
            if len(all_risks) > 1:
                return templates.get("multiple", "前方有多个指引标识")
            else:
                template = templates.get("guide", "前方{distance}米处有{object}，建议跟随")
                return template.format(object=class_name, distance=distance_text)
    
    def _determine_action_type(self, highest_risk: ObjectRisk, 
                             all_risks: List[ObjectRisk]) -> ActionType:
        """确定动作类型"""
        risk_level = highest_risk.risk_level
        
        # L3级别：强制停止
        if risk_level == RiskLevel.L3:
            return ActionType.STOP
        
        # L2级别：高危预警
        elif risk_level == RiskLevel.L2:
            if (highest_risk.time_to_collide and 
                highest_risk.time_to_collide < 3):
                return ActionType.STOP
            else:
                return ActionType.BRAKE_AND_WARN
        
        # L1级别：避让绕行
        elif risk_level == RiskLevel.L1:
            if highest_risk.distance_risk > 0.8:  # 距离很近
                return ActionType.SIDE_STEP
            else:
                return ActionType.BRAKE_AND_WARN
        
        # L0级别：跟随指引
        else:
            return ActionType.FOLLOW_GUIDE
    
    def _get_vibration_pattern(self, action: ActionType, risk_level: RiskLevel) -> str:
        """获取振动模式"""
        if action == ActionType.STOP:
            return "emergency"  # 紧急振动
        elif action == ActionType.BRAKE_AND_WARN:
            return "warning"    # 警告振动
        elif action == ActionType.SIDE_STEP:
            return "short"      # 短振动
        elif action == ActionType.FOLLOW_GUIDE:
            return "gentle"     # 轻柔振动
        else:
            return "none"       # 无振动
    
    def _smooth_action_decision(self, new_action: ActionType) -> ActionType:
        """平滑动作决策，避免频繁切换"""
        self.action_history.append(new_action)
        
        # 保持历史记录长度
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
        
        # 如果历史记录不足，直接返回新动作
        if len(self.action_history) < 3:
            return new_action
        
        # 统计最近几次动作
        recent_actions = self.action_history[-3:]
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 如果有动作出现2次以上，使用该动作
        for action, count in action_counts.items():
            if count >= 2:
                return action
        
        # 否则返回新动作
        return new_action
    
    def plan_action(self, risks: List[ObjectRisk], 
                   frame_info: Optional[Dict[str, Any]] = None) -> ActionPlan:
        """规划动作方案"""
        # 如果没有风险物体，继续前进
        if not risks:
            return ActionPlan(
                action=ActionType.CONTINUE,
                priority=1,
                speech_text="前方安全，可以继续前进",
                confidence=1.0
            )
        
        # 智能风险去重和优先级选择
        filtered_risks = self._filter_and_prioritize_risks(risks)
        
        # 获取最高风险物体
        highest_risk = filtered_risks[0]
        
        # 确定动作类型
        raw_action = self._determine_action_type(highest_risk, filtered_risks)
        action = self._smooth_action_decision(raw_action)
        
        # 计算目标方位角和避让方向
        target_bearing = None
        avoidance_direction = None
        if action == ActionType.SIDE_STEP and frame_info:
            target_bearing, avoidance_direction = self._calculate_avoidance_bearing(
                highest_risk, filtered_risks, frame_info
            )
        
        # 生成语音文本
        speech_text = self._generate_speech_text(highest_risk, filtered_risks, target_bearing, avoidance_direction)
        
        # 获取振动模式
        vibration_pattern = self._get_vibration_pattern(action, highest_risk.risk_level)
        
        # 确定优先级
        priority = highest_risk.risk_level.value + 1
        
        # 计算决策置信度
        confidence = min(1.0, highest_risk.risk_score * 1.2)
        
        return ActionPlan(
            action=action,
            priority=priority,
            speech_text=speech_text,
            target_bearing=target_bearing,
            vibration_pattern=vibration_pattern,
            confidence=confidence
        )
    
    def _filter_and_prioritize_risks(self, risks: List[ObjectRisk]) -> List[ObjectRisk]:
        """
        智能风险去重和优先级选择
        确保只播报最危险的单个风险信号
        """
        if not risks:
            return risks
        
        # 第一步：按风险等级和分数排序
        sorted_risks = sorted(risks, key=lambda x: (x.risk_level.value, x.risk_score), reverse=True)
        
        # 第二步：去重相似风险对象
        filtered_risks = []
        seen_objects = set()
        
        for risk in sorted_risks:
            # 创建对象标识符（类别+大致位置）
            distance_key = "close" if hasattr(risk, 'distance_m') and risk.distance_m and risk.distance_m < 3.0 else "far"
            object_key = f"{risk.class_name}_{distance_key}"
            
            # 如果是相同类型的对象在相似位置，只保留风险最高的
            if object_key not in seen_objects:
                filtered_risks.append(risk)
                seen_objects.add(object_key)
        
        # 第三步：应用智能优先级规则
        if len(filtered_risks) > 1:
            # 检查是否有L3级别的紧急风险
            l3_risks = [r for r in filtered_risks if r.risk_level.value >= 3]
            if l3_risks:
                return [l3_risks[0]]  # 只播报最紧急的
            
            # 检查是否有L2级别的高危风险
            l2_risks = [r for r in filtered_risks if r.risk_level.value >= 2]
            if l2_risks:
                # 如果有碰撞预警，优先播报
                collision_risks = [r for r in l2_risks if r.time_to_collide and r.time_to_collide < 3.0]
                if collision_risks:
                    return [collision_risks[0]]
                
                # 如果有高速风险，优先播报
                high_speed_risks = [r for r in l2_risks if r.velocity_risk > 0.7]
                if high_speed_risks:
                    return [high_speed_risks[0]]
                
                # 否则播报距离最近的L2风险
                return [l2_risks[0]]
            
            # 如果只有L1和L0风险，检查是否有多个障碍物需要综合播报
            if len(filtered_risks) > 3:
                # 创建综合风险对象
                combined_risk = ObjectRisk(
                    class_id=-1,
                    class_name="多个障碍物",
                    risk_level=filtered_risks[0].risk_level,
                    risk_score=filtered_risks[0].risk_score,
                    distance_risk=filtered_risks[0].distance_risk,
                    velocity_risk=max(r.velocity_risk for r in filtered_risks[:3])
                )
                return [combined_risk]
        
        return filtered_risks[:1]  # 最多只返回一个风险对象
    
    def reset_history(self):
        """重置动作历史记录"""
        self.action_history.clear()
        logger.info("动作历史记录已重置")
