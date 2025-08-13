#!/usr/bin/env python3
"""
风险评估模块 - BlindStar决策系统核心组件
负责根据YOLO检测结果、深度信息和速度数据评估环境风险等级
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from .depth_hazard_detector import DepthHazardDetector, DepthHazard, DepthHazardType

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级枚举"""
    L0 = 0  # 🟢 提示指引 - 路面标线、导盲砖
    L1 = 1  # 🟡 避让绕行 - 行人、自行车、路灯杆
    L2 = 2  # 🟠 高危预警 - 汽车、摩托车、卡车
    L3 = 3  # 🔴 强制停止 - 红灯、停止标志、护栏


class ObjectRisk:
    """单个物体的风险评估结果"""
    def __init__(self, 
                 class_id: int,
                 class_name: str,
                 risk_level: RiskLevel,
                 risk_score: float,
                 distance_risk: float,
                 velocity_risk: float,
                 time_to_collide: Optional[float] = None,
                 is_depth_hazard: bool = False,
                 depth_hazard: Optional[DepthHazard] = None):
        self.class_id = class_id
        self.class_name = class_name
        self.risk_level = risk_level
        self.risk_score = risk_score  # 综合风险分数 0-1
        self.distance_risk = distance_risk  # 距离风险分数 0-1
        self.velocity_risk = velocity_risk  # 速度风险分数 0-1
        self.time_to_collide = time_to_collide  # 碰撞时间(秒)
        self.is_depth_hazard = is_depth_hazard  # 是否为深度危险
        self.depth_hazard = depth_hazard  # 深度危险详细信息


class RiskAssessor:
    """风险评估器 - 评估检测物体的风险等级和危险程度"""
    
    def __init__(self):
        """初始化风险评估器"""
        # COCO类别到风险等级的映射
        self.class_risk_mapping = self._init_class_risk_mapping()
        
        # 风险评估参数
        self.safe_distance_threshold = 2.0  # 安全距离阈值(米)
        self.danger_distance_threshold = 1.0  # 危险距离阈值(米)
        self.high_velocity_threshold = 3.0  # 高速度阈值(m/s) - 提高阈值，约10.8km/h
        self.extreme_velocity_threshold = 8.0  # 极高速度阈值(m/s) - 约28.8km/h
        self.collision_time_threshold = 3.0  # 碰撞时间预警阈值(秒)
        self.speed_risk_promotion_threshold = 0.7  # 速度风险提升阈值
        
        # 初始化深度危险检测器
        self.depth_hazard_detector = DepthHazardDetector()
        
        logger.info("✅ 风险评估器初始化完成")
    
    def _init_class_risk_mapping(self) -> Dict[int, RiskLevel]:
        """初始化COCO类别到风险等级的映射"""
        mapping = {}
        
        # L0 - 提示指引类 (路面标线、导盲砖等)
        l0_classes = []  # COCO数据集中没有直接对应的类别，需要自定义模型
        
        # L1 - 避让绕行类 (行人、动物、静态障碍物)
        l1_classes = [
            0,   # person - 行人
            15,  # cat - 猫
            16,  # dog - 狗
            17,  # horse - 马
            18,  # sheep - 羊
            19,  # cow - 牛
            20,  # elephant - 大象
            21,  # bear - 熊
            22,  # zebra - 斑马
            23,  # giraffe - 长颈鹿
            1,   # bicycle - 自行车
            36,  # skateboard - 滑板
            41,  # surfboard - 冲浪板
            42,  # tennis racket - 网球拍
            64,  # potted plant - 盆栽
            65,  # bed - 床
            70,  # toilet - 厕所
            72,  # tv - 电视
        ]
        
        # L2 - 高危预警类 (机动车辆)
        l2_classes = [
            2,   # car - 汽车
            3,   # motorcycle - 摩托车
            5,   # bus - 公交车
            7,   # truck - 卡车
            4,   # airplane - 飞机
            6,   # train - 火车
            8,   # boat - 船
        ]
        
        # L3 - 强制停止类 (交通信号、危险标志)
        l3_classes = [
            9,   # traffic light - 交通灯 (需要结合颜色检测)
            11,  # stop sign - 停止标志
            12,  # parking meter - 停车计时器
        ]
        
        # 构建映射字典
        for class_id in l0_classes:
            mapping[class_id] = RiskLevel.L0
        for class_id in l1_classes:
            mapping[class_id] = RiskLevel.L1
        for class_id in l2_classes:
            mapping[class_id] = RiskLevel.L2
        for class_id in l3_classes:
            mapping[class_id] = RiskLevel.L3
        
        # 默认未分类物体为L1级别
        return mapping
    
    def get_class_risk_level(self, class_id: int, class_name: str) -> RiskLevel:
        """获取类别的基础风险等级"""
        # 特殊处理交通灯 - 需要根据颜色判断
        if class_id == 9:  # traffic light
            if 'red' in class_name.lower():
                return RiskLevel.L3
            elif 'green' in class_name.lower():
                return RiskLevel.L0
            else:
                return RiskLevel.L1  # 黄灯或未知颜色
        
        return self.class_risk_mapping.get(class_id, RiskLevel.L1)
    
    def calculate_distance_risk(self, distance_m: Optional[float]) -> float:
        """计算距离风险分数 (0-1)"""
        if distance_m is None or distance_m < 0:
            return 0.5  # 未知距离，中等风险
        
        if distance_m <= self.danger_distance_threshold:
            return 1.0  # 极高风险
        elif distance_m <= self.safe_distance_threshold:
            # 线性插值
            ratio = (self.safe_distance_threshold - distance_m) / \
                   (self.safe_distance_threshold - self.danger_distance_threshold)
            return 0.5 + 0.5 * ratio
        else:
            # 距离越远风险越低，但不为0
            return max(0.1, 1.0 / (distance_m * 0.5))
    
    def calculate_velocity_risk(self, velocity_mps: Optional[float]) -> float:
        """计算速度风险分数 (0-1)"""
        if velocity_mps is None:
            return 0.2  # 未知速度，低风险
        
        abs_velocity = abs(velocity_mps)
        
        # 极高速度：直接返回最高风险
        if abs_velocity >= self.extreme_velocity_threshold:
            return 1.0
        
        # 高速度：非线性增长风险
        elif abs_velocity >= self.high_velocity_threshold:
            # 在高速度和极高速度之间使用指数增长
            ratio = (abs_velocity - self.high_velocity_threshold) / \
                   (self.extreme_velocity_threshold - self.high_velocity_threshold)
            return 0.6 + 0.4 * (ratio ** 0.5)  # 0.6-1.0区间，平方根增长
        
        # 中等速度：线性增长
        elif abs_velocity >= 1.0:  # 1m/s以上开始有明显风险
            ratio = (abs_velocity - 1.0) / (self.high_velocity_threshold - 1.0)
            return 0.3 + 0.3 * ratio  # 0.3-0.6区间
        
        # 低速度：缓慢增长
        else:
            return abs_velocity * 0.3  # 0-0.3区间
    
    def calculate_time_to_collide(self, distance_m: Optional[float], 
                                velocity_mps: Optional[float]) -> Optional[float]:
        """计算碰撞时间 (秒)"""
        if distance_m is None or velocity_mps is None or distance_m <= 0:
            return None
        
        # 只有物体朝向摄像头移动时才计算碰撞时间
        if velocity_mps > 0.1:  # 正值表示靠近
            return distance_m / velocity_mps
        
        return None
    
    def assess_object_risk(self, detection_data: Dict[str, Any]) -> ObjectRisk:
        """评估单个检测物体的风险"""
        class_id = detection_data.get('class_id', -1)
        class_name = detection_data.get('class_name', 'unknown')
        distance_m = detection_data.get('distance_m')
        velocity_mps = detection_data.get('velocity_mps')
        
        # 获取基础风险等级
        base_risk_level = self.get_class_risk_level(class_id, class_name)
        
        # 计算各项风险分数
        distance_risk = self.calculate_distance_risk(distance_m)
        velocity_risk = self.calculate_velocity_risk(velocity_mps)
        time_to_collide = self.calculate_time_to_collide(distance_m, velocity_mps)
        
        # 计算综合风险分数
        base_score = base_risk_level.value / 3.0  # 基础分数 0-1
        
        # 动态调整风险等级
        dynamic_risk_level = base_risk_level
        
        # 调整权重：提高速度风险的影响力
        risk_score = base_score * 0.3 + distance_risk * 0.4 + velocity_risk * 0.3
        
        # 特殊情况1：即将碰撞的物体风险等级提升
        if time_to_collide is not None and time_to_collide < self.collision_time_threshold:
            if base_risk_level == RiskLevel.L1:
                dynamic_risk_level = RiskLevel.L2
            elif base_risk_level == RiskLevel.L0:
                dynamic_risk_level = RiskLevel.L1
            risk_score = min(1.0, risk_score * 1.5)
        
        # 特殊情况2：高速物体风险等级提升
        if velocity_risk >= self.speed_risk_promotion_threshold:
            # 根据速度风险程度提升风险等级
            if velocity_risk >= 0.9:  # 极高速度
                if base_risk_level == RiskLevel.L0:
                    dynamic_risk_level = RiskLevel.L2  # 直接跳到L2
                elif base_risk_level == RiskLevel.L1:
                    dynamic_risk_level = RiskLevel.L3  # 提升到最高级
                elif base_risk_level == RiskLevel.L2:
                    dynamic_risk_level = RiskLevel.L3  # 提升到最高级
                risk_score = min(1.0, risk_score * 1.8)
            elif velocity_risk >= 0.7:  # 高速度
                if base_risk_level == RiskLevel.L0:
                    dynamic_risk_level = RiskLevel.L1
                elif base_risk_level == RiskLevel.L1:
                    dynamic_risk_level = RiskLevel.L2
                elif base_risk_level == RiskLevel.L2:
                    dynamic_risk_level = RiskLevel.L3
                risk_score = min(1.0, risk_score * 1.3)
        
        # 创建ObjectRisk对象
        risk_obj = ObjectRisk(
            class_id=class_id,
            class_name=class_name,
            risk_level=dynamic_risk_level,
            risk_score=risk_score,
            distance_risk=distance_risk,
            velocity_risk=velocity_risk,
            time_to_collide=time_to_collide
        )
        
        # 手动添加距离信息属性
        risk_obj.distance_m = distance_m
        
        return risk_obj
    
    def _convert_depth_hazard_to_risk(self, depth_hazard: DepthHazard) -> ObjectRisk:
        """将深度危险转换为ObjectRisk"""
        # 根据深度危险类型确定风险等级
        hazard_risk_mapping = {
            DepthHazardType.CLIFF: RiskLevel.L3,        # 悬崖 - 强制停止
            DepthHazardType.DEEP_HOLE: RiskLevel.L2,    # 深坑 - 高危预警
            DepthHazardType.STEP_DOWN: RiskLevel.L2,    # 向下台阶 - 高危预警
            DepthHazardType.STEP_UP: RiskLevel.L1,      # 向上台阶 - 避让绕行
            DepthHazardType.SLOPE_DOWN: RiskLevel.L1,   # 下坡 - 避让绕行
            DepthHazardType.SLOPE_UP: RiskLevel.L1,     # 上坡 - 避让绕行
            DepthHazardType.SURFACE_BREAK: RiskLevel.L1, # 路面破损 - 避让绕行
            DepthHazardType.UNKNOWN_DEPTH: RiskLevel.L1  # 未知深度 - 避让绕行
        }
        
        risk_level = hazard_risk_mapping.get(depth_hazard.hazard_type, RiskLevel.L1)
        
        # 计算距离风险
        distance_risk = self.calculate_distance_risk(depth_hazard.distance_to_camera)
        
        # 深度危险没有速度，速度风险为0
        velocity_risk = 0.0
        
        # 综合风险分数基于严重程度和距离
        risk_score = depth_hazard.severity * 0.7 + distance_risk * 0.3
        
        # 创建ObjectRisk对象
        risk_obj = ObjectRisk(
            class_id=-1,  # 深度危险使用特殊ID
            class_name=depth_hazard.hazard_type.value,
            risk_level=risk_level,
            risk_score=risk_score,
            distance_risk=distance_risk,
            velocity_risk=velocity_risk,
            time_to_collide=None,
            is_depth_hazard=True,
            depth_hazard=depth_hazard
        )
        
        # 手动添加距离信息属性
        risk_obj.distance_m = depth_hazard.distance_to_camera
        
        return risk_obj
    
    def assess_frame_risks(self, detections: List[Dict[str, Any]], 
                         depth_map: Optional[np.ndarray] = None,
                         frame_info: Optional[Dict[str, Any]] = None) -> List[ObjectRisk]:
        """评估整帧中所有检测物体的风险，包括深度危险"""
        risks = []
        
        # 评估YOLO检测的物体风险
        for detection in detections:
            try:
                risk = self.assess_object_risk(detection)
                risks.append(risk)
            except Exception as e:
                logger.warning(f"评估物体风险时出错: {e}")
                continue
        
        # 检测并评估深度危险
        if depth_map is not None:
            try:
                depth_hazards = self.depth_hazard_detector.detect_depth_hazards(
                    depth_map, frame_info
                )
                
                for depth_hazard in depth_hazards:
                    depth_risk = self._convert_depth_hazard_to_risk(depth_hazard)
                    risks.append(depth_risk)
                    
                logger.debug(f"检测到 {len(depth_hazards)} 个深度危险")
                
            except Exception as e:
                logger.error(f"深度危险检测时出错: {e}")
        
        # 按风险等级和分数排序
        risks.sort(key=lambda x: (x.risk_level.value, x.risk_score), reverse=True)
        
        return risks
    
    def get_highest_risk(self, risks: List[ObjectRisk]) -> Optional[ObjectRisk]:
        """获取最高风险的物体"""
        if not risks:
            return None
        
        return risks[0]  # 已经按风险排序
    
    def filter_risks_by_level(self, risks: List[ObjectRisk], 
                            min_level: RiskLevel) -> List[ObjectRisk]:
        """筛选指定风险等级以上的物体"""
        return [risk for risk in risks if risk.risk_level.value >= min_level.value]
