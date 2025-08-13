#!/usr/bin/env python3
"""
BlindStar 无障碍辅助模块
专门为盲人用户优化的功能增强
"""

import logging
import time
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AccessibilityLevel(Enum):
    """无障碍级别"""
    BASIC = "basic"          # 基础无障碍
    ENHANCED = "enhanced"    # 增强无障碍
    FULL = "full"           # 完全无障碍


@dataclass
class AccessibilityFeature:
    """无障碍功能配置"""
    # 语音设置
    speech_rate: int = 120           # 语速（词/分钟）
    speech_volume: float = 0.9       # 音量
    speech_pitch: int = 0            # 音调
    
    # 播报设置
    detailed_descriptions: bool = True    # 详细描述
    distance_units: str = "meters"       # 距离单位
    direction_precision: str = "detailed" # 方向精度：simple/detailed
    
    # 交互设置
    confirmation_required: bool = True    # 需要确认
    repeat_instructions: bool = True      # 重复指令
    emergency_priority: bool = True       # 紧急优先
    
    # 导航设置
    announce_landmarks: bool = True       # 播报地标
    announce_intersections: bool = True   # 播报路口
    announce_obstacles: bool = True       # 播报障碍物
    
    # 时间设置
    instruction_interval: float = 3.0    # 指令间隔
    warning_advance_time: float = 5.0    # 提前警告时间


class AccessibilityHelper:
    """无障碍辅助器"""
    
    def __init__(self, level: AccessibilityLevel = AccessibilityLevel.ENHANCED):
        self.level = level
        self.features = AccessibilityFeature()
        self.logger = logging.getLogger(__name__)
        
        # 根据级别调整设置
        self._adjust_settings_by_level()
        
        # 语音模板
        self.voice_templates = self._load_voice_templates()
        
        # 常用地点的无障碍信息
        self.accessibility_info = self._load_accessibility_info()
        
        self.logger.info(f"[无障碍] 初始化完成，级别: {level.value}")
    
    def _adjust_settings_by_level(self):
        """根据无障碍级别调整设置"""
        if self.level == AccessibilityLevel.BASIC:
            self.features.detailed_descriptions = False
            self.features.direction_precision = "simple"
            self.features.announce_landmarks = False
            
        elif self.level == AccessibilityLevel.ENHANCED:
            self.features.detailed_descriptions = True
            self.features.direction_precision = "detailed"
            self.features.announce_landmarks = True
            
        elif self.level == AccessibilityLevel.FULL:
            self.features.detailed_descriptions = True
            self.features.direction_precision = "detailed"
            self.features.announce_landmarks = True
            self.features.announce_intersections = True
            self.features.announce_obstacles = True
            self.features.speech_rate = 100  # 稍慢的语速
    
    def _load_voice_templates(self) -> Dict[str, str]:
        """加载语音模板"""
        return {
            # 导航指令模板
            "start_navigation": "开始导航到{destination}，预计距离{distance}",
            "turn_instruction": "前方{distance}后{direction}转",
            "continue_straight": "继续直行{distance}",
            "arrival": "已到达目的地{destination}",
            "rerouting": "正在重新规划路线",
            
            # 障碍物警告模板
            "obstacle_ahead": "注意，前方{distance}有{obstacle_type}",
            "obstacle_left": "注意，左侧{distance}有{obstacle_type}",
            "obstacle_right": "注意，右侧{distance}有{obstacle_type}",
            
            # POI播报模板
            "poi_found": "找到{count}个{poi_type}",
            "poi_nearest": "最近的{poi_type}是{name}，距离{distance}",
            "poi_direction": "{name}在您的{direction}方向，距离{distance}",
            
            # 位置信息模板
            "current_location": "您当前位置：{location}",
            "nearby_landmarks": "附近地标：{landmarks}",
            "intersection_ahead": "前方{distance}是{intersection_name}路口",
            
            # 确认和反馈模板
            "confirm_destination": "确认导航到{destination}吗？",
            "confirm_action": "确认{action}吗？",
            "action_completed": "{action}已完成",
            "action_failed": "{action}失败，{reason}",
            
            # 紧急情况模板
            "emergency_stop": "紧急停止！前方有危险",
            "emergency_obstacle": "危险！{direction}方向有{obstacle_type}",
            "safe_to_proceed": "安全，可以继续前进"
        }
    
    def _load_accessibility_info(self) -> Dict[str, Dict[str, Any]]:
        """加载无障碍信息数据库"""
        return {
            # 交通设施无障碍信息
            "地铁站": {
                "accessibility_features": ["电梯", "盲道", "语音提示"],
                "navigation_tips": "寻找盲道，跟随语音提示",
                "safety_notes": "注意站台边缘，听候车提示音"
            },
            "公交站": {
                "accessibility_features": ["盲道", "语音报站"],
                "navigation_tips": "寻找站牌附近的盲道",
                "safety_notes": "注意车辆进站声音"
            },
            
            # 公共设施无障碍信息
            "银行": {
                "accessibility_features": ["无障碍通道", "语音ATM"],
                "navigation_tips": "寻找主入口，通常有无障碍标识",
                "safety_notes": "注意自动门和台阶"
            },
            "医院": {
                "accessibility_features": ["无障碍通道", "导诊服务", "电梯"],
                "navigation_tips": "寻找咨询台获得帮助",
                "safety_notes": "医院人流较多，注意避让"
            },
            
            # 商业设施无障碍信息
            "超市": {
                "accessibility_features": ["无障碍通道", "购物车"],
                "navigation_tips": "入口通常有自动门",
                "safety_notes": "注意购物车和顾客"
            }
        }
    
    def format_navigation_instruction(self, instruction_type: str, **kwargs) -> str:
        """格式化导航指令"""
        try:
            template = self.voice_templates.get(instruction_type, "")
            if not template:
                return str(kwargs.get('fallback', ''))
            
            # 处理距离格式化
            if 'distance' in kwargs:
                kwargs['distance'] = self._format_distance(kwargs['distance'])
            
            # 处理方向格式化
            if 'direction' in kwargs:
                kwargs['direction'] = self._format_direction(kwargs['direction'])
            
            return template.format(**kwargs)
            
        except Exception as e:
            self.logger.error(f"[无障碍] 指令格式化失败: {e}")
            return str(kwargs.get('fallback', '导航指令'))
    
    def _format_distance(self, distance: float) -> str:
        """格式化距离描述"""
        if distance < 1:
            return "不到1米"
        elif distance < 10:
            return f"{int(distance)}米"
        elif distance < 100:
            return f"{int(distance/10)*10}米"
        elif distance < 1000:
            return f"{int(distance/50)*50}米"
        else:
            km = distance / 1000
            if km < 10:
                return f"{km:.1f}公里"
            else:
                return f"{int(km)}公里"
    
    def _format_direction(self, direction: str) -> str:
        """格式化方向描述"""
        if self.features.direction_precision == "simple":
            # 简化方向描述
            direction_map = {
                "北": "前方", "东北": "右前方", "东": "右侧", "东南": "右后方",
                "南": "后方", "西南": "左后方", "西": "左侧", "西北": "左前方"
            }
            return direction_map.get(direction, direction)
        else:
            # 详细方向描述
            return direction
    
    def enhance_poi_description(self, poi_info: Dict[str, Any]) -> str:
        """增强POI描述"""
        name = poi_info.get('name', '未知地点')
        distance = poi_info.get('distance', 0)
        poi_type = poi_info.get('type', '地点')
        
        # 基础描述
        description = f"{name}，距离{self._format_distance(distance)}"
        
        # 添加无障碍信息
        if poi_type in self.accessibility_info:
            accessibility = self.accessibility_info[poi_type]
            if self.features.detailed_descriptions:
                features = accessibility.get('accessibility_features', [])
                if features:
                    description += f"，具有{', '.join(features[:2])}等无障碍设施"
        
        return description
    
    def generate_safety_warning(self, obstacle_type: str, distance: float, 
                              direction: str = "前方") -> str:
        """生成安全警告"""
        # 根据距离确定紧急程度
        if distance < 2:
            urgency = "紧急"
            template_key = "emergency_obstacle"
        elif distance < 5:
            urgency = "注意"
            template_key = f"obstacle_{direction.lower()}" if direction.lower() in ['left', 'right'] else "obstacle_ahead"
        else:
            urgency = "提醒"
            template_key = "obstacle_ahead"
        
        # 格式化警告信息
        warning = self.format_navigation_instruction(
            template_key,
            distance=self._format_distance(distance),
            obstacle_type=obstacle_type,
            direction=direction,
            fallback=f"{urgency}：{direction}{self._format_distance(distance)}有{obstacle_type}"
        )
        
        return f"{urgency}：{warning}"
    
    def should_announce(self, announcement_type: str, last_announcement_time: float = 0) -> bool:
        """判断是否应该播报"""
        current_time = time.time()
        
        # 检查时间间隔
        if current_time - last_announcement_time < self.features.instruction_interval:
            return False
        
        # 检查播报类型设置
        type_settings = {
            'landmark': self.features.announce_landmarks,
            'intersection': self.features.announce_intersections,
            'obstacle': self.features.announce_obstacles
        }
        
        return type_settings.get(announcement_type, True)
    
    def get_voice_settings(self) -> Dict[str, Any]:
        """获取语音设置"""
        return {
            'rate': self.features.speech_rate,
            'volume': self.features.speech_volume,
            'pitch': self.features.speech_pitch
        }
    
    def process_voice_command(self, command: str) -> Dict[str, Any]:
        """处理语音命令，增加无障碍友好的解析"""
        command = command.strip().lower()
        
        # 无障碍友好的命令模式
        accessibility_patterns = {
            # 导航命令
            r'(带我去|帮我找|我要去|导航到)\s*(.+)': {
                'type': 'navigation',
                'action': 'set_destination',
                'destination': 2
            },
            
            # 查询命令
            r'(附近有什么|周围有什么|找找)\s*(.*)': {
                'type': 'poi_search',
                'action': 'search_nearby',
                'keyword': 2
            },
            
            # 位置查询
            r'(我在哪里|当前位置|这是哪里)': {
                'type': 'location',
                'action': 'announce_location'
            },
            
            # 重复指令
            r'(再说一遍|重复|没听清)': {
                'type': 'system',
                'action': 'repeat_last'
            },
            
            # 紧急停止
            r'(停止|暂停|等等)': {
                'type': 'system',
                'action': 'emergency_stop'
            },
            
            # 帮助命令
            r'(帮助|怎么用|说明)': {
                'type': 'system',
                'action': 'help'
            }
        }
        
        for pattern, command_info in accessibility_patterns.items():
            match = re.search(pattern, command)
            if match:
                result = {
                    'type': command_info['type'],
                    'action': command_info['action'],
                    'original_text': command,
                    'confidence': 0.9
                }
                
                # 提取参数
                if 'destination' in command_info:
                    result['destination'] = match.group(command_info['destination']).strip()
                elif 'keyword' in command_info:
                    keyword = match.group(command_info['keyword']).strip()
                    result['keyword'] = keyword if keyword else "所有地点"
                
                return result
        
        # 如果没有匹配，返回未知命令
        return {
            'type': 'unknown',
            'action': 'unknown',
            'original_text': command,
            'confidence': 0.1
        }
    
    def get_help_text(self) -> str:
        """获取帮助文本"""
        help_text = """
BlindStar 语音导航帮助：

导航命令：
- "带我去天安门" - 开始导航
- "我要去最近的银行" - 查找并导航
- "导航到北京大学" - 设置目的地

查询命令：
- "附近有什么餐厅" - 查找附近餐厅
- "周围有什么" - 查找所有附近地点
- "找找超市" - 查找特定类型地点

位置命令：
- "我在哪里" - 播报当前位置
- "这是哪里" - 获取位置信息

控制命令：
- "停止" - 紧急停止
- "再说一遍" - 重复上次播报
- "帮助" - 获取帮助信息

说话时请清晰发音，系统会自动识别您的指令。
        """
        return help_text.strip()
    
    def update_settings(self, **kwargs):
        """更新无障碍设置"""
        for key, value in kwargs.items():
            if hasattr(self.features, key):
                setattr(self.features, key, value)
                self.logger.info(f"[无障碍] 设置已更新: {key} = {value}")
            else:
                self.logger.warning(f"[无障碍] 未知设置项: {key}")
