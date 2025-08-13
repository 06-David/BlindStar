#!/usr/bin/env python3
"""
BlindStar 盲人导航助手
专门为盲人用户设计的导航辅助功能
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .navigation_context import NavigationContextManager, GPSLocation
from .accessibility_helper import AccessibilityHelper, AccessibilityLevel
from .tts_engine import TTSEngine
from .stt_engine import STTEngine

logger = logging.getLogger(__name__)


class NavigationMode(Enum):
    """导航模式"""
    EXPLORATION = "exploration"    # 探索模式 - 自由探索周围环境
    DESTINATION = "destination"    # 目的地模式 - 前往特定目的地
    GUIDANCE = "guidance"          # 引导模式 - 跟随引导到达目标


@dataclass
class BlindNavigationConfig:
    """盲人导航配置"""
    # 语音设置
    speech_rate: int = 120
    speech_volume: float = 0.9
    detailed_descriptions: bool = True
    
    # 安全设置
    obstacle_warning_distance: float = 5.0
    emergency_stop_distance: float = 2.0
    safe_walking_speed: float = 1.0  # 米/秒
    
    # 导航设置
    instruction_frequency: float = 10.0  # 每10米播报一次
    landmark_announcement: bool = True
    intersection_announcement: bool = True
    
    # 无障碍设置
    prefer_accessible_routes: bool = True
    avoid_stairs: bool = True
    announce_accessibility_features: bool = True


class BlindNavigationAssistant:
    """盲人导航助手"""
    
    def __init__(self, config: BlindNavigationConfig = None):
        self.config = config or BlindNavigationConfig()
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.nav_context = NavigationContextManager()
        self.accessibility = AccessibilityHelper(AccessibilityLevel.FULL)
        self.tts_engine = None
        self.stt_engine = None
        
        # 状态管理
        self.current_mode = NavigationMode.EXPLORATION
        self.is_active = False
        self.last_announcement_time = 0
        self.last_position = None
        self.walking_speed = 0.0
        
        # 环境感知
        self.nearby_obstacles = []
        self.nearby_landmarks = []
        self.current_intersection = None
        
        # 语音交互历史
        self.conversation_history = []
        self.pending_confirmations = []
        
        self.logger.info("[盲人导航] 导航助手初始化完成")
    
    def initialize_engines(self, tts_engine: TTSEngine = None, stt_engine: STTEngine = None):
        """初始化语音引擎"""
        self.tts_engine = tts_engine or TTSEngine(accessibility_level="full")
        self.stt_engine = stt_engine or STTEngine()
        
        # 配置TTS引擎
        if self.tts_engine:
            voice_settings = self.accessibility.get_voice_settings()
            self.tts_engine.engine.setProperty('rate', voice_settings['rate'])
            self.tts_engine.engine.setProperty('volume', voice_settings['volume'])
        
        self.logger.info("[盲人导航] 语音引擎初始化完成")
    
    def start_navigation_session(self, mode: NavigationMode = NavigationMode.EXPLORATION):
        """开始导航会话"""
        self.current_mode = mode
        self.is_active = True
        
        # 播报欢迎信息
        welcome_message = self._get_welcome_message(mode)
        self._speak_with_priority(welcome_message, priority="high")
        
        # 播报当前位置信息
        self._announce_current_context()
        
        self.logger.info(f"[盲人导航] 导航会话开始，模式: {mode.value}")
    
    def stop_navigation_session(self):
        """停止导航会话"""
        self.is_active = False
        
        goodbye_message = "导航会话已结束。感谢使用BlindStar导航助手。"
        self._speak_with_priority(goodbye_message, priority="high")
        
        self.logger.info("[盲人导航] 导航会话结束")
    
    def update_location(self, location: GPSLocation, obstacles: List[Dict] = None,
                       landmarks: List[Dict] = None):
        """更新位置和环境信息"""
        if not self.is_active:
            return
        
        # 更新导航上下文
        self.nav_context.update_location(location)
        
        # 计算行走速度
        if self.last_position:
            distance = location.distance_to(self.last_position)
            time_diff = location.timestamp - self.last_position.timestamp
            if time_diff > 0:
                self.walking_speed = distance / time_diff
        
        self.last_position = location
        
        # 更新环境信息
        if obstacles:
            self.nearby_obstacles = obstacles
            self._check_obstacle_warnings()
        
        if landmarks:
            self.nearby_landmarks = landmarks
            self._check_landmark_announcements()
        
        # 检查是否需要播报导航指令
        self._check_navigation_announcements()
    
    def process_voice_command(self, command_text: str) -> bool:
        """处理语音命令"""
        if not command_text.strip():
            return False
        
        # 使用增强的语音命令解析
        if self.stt_engine:
            parsed = self.stt_engine.parse_enhanced_voice_command(command_text)
        else:
            parsed = {'command_type': 'unknown', 'parsed_command': None}
        
        command_type = parsed.get('command_type', 'unknown')
        command_info = parsed.get('parsed_command', {})
        
        self.logger.info(f"[盲人导航] 处理语音命令: {command_text} -> {command_type}")
        
        # 处理不同类型的命令
        if command_type == 'navigation':
            return self._handle_navigation_command(command_info)
        elif command_type == 'accessibility':
            return self._handle_accessibility_command(command_info)
        elif command_type == 'poi':
            return self._handle_poi_command(command_info)
        elif command_type == 'system':
            return self._handle_system_command(command_info)
        else:
            return self._handle_unknown_command(command_text)
    
    def _handle_navigation_command(self, command_info: Dict) -> bool:
        """处理导航命令"""
        command = command_info.get('command', command_info.get('type', ''))
        
        if command == 'set_destination':
            destination = command_info.get('destination', command_info.get('target', ''))
            return self._set_destination_by_name(destination)
        
        elif command == 'start_navigation':
            return self._start_navigation()
        
        elif command == 'stop_navigation':
            return self._stop_navigation()
        
        elif command == 'current_location':
            return self._announce_current_location()
        
        elif command == 'remaining_distance':
            return self._announce_remaining_distance()
        
        elif command == 'next_instruction':
            return self._announce_next_instruction()
        
        else:
            self._speak_with_priority(f"不理解的导航命令: {command}")
            return False
    
    def _handle_accessibility_command(self, command_info: Dict) -> bool:
        """处理无障碍命令"""
        command = command_info.get('command', '')
        
        if command == 'repeat_last':
            return self._repeat_last_instruction()
        
        elif command == 'detailed_info':
            return self._provide_detailed_info()
        
        elif command == 'describe_environment':
            return self._describe_environment()
        
        elif command == 'safety_check':
            return self._perform_safety_check()
        
        elif command == 'direction_help':
            return self._provide_direction_help()
        
        elif command == 'emergency_stop':
            return self._emergency_stop()
        
        elif command == 'help_request':
            return self._provide_help()
        
        elif command in ['slow_down', 'speed_up']:
            return self._adjust_speech_speed(command)
        
        else:
            return False
    
    def _handle_poi_command(self, command_info: Dict) -> bool:
        """处理POI查询命令"""
        keyword = command_info.get('keyword', '')
        
        if not keyword:
            self._speak_with_priority("请说明您要查找的地点类型")
            return False
        
        # 这里应该调用POI查询服务
        self._speak_with_priority(f"正在查找附近的{keyword}...")
        
        # 模拟POI查询结果
        poi_results = [
            {'name': f'示例{keyword}', 'distance': 150, 'direction': '东北方向'}
        ]
        
        return self._announce_poi_results(poi_results, keyword)
    
    def _set_destination_by_name(self, destination_name: str) -> bool:
        """通过名称设置目的地"""
        if not destination_name:
            self._speak_with_priority("请说明您要去的地方")
            return False
        
        # 确认目的地
        confirmation = f"确认要导航到{destination_name}吗？请说是或否"
        self._speak_with_priority(confirmation, priority="high")
        
        # 这里应该等待用户确认，简化处理
        self.current_mode = NavigationMode.DESTINATION
        self._speak_with_priority(f"好的，开始导航到{destination_name}")
        
        return True
    
    def _announce_current_location(self) -> bool:
        """播报当前位置"""
        context = self.nav_context.get_context()
        
        if not context.current_location:
            self._speak_with_priority("当前位置信息不可用")
            return False
        
        location_info = {
            'current_location': f"纬度{context.current_location.latitude:.4f}，经度{context.current_location.longitude:.4f}",
            'landmarks': [landmark['name'] for landmark in self.nearby_landmarks[:3]],
            'accessibility': self._get_nearby_accessibility_features()
        }
        
        if self.tts_engine:
            self.tts_engine.speak_location_context(location_info)
        else:
            self._speak_with_priority(f"您当前位置：{location_info['current_location']}")
        
        return True
    
    def _describe_environment(self) -> bool:
        """描述周围环境"""
        description_parts = []
        
        # 障碍物信息
        if self.nearby_obstacles:
            obstacle_count = len(self.nearby_obstacles)
            description_parts.append(f"周围检测到{obstacle_count}个障碍物")
            
            # 描述最近的障碍物
            closest_obstacle = min(self.nearby_obstacles, key=lambda x: x.get('distance', float('inf')))
            distance = closest_obstacle.get('distance', 0)
            obstacle_type = closest_obstacle.get('type', '物体')
            direction = closest_obstacle.get('direction', '前方')
            
            description_parts.append(f"最近的是{direction}{distance:.1f}米处的{obstacle_type}")
        
        # 地标信息
        if self.nearby_landmarks:
            landmark_names = [landmark['name'] for landmark in self.nearby_landmarks[:3]]
            description_parts.append(f"附近地标包括：{', '.join(landmark_names)}")
        
        # 路面信息
        description_parts.append("路面状况良好，可以安全通行")
        
        if description_parts:
            full_description = "。".join(description_parts)
            self._speak_with_priority(full_description, priority="normal")
        else:
            self._speak_with_priority("周围环境相对空旷，未检测到明显障碍物")
        
        return True
    
    def _perform_safety_check(self) -> bool:
        """执行安全检查"""
        safety_issues = []
        
        # 检查近距离障碍物
        dangerous_obstacles = [
            obs for obs in self.nearby_obstacles 
            if obs.get('distance', float('inf')) < self.config.emergency_stop_distance
        ]
        
        if dangerous_obstacles:
            safety_issues.append("前方有近距离障碍物")
        
        # 检查行走速度
        if self.walking_speed > self.config.safe_walking_speed:
            safety_issues.append("当前行走速度较快，建议放慢")
        
        if safety_issues:
            warning = "安全提醒：" + "，".join(safety_issues)
            self._speak_with_priority(warning, priority="high")
        else:
            self._speak_with_priority("当前环境安全，可以继续前进")
        
        return True
    
    def _speak_with_priority(self, text: str, priority: str = "normal"):
        """带优先级的语音播报"""
        if self.tts_engine:
            self.tts_engine.speak_with_priority(text, priority=priority)
        else:
            logger.info(f"[语音播报] {text}")
    
    def _get_welcome_message(self, mode: NavigationMode) -> str:
        """获取欢迎信息"""
        mode_messages = {
            NavigationMode.EXPLORATION: "欢迎使用BlindStar导航助手。当前为探索模式，我将为您描述周围环境。",
            NavigationMode.DESTINATION: "欢迎使用BlindStar导航助手。当前为目的地模式，请告诉我您要去的地方。",
            NavigationMode.GUIDANCE: "欢迎使用BlindStar导航助手。当前为引导模式，我将为您提供详细的行走指引。"
        }
        
        return mode_messages.get(mode, "欢迎使用BlindStar导航助手。")
    
    def _announce_current_context(self):
        """播报当前上下文"""
        # 播报当前时间
        current_time = time.strftime("%H点%M分")
        self._speak_with_priority(f"现在是{current_time}")
        
        # 播报当前位置（如果可用）
        if self.nav_context.get_context().current_location:
            self._announce_current_location()
    
    def _get_nearby_accessibility_features(self) -> List[str]:
        """获取附近的无障碍设施"""
        # 这里应该查询实际的无障碍设施数据
        # 目前返回示例数据
        return ["盲道", "语音提示", "无障碍通道"]
    
    def _check_obstacle_warnings(self):
        """检查障碍物警告"""
        for obstacle in self.nearby_obstacles:
            distance = obstacle.get('distance', float('inf'))
            obstacle_type = obstacle.get('type', '障碍物')
            direction = obstacle.get('direction', '前方')
            
            if distance < self.config.emergency_stop_distance:
                if self.tts_engine:
                    self.tts_engine.speak_obstacle_warning(
                        obstacle_type, distance, direction, urgency="emergency"
                    )
                else:
                    self._speak_with_priority(f"危险！{direction}{distance:.1f}米有{obstacle_type}", priority="emergency")
            
            elif distance < self.config.obstacle_warning_distance:
                if self.tts_engine:
                    self.tts_engine.speak_obstacle_warning(
                        obstacle_type, distance, direction, urgency="high"
                    )
                else:
                    self._speak_with_priority(f"注意：{direction}{distance:.1f}米有{obstacle_type}", priority="high")
    
    def _check_landmark_announcements(self):
        """检查地标播报"""
        if not self.config.landmark_announcement:
            return
        
        current_time = time.time()
        if current_time - self.last_announcement_time < 10:  # 10秒内不重复播报
            return
        
        for landmark in self.nearby_landmarks:
            distance = landmark.get('distance', float('inf'))
            if distance < 50:  # 50米内的地标
                name = landmark.get('name', '地标')
                self._speak_with_priority(f"经过{name}")
                self.last_announcement_time = current_time
                break
    
    def _check_navigation_announcements(self):
        """检查导航播报"""
        if self.current_mode != NavigationMode.DESTINATION:
            return
        
        # 这里应该检查是否需要播报导航指令
        # 简化实现
        pass
    
    def _repeat_last_instruction(self) -> bool:
        """重复最后一条指令"""
        if self.tts_engine:
            return self.tts_engine.repeat_last_instruction()
        else:
            self._speak_with_priority("没有可重复的指令")
            return False
    
    def _provide_help(self) -> bool:
        """提供帮助信息"""
        help_text = self.accessibility.get_help_text()
        self._speak_with_priority(help_text, priority="normal")
        return True
    
    def _emergency_stop(self) -> bool:
        """紧急停止"""
        self._speak_with_priority("紧急停止！请立即停下", priority="emergency")
        return True
