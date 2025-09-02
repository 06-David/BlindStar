#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时导航服务
提供真正的实时位置跟踪和导航指令播报
"""

import time
import threading
import logging
import math
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class NavigationState(Enum):
    """导航状态"""
    IDLE = "idle"           # 空闲
    PLANNING = "planning"   # 规划中
    NAVIGATING = "navigating"  # 导航中
    PAUSED = "paused"      # 暂停
    COMPLETED = "completed"  # 完成
    ERROR = "error"        # 错误

@dataclass
class NavigationStep:
    """导航步骤"""
    step_id: int
    instruction: str
    distance: int
    duration: int
    completed: bool = False
    announced: bool = False

class RealtimeNavigationService:
    """实时导航服务"""
    
    def __init__(self, location_service, routing_service, tts_service):
        self.location_service = location_service
        self.routing_service = routing_service
        self.tts_service = tts_service
        
        # 导航状态
        self.state = NavigationState.IDLE
        self.current_route = None
        self.current_step_index = 0
        self.navigation_steps: List[NavigationStep] = []
        
        # 位置跟踪
        self.current_location = None
        self.destination = None
        self.last_announcement_time = 0
        
        # 线程控制
        self.navigation_thread = None
        self.stop_navigation_flag = False
        
        # 配置参数
        self.update_interval = 5  # 位置更新间隔（秒）
        self.announcement_interval = 30  # 播报间隔（秒）
        self.step_completion_threshold = 50  # 步骤完成阈值（米）
        
        logger.info("✅ 实时导航服务初始化完成")
    
    def start_navigation(self, destination_coordinates, destination_name: str = "目的地") -> bool:
        """
        开始实时导航
        
        Args:
            destination_coordinates: 目的地坐标
            destination_name: 目的地名称
            
        Returns:
            是否成功开始导航
        """
        try:
            if self.state == NavigationState.NAVIGATING:
                logger.warning("导航已在进行中")
                return False
            
            logger.info(f"🧭 开始实时导航到: {destination_name}")
            self.state = NavigationState.PLANNING
            
            # 获取当前位置
            self.current_location = self.location_service.get_current_location()
            if not self.current_location:
                logger.error("❌ 无法获取当前位置")
                self.state = NavigationState.ERROR
                self.tts_service.speak_error("无法获取当前位置，导航启动失败")
                return False
            
            self.destination = destination_coordinates
            
            # 规划路径
            self.tts_service.speak_info(f"正在规划到{destination_name}的路径...")
            
            # 简单坐标类
            class Coordinates:
                def __init__(self, longitude, latitude):
                    self.longitude = longitude
                    self.latitude = latitude
            
            origin = Coordinates(self.current_location.longitude, self.current_location.latitude)
            dest = Coordinates(destination_coordinates.longitude, destination_coordinates.latitude)
            
            self.current_route = self.routing_service.plan_walking_route_enhanced(origin, dest)
            
            if not self.current_route:
                logger.error("❌ 路径规划失败")
                self.state = NavigationState.ERROR
                self.tts_service.speak_error("路径规划失败，无法开始导航")
                return False
            
            # 准备导航步骤
            self._prepare_navigation_steps()
            
            # 播报路径信息
            route_info = f"路径规划完成。总距离{self.current_route.distance}米，预计用时{self.current_route.duration//60}分{self.current_route.duration%60}秒"
            self.tts_service.speak_info(route_info)
            
            # 开始导航
            self.state = NavigationState.NAVIGATING
            self.stop_navigation_flag = False
            self.current_step_index = 0
            
            # 启动导航线程
            self.navigation_thread = threading.Thread(target=self._navigation_loop, daemon=True)
            self.navigation_thread.start()
            
            # 播报开始导航
            self.tts_service.speak_navigation("导航开始，请注意安全")
            
            # 播报第一个步骤
            if self.navigation_steps:
                first_step = self.navigation_steps[0]
                self.tts_service.speak_navigation(f"第一步：{first_step.instruction}")
                first_step.announced = True
                self.last_announcement_time = time.time()
            
            logger.info("✅ 实时导航启动成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动导航失败: {e}")
            self.state = NavigationState.ERROR
            self.tts_service.speak_error("导航启动失败")
            return False
    
    def _prepare_navigation_steps(self):
        """准备导航步骤"""
        self.navigation_steps = []
        
        for i, route_step in enumerate(self.current_route.steps):
            nav_step = NavigationStep(
                step_id=i,
                instruction=route_step.instruction,
                distance=route_step.distance,
                duration=route_step.duration
            )
            self.navigation_steps.append(nav_step)
        
        logger.info(f"准备了{len(self.navigation_steps)}个导航步骤")
    
    def _navigation_loop(self):
        """导航主循环"""
        try:
            while not self.stop_navigation_flag and self.state == NavigationState.NAVIGATING:
                # 更新位置
                self._update_location()
                
                # 检查步骤完成
                self._check_step_completion()
                
                # 定期播报
                self._periodic_announcement()
                
                # 检查是否到达目的地
                if self._check_destination_reached():
                    break
                
                # 等待下次更新
                time.sleep(self.update_interval)
            
            if self.state == NavigationState.NAVIGATING:
                self._complete_navigation()
                
        except Exception as e:
            logger.error(f"❌ 导航循环异常: {e}")
            self.state = NavigationState.ERROR
            self.tts_service.speak_error("导航过程中出现错误")
    
    def _update_location(self):
        """更新当前位置"""
        try:
            # 在实际应用中，这里应该获取实时GPS位置
            # 目前使用模拟位置更新
            logger.debug("更新位置信息...")
            
            # 模拟位置变化（实际应用中替换为真实GPS）
            if hasattr(self, '_simulation_step'):
                self._simulation_step += 1
            else:
                self._simulation_step = 0
            
            # 每2次更新模拟前进一步
            if self._simulation_step % 2 == 0:
                self._simulate_progress()
                
        except Exception as e:
            logger.error(f"位置更新失败: {e}")
    
    def _simulate_progress(self):
        """模拟导航进度（实际应用中替换为真实位置判断）"""
        if self.current_step_index < len(self.navigation_steps):
            current_step = self.navigation_steps[self.current_step_index]
            
            # 模拟步骤完成
            if not current_step.completed:
                # 模拟随机完成步骤
                import random
                if random.random() > 0.7:  # 30%概率完成当前步骤
                    current_step.completed = True
                    logger.info(f"模拟完成步骤 {current_step.step_id + 1}: {current_step.instruction}")
    
    def _check_step_completion(self):
        """检查步骤完成情况"""
        if self.current_step_index >= len(self.navigation_steps):
            return
        
        current_step = self.navigation_steps[self.current_step_index]
        
        if current_step.completed and not current_step.announced:
            # 当前步骤完成，播报下一步骤
            self.current_step_index += 1
            
            if self.current_step_index < len(self.navigation_steps):
                next_step = self.navigation_steps[self.current_step_index]
                self.tts_service.speak_navigation(f"第{self.current_step_index + 1}步：{next_step.instruction}")
                next_step.announced = True
                self.last_announcement_time = time.time()
                
                logger.info(f"播报步骤 {self.current_step_index + 1}: {next_step.instruction}")
            else:
                # 所有步骤完成
                logger.info("所有导航步骤已完成")
    
    def _periodic_announcement(self):
        """定期播报"""
        current_time = time.time()
        
        if current_time - self.last_announcement_time >= self.announcement_interval:
            if self.current_step_index < len(self.navigation_steps):
                current_step = self.navigation_steps[self.current_step_index]
                
                # 播报当前步骤和剩余距离
                remaining_steps = len(self.navigation_steps) - self.current_step_index
                announcement = f"当前：{current_step.instruction}。还有{remaining_steps}个步骤"
                
                self.tts_service.speak_navigation(announcement)
                self.last_announcement_time = current_time
                
                logger.info(f"定期播报: {announcement}")
    
    def _check_destination_reached(self):
        """检查是否到达目的地"""
        # 如果所有步骤都完成，认为到达目的地
        if self.current_step_index >= len(self.navigation_steps):
            return True
        
        # 实际应用中，这里应该计算当前位置与目的地的距离
        return False
    
    def _complete_navigation(self):
        """完成导航"""
        self.state = NavigationState.COMPLETED
        self.tts_service.speak_navigation("恭喜您已到达目的地！导航结束")
        logger.info("🎉 导航完成")
    
    def pause_navigation(self):
        """暂停导航"""
        if self.state == NavigationState.NAVIGATING:
            self.state = NavigationState.PAUSED
            self.tts_service.speak_info("导航已暂停")
            logger.info("⏸️ 导航暂停")
    
    def resume_navigation(self):
        """恢复导航"""
        if self.state == NavigationState.PAUSED:
            self.state = NavigationState.NAVIGATING
            self.tts_service.speak_info("导航已恢复")
            logger.info("▶️ 导航恢复")
    
    def stop_navigation(self):
        """停止导航"""
        self.stop_navigation_flag = True
        self.state = NavigationState.IDLE
        self.tts_service.speak_info("导航已停止")
        logger.info("⏹️ 导航停止")
    
    def get_navigation_status(self) -> dict:
        """获取导航状态"""
        return {
            "state": self.state.value,
            "current_step": self.current_step_index + 1 if self.current_step_index < len(self.navigation_steps) else None,
            "total_steps": len(self.navigation_steps),
            "current_instruction": self.navigation_steps[self.current_step_index].instruction if self.current_step_index < len(self.navigation_steps) else None,
            "route_distance": self.current_route.distance if self.current_route else None,
            "route_duration": self.current_route.duration if self.current_route else None
        }

# 全局实例
_global_realtime_nav_service = None

def get_realtime_navigation_service(location_service=None, routing_service=None, tts_service=None):
    """获取全局实时导航服务实例"""
    global _global_realtime_nav_service
    
    if _global_realtime_nav_service is None:
        if not all([location_service, routing_service, tts_service]):
            # 自动获取服务
            from src.navigation.enhanced_location import get_location_service
            from src.navigation.enhanced_routing import get_routing_service
            from src.voice.tts_service import get_tts_service
            
            location_service = location_service or get_location_service()
            routing_service = routing_service or get_routing_service()
            tts_service = tts_service or get_tts_service()
        
        _global_realtime_nav_service = RealtimeNavigationService(
            location_service, routing_service, tts_service
        )
    
    return _global_realtime_nav_service
