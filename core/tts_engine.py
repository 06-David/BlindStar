import pyttsx3
import threading
import time
import logging
import os
import subprocess
from typing import Optional, List, Dict, Any


class TTSEngine:
    def __init__(self, accessibility_level: str = "enhanced"):
        self.engine = pyttsx3.init()
        self.last_speak_time = 0
        self.cooldown = 0.5  # 降低冷却时间到0.5秒
        self.stt_engine = None  # STT引擎引用
        self._speaking_lock = threading.Lock()  # 添加线程锁
        self._is_speaking = False  # 播报状态标志

        # 无障碍功能
        self.accessibility_level = accessibility_level
        self.last_instruction = ""  # 记录最后一条指令，用于重复播报
        self.speech_queue = []  # 语音队列
        self.priority_levels = {
            'emergency': 0,    # 紧急情况
            'high': 1,         # 高优先级（导航指令）
            'normal': 2,       # 正常优先级
            'low': 3           # 低优先级（背景信息）
        }

        self.setup_voice()

    def set_stt_engine(self, stt_engine):
        """设置STT引擎引用，用于音频设备互斥"""
        self.stt_engine = stt_engine

    def setup_voice(self):
        """配置语音参数"""
        try:
            self.engine.setProperty('rate', 150)  # 语速
            self.engine.setProperty('volume', 1.0)  # 最大音量

            # 获取可用语音
            voices = self.engine.getProperty('voices')
            logging.info(f"[TTS] 找到 {len(voices)} 个可用语音")

            # 尝试设置中文语音
            chinese_voice_found = False
            for i, voice in enumerate(voices):
                logging.info(f"[TTS] 语音 {i}: {voice.name} - {voice.id}")
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    chinese_voice_found = True
                    logging.info(f"[TTS] 设置中文语音: {voice.name}")
                    break

            if not chinese_voice_found:
                logging.warning("[TTS] 未找到中文语音，使用默认语音")

        except Exception as e:
            logging.error(f"[TTS] 语音配置失败: {e}")

    def _speak_with_sapi(self, text: str):
        """使用Windows SAPI直接播报"""
        try:
            # 使用PowerShell调用Windows语音合成
            ps_command = f'Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak("{text}")'
            subprocess.run(['powershell', '-Command', ps_command], check=True, capture_output=True)
            logging.info("[TTS] SAPI播报完成")
            return True
        except Exception as e:
            logging.error(f"[TTS] SAPI播报失败: {e}")
            return False

    def speak(self, text: str, blocking: bool = False):
        """播报文本 - 带SAPI备用方案"""
        if not text.strip():
            return

        current_time = time.time()
        time_since_last = current_time - self.last_speak_time

        if time_since_last < self.cooldown:
            logging.info(f"[TTS] 冷却中，跳过播报: {text} (剩余{self.cooldown - time_since_last:.1f}秒)")
            return

        self.last_speak_time = current_time
        logging.info(f"[TTS] 开始播报: {text}")

        # 首先尝试SAPI直接播报
        if self._speak_with_sapi(text):
            logging.info("[TTS] SAPI播报成功")
            return

        # SAPI失败时使用pyttsx3
        try:
            logging.info("[TTS] SAPI失败，尝试pyttsx3")
            
            # 停止之前的播报
            self.engine.stop()
            
            # 添加文本
            self.engine.say(text)
            
            if blocking:
                # 阻塞模式
                logging.info("[TTS] pyttsx3阻塞模式播报")
                self.engine.runAndWait()
                logging.info("[TTS] pyttsx3阻塞播报完成")
            else:
                # 异步模式 - 使用简单线程
                def speak_async():
                    try:
                        logging.info("[TTS] pyttsx3异步播报开始")
                        self.engine.runAndWait()
                        logging.info("[TTS] pyttsx3异步播报完成")
                    except Exception as e:
                        logging.error(f"[TTS] pyttsx3异步播报失败: {e}")

                thread = threading.Thread(target=speak_async, daemon=True)
                thread.start()

        except Exception as e:
            logging.error(f"[TTS] 所有TTS方案都失败: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """停止TTS引擎"""
        try:
            self.engine.stop()
            logging.info("[TTS] TTS引擎已停止")
        except Exception as e:
            logging.error(f"[TTS] 停止TTS引擎失败: {e}")

    def set_rate(self, rate: int):
        """设置语速 (50-300)"""
        try:
            self.engine.setProperty('rate', rate)
            logging.info(f"[TTS] 语速设置为: {rate}")
        except Exception as e:
            logging.error(f"[TTS] 设置语速失败: {e}")

    def set_volume(self, volume: float):
        """设置音量 (0.0-1.0)"""
        try:
            self.engine.setProperty('volume', volume)
            logging.info(f"[TTS] 音量设置为: {volume}")
        except Exception as e:
            logging.error(f"[TTS] 设置音量失败: {e}")

    def speak_navigation(self, instruction: str, priority: str = "normal"):
        """
        导航专用语音播报

        Args:
            instruction: 导航指令
            priority: 优先级 ("low", "normal", "high", "emergency")
        """
        if not instruction.strip():
            return

        # 根据优先级调整冷却时间
        priority_cooldowns = {
            "emergency": 0.0,    # 紧急情况立即播报
            "high": 0.2,         # 高优先级短冷却
            "normal": 0.5,       # 正常冷却
            "low": 2.0           # 低优先级长冷却
        }

        original_cooldown = self.cooldown
        self.cooldown = priority_cooldowns.get(priority, 0.5)

        try:
            # 添加导航前缀以区分类型
            if priority == "emergency":
                prefixed_instruction = f"紧急：{instruction}"
            elif priority == "high":
                prefixed_instruction = f"注意：{instruction}"
            else:
                prefixed_instruction = instruction

            self.speak(prefixed_instruction, blocking=(priority == "emergency"))

        finally:
            # 恢复原始冷却时间
            self.cooldown = original_cooldown

    def speak_poi_results(self, poi_results: list, max_results: int = 3):
        """
        播报POI查询结果

        Args:
            poi_results: POI查询结果列表
            max_results: 最大播报数量
        """
        if not poi_results:
            self.speak("附近没有找到相关地点")
            return

        # 构建播报内容
        results_to_announce = poi_results[:max_results]

        if len(poi_results) > max_results:
            intro = f"找到{len(poi_results)}个地点，最近{max_results}个是："
        else:
            intro = f"找到{len(poi_results)}个地点："

        # 播报介绍
        self.speak(intro)

        # 逐个播报结果
        for i, poi in enumerate(results_to_announce):
            name = poi.get('name', '未知地点')
            distance = poi.get('distance', 0)

            if distance < 1000:
                distance_desc = f"{int(distance)}米"
            else:
                distance_desc = f"{distance/1000:.1f}公里"

            result_text = f"{i+1}. {name}，距离{distance_desc}"
            self.speak(result_text)

    def speak_navigation_status(self, status_info: dict):
        """
        播报导航状态信息

        Args:
            status_info: 导航状态信息字典
        """
        if not status_info.get('enabled', False):
            self.speak("导航功能未启用")
            return

        mode = status_info.get('mode', 'unknown')
        state = status_info.get('state', 'unknown')

        mode_names = {
            'assist': '辅助模式',
            'guide': '引导模式',
            'full': '完整导航模式'
        }

        state_names = {
            'idle': '空闲',
            'planning': '规划中',
            'navigating': '导航中',
            'rerouting': '重新规划',
            'arrived': '已到达'
        }

        mode_desc = mode_names.get(mode, mode)
        state_desc = state_names.get(state, state)

        status_text = f"当前导航状态：{mode_desc}，{state_desc}"

        # 添加距离信息
        distance = status_info.get('distance_to_destination')
        if distance is not None:
            if distance < 1000:
                distance_desc = f"{int(distance)}米"
            else:
                distance_desc = f"{distance/1000:.1f}公里"
            status_text += f"，距离目的地{distance_desc}"

        self.speak(status_text)

    def speak_detection_result(self, detections: list):
        """播报检测结果"""
        if not detections:
            self.speak("前方安全")
            return

        # 处理DetectionResult对象或字典
        processed_detections = []
        for detection in detections:
            if hasattr(detection, 'distance') and hasattr(detection, 'class_name'):
                # DetectionResult对象
                processed_detections.append({
                    'name': detection.class_name,
                    'distance': detection.distance if detection.distance is not None else 999,
                    'position_x': detection.center[0] if hasattr(detection, 'center') else 0.5
                })
            elif isinstance(detection, dict):
                # 字典格式
                processed_detections.append(detection)
            else:
                continue

        if not processed_detections:
            self.speak("前方安全")
            return

        # 找到最近物体
        closest = min(processed_detections, key=lambda x: x.get('distance', 999))
        direction = self._get_direction(closest.get('position_x', 0.5))

        # 生成播报文本
        distance = closest.get('distance', 0)
        name = closest.get('name', '物体')
        if distance < 1000:
            text = f"{direction}{distance:.1f}米处有{name}"
        else:
            text = f"{direction}{distance/1000:.1f}公里处有{name}"
        self.speak(text)

    def speak_with_priority(self, text: str, priority: str = "normal",
                           interrupt: bool = False) -> bool:
        """
        带优先级的语音播报 - 为盲人用户优化

        Args:
            text: 要播报的文本
            priority: 优先级 (emergency, high, normal, low)
            interrupt: 是否中断当前播报

        Returns:
            是否成功播报
        """
        if not text.strip():
            return False

        # 记录指令用于重复播报
        if priority in ['emergency', 'high']:
            self.last_instruction = text

        # 紧急情况立即播报
        if priority == 'emergency':
            return self._emergency_speak(text)

        # 高优先级可以中断当前播报
        if priority == 'high' and interrupt:
            return self._interrupt_and_speak(text)

        # 正常播报
        return self.speak(text)

    def _emergency_speak(self, text: str) -> bool:
        """紧急情况播报"""
        try:
            # 停止当前播报
            if self._is_speaking:
                self.engine.stop()

            # 立即播报
            emergency_text = f"紧急提醒：{text}"
            self.engine.say(emergency_text)
            self.engine.runAndWait()

            logging.info(f"[TTS] 紧急播报: {emergency_text}")
            return True

        except Exception as e:
            logging.error(f"[TTS] 紧急播报失败: {e}")
            return False

    def _interrupt_and_speak(self, text: str) -> bool:
        """中断当前播报并播报新内容"""
        try:
            if self._is_speaking:
                self.engine.stop()
                time.sleep(0.1)  # 短暂等待

            return self.speak(text)

        except Exception as e:
            logging.error(f"[TTS] 中断播报失败: {e}")
            return False

    def repeat_last_instruction(self) -> bool:
        """重复最后一条指令"""
        if self.last_instruction:
            repeat_text = f"重复：{self.last_instruction}"
            return self.speak(repeat_text)
        else:
            return self.speak("没有可重复的指令")

    def speak_obstacle_warning(self, obstacle_type: str, distance: float,
                             direction: str = "前方", urgency: str = "normal") -> bool:
        """
        障碍物警告播报 - 为盲人用户优化

        Args:
            obstacle_type: 障碍物类型
            distance: 距离
            direction: 方向
            urgency: 紧急程度

        Returns:
            是否成功播报
        """
        # 根据距离和障碍物类型确定紧急程度
        if distance < 2 or obstacle_type in ['车辆', '汽车', '卡车', 'car', 'truck']:
            priority = 'emergency'
            prefix = "危险"
        elif distance < 5:
            priority = 'high'
            prefix = "注意"
        else:
            priority = 'normal'
            prefix = "提醒"

        # 格式化距离
        if distance < 1:
            distance_text = "不到1米"
        elif distance < 10:
            distance_text = f"{int(distance)}米"
        else:
            distance_text = f"约{int(distance)}米"

        warning_text = f"{prefix}：{direction}{distance_text}有{obstacle_type}"

        return self.speak_with_priority(warning_text, priority=priority, interrupt=True)

    def announce_arrival(self, destination: str, accessibility_info: Dict[str, Any] = None) -> bool:
        """
        到达目的地播报 - 包含无障碍信息

        Args:
            destination: 目的地名称
            accessibility_info: 无障碍信息

        Returns:
            是否成功播报
        """
        arrival_text = f"已到达目的地：{destination}"

        # 添加无障碍信息
        if accessibility_info:
            if 'entrance_info' in accessibility_info:
                arrival_text += f"。{accessibility_info['entrance_info']}"

            if 'accessibility_features' in accessibility_info:
                features = accessibility_info['accessibility_features'][:2]
                if features:
                    arrival_text += f"。该地点具有{', '.join(features)}等无障碍设施"

        return self.speak_with_priority(arrival_text, priority="high")

    def _get_direction(self, position_x: float) -> str:
        """根据物体水平位置判断方向"""
        if position_x < 0.3:
            return "左侧"
        elif position_x > 0.7:
            return "右侧"
        else:
            return "前方"