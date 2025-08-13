import pyttsx3
import threading
import time
import logging
import os
import subprocess
from typing import Optional


class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.last_speak_time = 0
        self.cooldown = 0.5  # 降低冷却时间到0.5秒
        self.stt_engine = None  # STT引擎引用
        self._speaking_lock = threading.Lock()  # 添加线程锁
        self._is_speaking = False  # 播报状态标志
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

    def _get_direction(self, position_x: float) -> str:
        """根据物体水平位置判断方向"""
        if position_x < 0.3:
            return "左侧"
        elif position_x > 0.7:
            return "右侧"
        else:
            return "前方"