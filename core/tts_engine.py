import pyttsx3
import threading
import time
from typing import Optional


class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.last_speak_time = 0
        self.cooldown = 2.0  # 2秒冷却时间
        self.setup_voice()

    def setup_voice(self):
        """配置语音参数"""
        self.engine.setProperty('rate', 150)  # 语速
        self.engine.setProperty('volume', 0.8)  # 音量

        # 尝试设置中文语音
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'chinese' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

    def speak(self, text: str, blocking: bool = False):
        """语音播报文本"""
        current_time = time.time()
        if current_time - self.last_speak_time < self.cooldown:
            return  # 冷却时间内不播报

        self.last_speak_time = current_time

        if blocking:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            thread = threading.Thread(target=self._async_speak, args=(text,))
            thread.daemon = True
            thread.start()

    def _async_speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

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