#!/usr/bin/env python3
"""
BlindStar语音播报服务
为盲人用户提供文字转语音(TTS)功能
"""

import logging
import threading
import queue
import time
from typing import Optional, Callable
from enum import Enum

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ pyttsx3未安装，语音功能不可用")

logger = logging.getLogger(__name__)

class VoicePriority(Enum):
    """语音播报优先级"""
    EMERGENCY = 1    # 紧急情况
    NAVIGATION = 2   # 导航指令
    INFORMATION = 3  # 一般信息
    DEBUG = 4        # 调试信息

class TTSService:
    """文字转语音服务"""
    
    def __init__(self):
        self.engine = None
        self.is_initialized = False
        self.is_speaking = False
        self.speech_queue = queue.PriorityQueue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # 语音配置
        self.voice_config = {
            'rate': 150,        # 语速 (每分钟字数)
            'volume': 0.9,      # 音量 (0.0-1.0)
            'voice_id': None    # 语音ID (None=默认)
        }
        
        self._initialize()
    
    def _initialize(self):
        """初始化TTS引擎"""
        if not TTS_AVAILABLE:
            logger.warning("TTS库不可用，语音功能将被禁用")
            return
        
        try:
            self.engine = pyttsx3.init()
            
            # 设置语音属性
            self.engine.setProperty('rate', self.voice_config['rate'])
            self.engine.setProperty('volume', self.voice_config['volume'])
            
            # 获取可用语音
            voices = self.engine.getProperty('voices')
            if voices:
                # 优先选择中文语音
                chinese_voice = None
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                        chinese_voice = voice
                        break
                
                if chinese_voice:
                    self.engine.setProperty('voice', chinese_voice.id)
                    logger.info(f"✅ 使用中文语音: {chinese_voice.name}")
                else:
                    logger.info(f"✅ 使用默认语音: {voices[0].name}")
            
            # 启动工作线程
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()
            
            self.is_initialized = True
            logger.info("✅ TTS服务初始化成功")
            
        except Exception as e:
            logger.error(f"❌ TTS服务初始化失败: {e}")
            self.is_initialized = False
    
    def speak(self, text: str, priority: VoicePriority = VoicePriority.INFORMATION,
              interrupt: bool = False) -> bool:
        """
        语音播报文字（修复：直接播报，避免队列问题）

        Args:
            text: 要播报的文字
            priority: 播报优先级
            interrupt: 是否中断当前播报

        Returns:
            bool: 是否成功播报
        """
        if not self.is_initialized or not text.strip():
            return False

        try:
            speech_text = text.strip()

            # 如果是高优先级且需要中断，停止当前播报
            if interrupt and priority.value <= VoicePriority.NAVIGATION.value:
                self._stop_current_speech()

            # 直接播报，避免队列复杂性
            if priority.value <= VoicePriority.NAVIGATION.value:
                # 高优先级：立即播报
                self._do_speak_immediate(speech_text)
            else:
                # 低优先级：检查是否正在播报
                if not self.is_speaking:
                    self._do_speak_immediate(speech_text)
                else:
                    # 正在播报时，加入简单队列
                    queue_item = (priority.value, time.time(), speech_text)
                    self.speech_queue.put(queue_item)

            return True

        except Exception as e:
            logger.error(f"❌ 语音播报失败: {e}")
            return False
    
    def speak_immediate(self, text: str) -> bool:
        """立即播报（最高优先级，中断当前播报）"""
        return self.speak(text, VoicePriority.EMERGENCY, interrupt=True)
    
    def speak_navigation(self, text: str) -> bool:
        """播报导航信息"""
        return self.speak(text, VoicePriority.NAVIGATION, interrupt=False)
    
    def speak_info(self, text: str) -> bool:
        """播报一般信息"""
        return self.speak(text, VoicePriority.INFORMATION, interrupt=False)
    
    def is_busy(self) -> bool:
        """检查是否正在播报"""
        return self.is_speaking or not self.speech_queue.empty()
    
    def stop_all(self):
        """停止所有播报"""
        self._clear_queue()
        self._stop_current_speech()
    
    def _speech_worker(self):
        """语音播报工作线程"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取播报任务（阻塞等待）
                priority, timestamp, speech_text = self.speech_queue.get(timeout=1.0)

                if self.stop_event.is_set():
                    break

                # 执行播报
                self._do_speak(speech_text)

                # 标记任务完成
                self.speech_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ 语音播报工作线程错误: {e}")
    
    def _do_speak(self, text: str):
        """执行实际的语音播报（队列版本）"""
        try:
            self.is_speaking = True
            logger.info(f"🔊 语音播报: {text}")

            # 使用pyttsx3播报
            self.engine.say(text)
            self.engine.runAndWait()

        except Exception as e:
            logger.error(f"❌ 语音播报执行失败: {e}")
        finally:
            self.is_speaking = False

    def _do_speak_immediate(self, text: str):
        """立即执行语音播报（使用Windows SAPI）"""
        try:
            logger.info(f"🔊 立即语音播报: {text}")

            # 停止当前播报
            if self.is_speaking:
                self._stop_current_speech()

            self.is_speaking = True

            # 首先尝试Windows SAPI（之前可以工作的方法）
            if self._speak_with_sapi(text):
                logger.info(f"✅ SAPI语音播报完成: {text[:30]}...")
                return

            # SAPI失败时使用pyttsx3
            logger.info("SAPI失败，尝试pyttsx3")
            self.engine.say(text)
            self.engine.runAndWait()

            logger.info(f"✅ pyttsx3语音播报完成: {text[:30]}...")

        except Exception as e:
            logger.error(f"❌ 立即语音播报失败: {e}")
        finally:
            self.is_speaking = False

    def _speak_with_sapi(self, text: str) -> bool:
        """使用Windows SAPI直接播报（恢复之前可以工作的方法）"""
        try:
            import subprocess
            # 使用PowerShell调用Windows语音合成
            ps_command = f'Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak("{text}")'
            subprocess.run(['powershell', '-Command', ps_command], check=True, capture_output=True)
            logger.info(f"✅ SAPI播报完成: {text[:30]}...")
            return True
        except Exception as e:
            logger.error(f"❌ SAPI播报失败: {e}")
            return False
    
    def _clear_queue(self):
        """清空播报队列"""
        try:
            while not self.speech_queue.empty():
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
        except queue.Empty:
            pass
    
    def _stop_current_speech(self):
        """停止当前播报"""
        try:
            if self.engine and self.is_speaking:
                self.engine.stop()
        except Exception as e:
            logger.error(f"❌ 停止当前播报失败: {e}")
    
    def set_voice_config(self, rate: int = None, volume: float = None, voice_id: str = None):
        """设置语音配置"""
        try:
            if not self.is_initialized:
                return False
            
            if rate is not None:
                self.voice_config['rate'] = rate
                self.engine.setProperty('rate', rate)
            
            if volume is not None:
                self.voice_config['volume'] = volume
                self.engine.setProperty('volume', volume)
            
            if voice_id is not None:
                self.voice_config['voice_id'] = voice_id
                self.engine.setProperty('voice', voice_id)
            
            logger.info("✅ 语音配置已更新")
            return True
            
        except Exception as e:
            logger.error(f"❌ 设置语音配置失败: {e}")
            return False
    
    def get_available_voices(self):
        """获取可用语音列表"""
        try:
            if not self.is_initialized:
                return []
            
            voices = self.engine.getProperty('voices')
            voice_list = []
            
            for voice in voices:
                voice_list.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', []),
                    'gender': getattr(voice, 'gender', 'unknown')
                })
            
            return voice_list
            
        except Exception as e:
            logger.error(f"❌ 获取语音列表失败: {e}")
            return []
    
    def cleanup(self):
        """清理资源"""
        try:
            self.stop_event.set()
            self._clear_queue()
            self._stop_current_speech()
            
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
            
            if self.engine:
                self.engine.stop()
            
            logger.info("✅ TTS服务清理完成")
            
        except Exception as e:
            logger.error(f"❌ TTS服务清理失败: {e}")

# 全局TTS服务实例
_tts_service = None

def get_tts_service() -> TTSService:
    """获取TTS服务单例"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service

def speak(text: str, priority: VoicePriority = VoicePriority.INFORMATION) -> bool:
    """便捷的语音播报函数"""
    tts = get_tts_service()
    return tts.speak(text, priority)

def speak_navigation(text: str) -> bool:
    """播报导航信息"""
    tts = get_tts_service()
    return tts.speak_navigation(text)

def speak_info(text: str) -> bool:
    """播报一般信息"""
    tts = get_tts_service()
    return tts.speak_info(text)

def speak_immediate(text: str) -> bool:
    """立即播报"""
    tts = get_tts_service()
    return tts.speak_immediate(text)
