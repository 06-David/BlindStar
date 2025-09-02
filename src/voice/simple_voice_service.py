#!/usr/bin/env python3
"""
简化的语音服务
使用Windows内置的语音合成和识别功能
"""

import logging
import subprocess
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class SimpleVoiceService:
    """简化的语音服务"""
    
    def __init__(self):
        self.is_speaking = False
        self.is_listening = False
        self.speech_callbacks = []
        
        logger.info("✅ 简化语音服务初始化完成")
    
    def speak(self, text: str, blocking: bool = True) -> bool:
        """
        使用Windows SAPI语音播报
        
        Args:
            text: 要播报的文字
            blocking: 是否阻塞等待播报完成
            
        Returns:
            bool: 是否成功播报
        """
        if not text.strip():
            return False
        
        try:
            self.is_speaking = True
            logger.info(f"🔊 语音播报: {text}")
            
            # 使用PowerShell调用Windows语音合成
            ps_command = f'''
            Add-Type -AssemblyName System.Speech
            $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $speak.Rate = 0
            $speak.Volume = 100
            $speak.Speak("{text}")
            '''
            
            if blocking:
                # 阻塞模式
                result = subprocess.run(
                    ['powershell', '-Command', ps_command], 
                    check=True, 
                    capture_output=True,
                    timeout=30
                )
                logger.info(f"✅ 阻塞语音播报完成: {text[:30]}...")
            else:
                # 异步模式
                def speak_async():
                    try:
                        subprocess.run(
                            ['powershell', '-Command', ps_command], 
                            check=True, 
                            capture_output=True,
                            timeout=30
                        )
                        logger.info(f"✅ 异步语音播报完成: {text[:30]}...")
                    except Exception as e:
                        logger.error(f"❌ 异步语音播报失败: {e}")
                    finally:
                        self.is_speaking = False
                
                thread = threading.Thread(target=speak_async, daemon=True)
                thread.start()
                
                if blocking:
                    thread.join()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 语音播报失败: {e}")
            return False
        finally:
            if blocking:
                self.is_speaking = False
    
    def speak_info(self, text: str) -> bool:
        """播报一般信息"""
        return self.speak(text, blocking=True)
    
    def speak_navigation(self, text: str) -> bool:
        """播报导航信息"""
        return self.speak(f"导航：{text}", blocking=True)
    
    def speak_immediate(self, text: str) -> bool:
        """立即播报（紧急信息）"""
        return self.speak(f"注意：{text}", blocking=True)
    
    def listen_for_command(self, timeout: int = 10) -> Optional[str]:
        """
        监听语音命令（使用Windows语音识别）
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            识别到的文字，如果失败返回None
        """
        try:
            self.is_listening = True
            logger.info("🎧 开始语音识别...")
            
            # 使用PowerShell调用Windows语音识别
            ps_command = f'''
            Add-Type -AssemblyName System.Speech
            $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
            $recognizer.SetInputToDefaultAudioDevice()
            $recognizer.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
            
            $result = $recognizer.Recognize([TimeSpan]::FromSeconds({timeout}))
            if ($result) {{
                Write-Output $result.Text
            }} else {{
                Write-Output "TIMEOUT"
            }}
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', ps_command],
                capture_output=True,
                text=True,
                timeout=timeout + 5
            )
            
            if result.returncode == 0:
                recognized_text = result.stdout.strip()
                if recognized_text and recognized_text != "TIMEOUT":
                    logger.info(f"🎧 识别结果: {recognized_text}")
                    return recognized_text
                else:
                    logger.info("🎧 语音识别超时")
                    return None
            else:
                logger.error(f"❌ 语音识别失败: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 语音识别异常: {e}")
            return None
        finally:
            self.is_listening = False
    
    def cleanup(self):
        """清理资源"""
        self.is_speaking = False
        self.is_listening = False
        logger.info("✅ 简化语音服务清理完成")

# 全局语音服务实例
_simple_voice_service = None

def get_simple_voice_service() -> SimpleVoiceService:
    """获取简化语音服务单例"""
    global _simple_voice_service
    if _simple_voice_service is None:
        _simple_voice_service = SimpleVoiceService()
    return _simple_voice_service
