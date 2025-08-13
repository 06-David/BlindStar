import vosk
import json
import threading
import queue
import pyaudio
from typing import Optional, Callable
import logging


class STTEngine:
    def __init__(self, model_path: str = "models/vosk-model-cn-0.22"):
        """
        初始化Vosk语音识别引擎
        
        Args:
            model_path (str): Vosk模型路径
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.listening = False
        self.callback = None
        
        # 初始化Vosk模型
        try:
            self.model = vosk.Model(model_path)
            self.rec = vosk.KaldiRecognizer(self.model, 16000)
            self.logger.info(f"Vosk模型加载成功: {model_path}")
        except Exception as e:
            self.logger.error(f"Vosk模型加载失败: {e}")
            raise
        
        # 音频设置
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        
        # 音频流
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 音频队列
        self.audio_queue = queue.Queue()

    def start_listening(self, callback: Callable[[str], None]):
        """开始持续监听语音命令"""
        if self.listening:
            return

        self.listening = True
        self.callback = callback
        
        # 启动音频流
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # 启动监听线程
        thread = threading.Thread(target=self._listen_loop)
        thread.daemon = True
        thread.start()
        
        self.logger.info("开始监听语音...")

    def stop_listening(self):
        """停止监听"""
        self.listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        self.logger.info("停止监听语音")

    def _listen_loop(self):
        """监听循环"""
        while self.listening:
            try:
                # 检查音频流是否已初始化
                if not self.stream:
                    continue
                    
                # 读取音频数据
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                
                # 发送到Vosk进行识别
                if self.rec.AcceptWaveform(data):
                    result = self.rec.Result()
                    text = self._parse_result(result)
                    
                    if text and self.callback:
                        self.logger.info(f"识别到语音: {text}")
                        self.callback(text)
                
            except Exception as e:
                self.logger.error(f"语音识别错误: {e}")

    def _parse_result(self, result: str) -> Optional[str]:
        """解析Vosk识别结果"""
        try:
            data = json.loads(result)
            text = data.get("text", "").strip()
            return text if text else None
        except json.JSONDecodeError:
            self.logger.error(f"解析识别结果失败: {result}")
            return None

    def recognize_speech(self, audio_data: bytes) -> Optional[str]:
        """识别音频数据"""
        try:
            if self.rec.AcceptWaveform(audio_data):
                result = self.rec.Result()
                return self._parse_result(result)
            return None
        except Exception as e:
            self.logger.error(f"语音识别失败: {e}")
            return None

    def __del__(self):
        """清理资源"""
        self.stop_listening()