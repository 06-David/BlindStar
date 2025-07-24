import speech_recognition as sr
import threading
from typing import Optional, Callable


class STTEngine:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = False
        self.callback = None
        self.setup_microphone()

    def setup_microphone(self):
        """配置麦克风参数"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def start_listening(self, callback: Callable[[str], None]):
        """开始持续监听语音命令"""
        if self.listening:
            return

        self.listening = True
        self.callback = callback
        thread = threading.Thread(target=self._listen_loop)
        thread.daemon = True
        thread.start()

    def stop_listening(self):
        """停止监听"""
        self.listening = False

    def _listen_loop(self):
        """监听循环"""
        while self.listening:
            try:
                with self.microphone as source:
                    print("正在聆听...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)

                text = self.recognize_speech(audio)
                if text and self.callback:
                    self.callback(text)

            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                print(f"语音识别错误: {e}")

    def recognize_speech(self, audio) -> Optional[str]:
        """识别语音内容"""
        try:
            # 优先使用在线识别（更准确）
            text = self.recognizer.recognize_google(audio, language='zh-CN')
            print(f"识别结果: {text}")
            return text
        except sr.UnknownValueError:
            print("无法识别语音")
        except sr.RequestError:
            try:
                # 在线失败时使用离线识别
                text = self.recognizer.recognize_sphinx(audio, language='zh-CN')
                print(f"离线识别结果: {text}")
                return text
            except:
                print("离线识别失败")
        return None