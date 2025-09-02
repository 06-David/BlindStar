#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音识别服务 - 支持中文语音命令识别
"""

import logging
import threading
import time
import re
from typing import Optional, Dict, List, Callable
import queue

# 先定义logger
logger = logging.getLogger(__name__)

# 检查Vosk语音识别库
try:
    import vosk
    import json
    VOSK_AVAILABLE = True
    logger.info("✅ Vosk库已加载")
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("⚠️ Vosk库未安装")

# 检查SpeechRecognition库（备选方案）
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    logger.info("✅ SpeechRecognition库已加载")
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("⚠️ SpeechRecognition库未安装")

# 检查PyAudio库
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    logger.info("✅ PyAudio库已加载")
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("⚠️ PyAudio库未安装，语音识别功能受限")

# 检查Vosk模型
import os
VOSK_MODEL_PATH = "models/vosk-model-cn-0.22"
VOSK_MODEL_AVAILABLE = os.path.exists(VOSK_MODEL_PATH) and os.path.isdir(VOSK_MODEL_PATH)

if VOSK_MODEL_AVAILABLE:
    logger.info(f"✅ Vosk中文模型已找到: {VOSK_MODEL_PATH}")
else:
    logger.warning(f"⚠️ Vosk中文模型未找到: {VOSK_MODEL_PATH}")

class VoiceCommand:
    """语音命令类"""
    def __init__(self, command: str, patterns: List[str], action: str, description: str):
        self.command = command
        self.patterns = patterns  # 匹配模式列表
        self.action = action
        self.description = description

class SpeechRecognitionService:
    """语音识别服务"""
    
    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self.vosk_model = None
        self.vosk_rec = None
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.listen_thread = None
        self.tts_service = None
        self.recognition_method = None  # 'vosk', 'speechrecognition', 或 None

        # 选择模式相关变量
        self.choice_mode = False
        self.choice_result = None
        self.choice_max = 0

        # 语音命令映射
        self.voice_commands = self._init_voice_commands()

        # 优先尝试Vosk，然后是SpeechRecognition
        if self._init_vosk():
            self.recognition_method = 'vosk'
            logger.info("🎤 使用Vosk离线语音识别")
        elif SPEECH_RECOGNITION_AVAILABLE and PYAUDIO_AVAILABLE:
            if self._init_speech_recognition():
                self.recognition_method = 'speechrecognition'
                logger.info("🎤 使用SpeechRecognition在线语音识别")

        if not self.recognition_method:
            logger.warning("⚠️ 语音识别不可用，将使用键盘输入模式")
            self._show_installation_tips()

    def _init_vosk(self):
        """初始化Vosk语音识别"""
        try:
            if not VOSK_AVAILABLE:
                logger.debug("Vosk库不可用")
                return False

            if not VOSK_MODEL_AVAILABLE:
                logger.debug(f"Vosk模型不可用: {VOSK_MODEL_PATH}")
                return False

            if not PYAUDIO_AVAILABLE:
                logger.debug("PyAudio不可用，无法使用Vosk")
                return False

            # 初始化Vosk模型
            logger.info("🎤 正在加载Vosk中文语音识别模型...")
            self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, 16000)

            # 初始化音频设备
            self.microphone = pyaudio.PyAudio()

            logger.info("✅ Vosk语音识别初始化成功")
            return True

        except Exception as e:
            logger.error(f"❌ Vosk初始化失败: {e}")
            return False

    def _show_installation_tips(self):
        """显示安装提示"""
        if not VOSK_AVAILABLE:
            logger.info("💡 安装Vosk库: pip install vosk")
        if not VOSK_MODEL_AVAILABLE:
            logger.info(f"💡 下载Vosk中文模型到: {VOSK_MODEL_PATH}")
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.info("💡 安装SpeechRecognition库: pip install speechrecognition")
        if not PYAUDIO_AVAILABLE:
            logger.info("💡 安装PyAudio库: pip install pyaudio")
    
    def _init_speech_recognition(self):
        """初始化SpeechRecognition语音识别"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                logger.debug("SpeechRecognition库未安装")
                return False

            if not PYAUDIO_AVAILABLE:
                logger.debug("PyAudio未安装，无法使用SpeechRecognition")
                return False

            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            # 调整环境噪音
            logger.info("🎤 正在调整麦克风环境噪音...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            logger.info("✅ SpeechRecognition语音识别初始化成功")
            return True

        except Exception as e:
            logger.error(f"❌ SpeechRecognition初始化失败: {e}")
            return False
    
    def _init_voice_commands(self) -> Dict[str, VoiceCommand]:
        """初始化语音命令"""
        commands = {
            # 主菜单命令
            "get_location": VoiceCommand(
                "get_location",
                ["获取位置", "定位", "我在哪里", "当前位置", "位置"],
                "1",
                "获取当前位置"
            ),
            "correct_location": VoiceCommand(
                "correct_location", 
                ["位置校正", "校正位置", "手动定位", "输入地址", "校正"],
                "2",
                "位置校正功能"
            ),
            "search_hospital": VoiceCommand(
                "search_hospital",
                ["搜索医院", "找医院", "医院", "看病", "就医"],
                "3", 
                "搜索附近医院"
            ),
            "search_bank": VoiceCommand(
                "search_bank",
                ["搜索银行", "找银行", "银行", "取钱", "ATM"],
                "4",
                "搜索附近银行"
            ),
            "search_supermarket": VoiceCommand(
                "search_supermarket",
                ["搜索超市", "找超市", "超市", "购物", "买东西"],
                "5",
                "搜索附近超市"
            ),
            "search_restaurant": VoiceCommand(
                "search_restaurant", 
                ["搜索餐厅", "找餐厅", "餐厅", "吃饭", "用餐"],
                "6",
                "搜索附近餐厅"
            ),
            "navigation_status": VoiceCommand(
                "navigation_status",
                ["导航状态", "查看状态", "状态", "进度"],
                "7",
                "查看导航状态"
            ),
            "stop_navigation": VoiceCommand(
                "stop_navigation",
                ["停止导航", "结束导航", "取消导航", "停止"],
                "8", 
                "停止导航"
            ),
            "help": VoiceCommand(
                "help",
                ["帮助", "使用说明", "怎么用", "说明"],
                "9",
                "查看帮助"
            ),
            "exit": VoiceCommand(
                "exit",
                ["退出", "结束", "关闭", "再见", "拜拜"],
                "0",
                "退出系统"
            ),
            
            # 数字命令
            "number_1": VoiceCommand("number_1", ["一", "1", "第一个"], "1", "选择第一个"),
            "number_2": VoiceCommand("number_2", ["二", "2", "第二个"], "2", "选择第二个"),
            "number_3": VoiceCommand("number_3", ["三", "3", "第三个"], "3", "选择第三个"),
            "number_4": VoiceCommand("number_4", ["四", "4", "第四个"], "4", "选择第四个"),
            "number_5": VoiceCommand("number_5", ["五", "5", "第五个"], "5", "选择第五个"),
            
            # 确认命令
            "confirm": VoiceCommand(
                "confirm",
                ["确认", "是的", "好的", "确定", "对", "是"],
                "yes",
                "确认操作"
            ),
            "cancel": VoiceCommand(
                "cancel", 
                ["取消", "不", "不是", "否", "算了"],
                "no",
                "取消操作"
            ),
            
            # 重复命令
            "repeat": VoiceCommand(
                "repeat",
                ["重复", "再说一遍", "没听清", "重新说"],
                "repeat",
                "重复上一次播报"
            )
        }
        
        return commands
    
    def set_tts_service(self, tts_service):
        """设置TTS服务"""
        self.tts_service = tts_service
    
    def start_listening(self):
        """开始语音监听"""
        if not self.recognition_method:
            logger.warning("⚠️ 语音识别不可用")
            return False

        if self.is_listening:
            return True

        self.is_listening = True

        if self.recognition_method == 'vosk':
            self.listen_thread = threading.Thread(target=self._vosk_listen_loop, daemon=True)
        else:
            self.listen_thread = threading.Thread(target=self._sr_listen_loop, daemon=True)

        self.listen_thread.start()

        logger.info(f"🎤 开始语音监听 ({self.recognition_method})...")
        if self.tts_service:
            self.tts_service.speak_info("语音识别已启动，请说出您的命令")

        return True
    
    def stop_listening(self):
        """停止语音监听"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        logger.info("🎤 语音监听已停止")

    def pause_listening(self):
        """暂停语音监听（用于子程序）"""
        if self.is_listening:
            self.is_listening = False
            if self.listen_thread:
                self.listen_thread.join(timeout=2)
            logger.info("🎤 语音监听已暂停")
            return True
        return False

    def resume_listening(self):
        """恢复语音监听（从子程序返回）"""
        if not self.is_listening:
            logger.info("🎤 恢复语音监听...")
            return self.start_listening()
        return True

    def _vosk_listen_loop(self):
        """Vosk语音监听循环"""
        try:
            # 音频流配置
            CHUNK = 4096
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000

            # 打开音频流
            stream = self.microphone.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            logger.info("🎤 Vosk语音监听循环已启动")

            while self.is_listening:
                try:
                    # 读取音频数据
                    data = stream.read(CHUNK, exception_on_overflow=False)

                    # Vosk识别
                    if self.vosk_rec.AcceptWaveform(data):
                        # 完整识别结果
                        result = json.loads(self.vosk_rec.Result())
                        text = result.get('text', '').strip()

                        if text:
                            logger.info(f"🎤 Vosk识别到语音: {text}")

                            # 检查是否在选择模式
                            if self.choice_mode:
                                # 处理数字选择
                                choice = self._parse_number_choice(text, self.choice_max)
                                if choice:
                                    logger.info(f"✅ 识别到选择: {choice}")
                                    self.choice_result = choice
                                else:
                                    logger.debug(f"❓ 无效选择: {text}")
                                    self.choice_result = "invalid"
                            else:
                                # 正常命令模式
                                command = self._parse_voice_command(text)
                                if command:
                                    self.command_queue.put(command)
                                    logger.info(f"✅ 识别到命令: {command}")

                                    if self.tts_service:
                                        self.tts_service.speak_info(f"收到命令：{text}")
                                else:
                                    logger.debug(f"❓ 未识别的命令: {text}")
                                    if self.tts_service:
                                        self.tts_service.speak_info("抱歉，没有理解您的命令，请重新说一遍")

                except Exception as e:
                    logger.error(f"❌ Vosk识别异常: {e}")
                    time.sleep(0.1)

            # 清理资源
            stream.stop_stream()
            stream.close()

        except Exception as e:
            logger.error(f"❌ Vosk监听循环异常: {e}")
            self.is_listening = False
    
    def _sr_listen_loop(self):
        """SpeechRecognition语音监听循环"""
        while self.is_listening:
            try:
                # 监听语音
                with self.microphone as source:
                    logger.debug("🎤 等待语音输入...")
                    # 设置较短的超时时间，避免长时间阻塞
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # 识别语音
                try:
                    # 使用百度语音识别（支持中文）
                    text = self.recognizer.recognize_google(audio, language='zh-CN')
                    logger.info(f"🎤 识别到语音: {text}")
                    
                    # 解析命令
                    command = self._parse_voice_command(text)
                    if command:
                        self.command_queue.put(command)
                        logger.info(f"✅ 识别到命令: {command}")
                        
                        if self.tts_service:
                            self.tts_service.speak_info(f"收到命令：{text}")
                    else:
                        logger.debug(f"❓ 未识别的命令: {text}")
                        if self.tts_service:
                            self.tts_service.speak_info("抱歉，没有理解您的命令，请重新说一遍")
                            
                except Exception as sr_e:
                    if SPEECH_RECOGNITION_AVAILABLE and "UnknownValueError" in str(type(sr_e)):
                        logger.debug("🎤 未识别到语音")
                    elif SPEECH_RECOGNITION_AVAILABLE and "RequestError" in str(type(sr_e)):
                        logger.error(f"❌ 语音识别服务错误: {sr_e}")
                    else:
                        logger.error(f"❌ 语音识别异常: {sr_e}")

            except Exception as timeout_e:
                if SPEECH_RECOGNITION_AVAILABLE and "WaitTimeoutError" in str(type(timeout_e)):
                    # 超时是正常的，继续监听
                    continue
                else:
                    logger.error(f"❌ 语音监听异常: {timeout_e}")
                    time.sleep(1)
            except Exception as e:
                logger.error(f"❌ 语音监听异常: {e}")
                time.sleep(1)
    
    def _parse_voice_command(self, text: str) -> Optional[str]:
        """解析语音命令"""
        text = text.strip().lower()
        
        # 遍历所有命令，查找匹配的模式
        for cmd_name, cmd_obj in self.voice_commands.items():
            for pattern in cmd_obj.patterns:
                if pattern.lower() in text:
                    return cmd_obj.action
        
        return None
    
    def get_voice_command(self, timeout: float = 30.0) -> Optional[str]:
        """获取语音命令（阻塞式）"""
        if not self.recognition_method:
            return None

        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_voice_input(self, prompt: str = "请说话") -> Optional[str]:
        """获取语音输入（用于地址输入等）"""
        if not self.recognition_method:
            return None

        if self.tts_service:
            self.tts_service.speak_info(prompt)

        if self.recognition_method == 'vosk':
            return self._get_vosk_input(prompt)
        else:
            return self._get_sr_input(prompt)

    def _get_vosk_input(self, prompt: str) -> Optional[str]:
        """使用Vosk获取语音输入"""
        try:
            # 音频流配置
            CHUNK = 4096
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000

            # 打开音频流
            stream = self.microphone.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            logger.info(f"🎤 Vosk等待语音输入: {prompt}")

            # 重置识别器
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, 16000)

            start_time = time.time()
            timeout = 10.0
            silence_count = 0
            max_silence = 30  # 约1.5秒的静音

            while time.time() - start_time < timeout:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)

                    if self.vosk_rec.AcceptWaveform(data):
                        result = json.loads(self.vosk_rec.Result())
                        text = result.get('text', '').strip()

                        if text:
                            logger.info(f"🎤 Vosk识别到输入: {text}")

                            if self.tts_service:
                                self.tts_service.speak_info(f"您说的是：{text}")

                            stream.stop_stream()
                            stream.close()
                            return text

                        silence_count = 0  # 重置静音计数
                    else:
                        # 检查部分结果
                        partial_result = json.loads(self.vosk_rec.PartialResult())
                        partial_text = partial_result.get('partial', '').strip()

                        if partial_text:
                            silence_count = 0  # 有语音活动，重置静音计数
                        else:
                            silence_count += 1

                        # 如果静音时间过长，检查是否有最终结果
                        if silence_count > max_silence:
                            final_result = json.loads(self.vosk_rec.FinalResult())
                            final_text = final_result.get('text', '').strip()

                            if final_text:
                                logger.info(f"🎤 Vosk识别到最终输入: {final_text}")

                                if self.tts_service:
                                    self.tts_service.speak_info(f"您说的是：{final_text}")

                                stream.stop_stream()
                                stream.close()
                                return final_text

                            silence_count = 0  # 重置计数继续监听

                except Exception as e:
                    logger.error(f"❌ Vosk输入识别异常: {e}")
                    continue

            # 超时处理
            stream.stop_stream()
            stream.close()

            logger.warning("🎤 Vosk语音输入超时")
            if self.tts_service:
                self.tts_service.speak_info("语音输入超时，请重试")
            return None

        except Exception as e:
            logger.error(f"❌ Vosk语音输入异常: {e}")
            if self.tts_service:
                self.tts_service.speak_info("语音输入出现错误")
            return None

    def _get_sr_input(self, prompt: str) -> Optional[str]:
        """使用SpeechRecognition获取语音输入"""
        try:
            with self.microphone as source:
                logger.info(f"🎤 SpeechRecognition等待语音输入: {prompt}")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)

            # 识别语音
            text = self.recognizer.recognize_google(audio, language='zh-CN')
            logger.info(f"🎤 SpeechRecognition识别到输入: {text}")

            if self.tts_service:
                self.tts_service.speak_info(f"您说的是：{text}")

            return text

        except Exception as e:
            if SPEECH_RECOGNITION_AVAILABLE and "WaitTimeoutError" in str(type(e)):
                logger.warning("🎤 SpeechRecognition语音输入超时")
                if self.tts_service:
                    self.tts_service.speak_info("语音输入超时，请重试")
                return None
            elif SPEECH_RECOGNITION_AVAILABLE and "UnknownValueError" in str(type(e)):
                logger.warning("🎤 SpeechRecognition未识别到语音")
                if self.tts_service:
                    self.tts_service.speak_info("没有听清您的话，请重新说一遍")
                return None
            else:
                logger.error(f"❌ SpeechRecognition语音输入异常: {e}")
                if self.tts_service:
                    self.tts_service.speak_info("语音输入出现错误")
                return None
        except Exception as e:
            logger.error(f"❌ SpeechRecognition语音输入异常: {e}")
            if self.tts_service:
                self.tts_service.speak_info("语音输入出现错误")
            return None
    
    def get_voice_choice(self, prompt: str, max_choice: int = 5) -> Optional[str]:
        """获取语音选择（数字选择）"""
        if not self.recognition_method:
            return None

        if self.tts_service:
            self.tts_service.speak_info(f"{prompt}，请说出数字")

        if self.recognition_method == "vosk":
            return self._get_vosk_choice(max_choice)
        elif self.recognition_method == "speech_recognition":
            return self._get_sr_choice(max_choice)
        else:
            logger.error("❌ 未知的语音识别方法")
            return None

    def _get_vosk_choice(self, max_choice: int) -> Optional[str]:
        """使用Vosk获取语音选择（通过等待主监听循环的结果）"""
        logger.info("🎤 等待Vosk数字选择...")

        # 设置选择模式标志
        self.choice_mode = True
        self.choice_result = None
        self.choice_max = max_choice

        try:
            # 等待语音输入，最多等待15秒
            start_time = time.time()
            timeout = 15

            while time.time() - start_time < timeout:
                if self.choice_result is not None:
                    result = self.choice_result
                    self.choice_mode = False
                    self.choice_result = None
                    self.choice_max = 0

                    if result == "timeout":
                        logger.warning("🎤 Vosk选择超时")
                        if self.tts_service:
                            self.tts_service.speak_info("选择超时，请重试")
                        return None
                    elif result == "invalid":
                        logger.debug("❓ 无效选择")
                        if self.tts_service:
                            self.tts_service.speak_info("请说出有效的数字选择")
                        return None
                    else:
                        if self.tts_service:
                            self.tts_service.speak_info(f"您选择了：{result}")
                        return result

                time.sleep(0.1)

            # 超时
            self.choice_mode = False
            self.choice_result = None
            self.choice_max = 0
            logger.warning("🎤 Vosk选择超时")
            if self.tts_service:
                self.tts_service.speak_info("选择超时，请重试")
            return None

        except Exception as e:
            self.choice_mode = False
            self.choice_result = None
            self.choice_max = 0
            logger.error(f"❌ Vosk语音选择异常: {e}")
            if self.tts_service:
                self.tts_service.speak_info("选择过程中出现错误")
            return None

    def _get_sr_choice(self, max_choice: int) -> Optional[str]:
        """使用SpeechRecognition获取语音选择"""
        if not SPEECH_RECOGNITION_AVAILABLE or not self.microphone or not self.recognizer:
            logger.error("❌ SpeechRecognition不可用")
            return None

        try:
            with self.microphone as source:
                logger.info(f"🎤 等待SpeechRecognition数字选择...")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)

            # 识别语音
            text = self.recognizer.recognize_google(audio, language='zh-CN')
            logger.info(f"🎤 SpeechRecognition识别到选择: {text}")

            # 解析数字
            choice = self._parse_number_choice(text, max_choice)
            if choice:
                if self.tts_service:
                    self.tts_service.speak_info(f"您选择了：{choice}")
                return choice
            else:
                if self.tts_service:
                    self.tts_service.speak_info("请说出有效的数字选择")
                return None

        except Exception as e:
            if SPEECH_RECOGNITION_AVAILABLE and "WaitTimeoutError" in str(type(e)):
                logger.warning("🎤 SpeechRecognition选择超时")
                if self.tts_service:
                    self.tts_service.speak_info("选择超时，请重试")
                return None
            elif SPEECH_RECOGNITION_AVAILABLE and "UnknownValueError" in str(type(e)):
                logger.warning("🎤 SpeechRecognition未识别到语音")
                if self.tts_service:
                    self.tts_service.speak_info("没有听清您的选择，请重新说一遍")
                return None
            else:
                logger.error(f"❌ SpeechRecognition语音选择异常: {e}")
                if self.tts_service:
                    self.tts_service.speak_info("选择过程中出现错误")
                return None
    
    def _parse_number_choice(self, text: str, max_choice: int) -> Optional[str]:
        """解析数字选择"""
        text = text.strip().lower()
        
        # 中文数字映射
        chinese_numbers = {
            "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
            "六": "6", "七": "7", "八": "8", "九": "9", "零": "0"
        }
        
        # 查找数字
        for chinese, digit in chinese_numbers.items():
            if chinese in text:
                if 1 <= int(digit) <= max_choice:
                    return digit
        
        # 查找阿拉伯数字
        import re
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                return num
        
        return None
    
    def get_available_commands(self) -> List[str]:
        """获取可用命令列表"""
        commands = []
        for cmd_obj in self.voice_commands.values():
            commands.append(f"{cmd_obj.description}: {', '.join(cmd_obj.patterns[:2])}")
        return commands
    
    def is_available(self) -> bool:
        """检查语音识别是否可用"""
        return self.recognition_method is not None

# 全局语音识别服务实例
_speech_service = None

def get_speech_recognition_service() -> SpeechRecognitionService:
    """获取语音识别服务实例"""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechRecognitionService()
    return _speech_service
