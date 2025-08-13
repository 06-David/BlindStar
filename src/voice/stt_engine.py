import vosk
import json
import threading
import queue
import pyaudio
import re
from typing import Optional, Callable, Dict, List
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

    # 导航相关的语音命令处理
    def extract_navigation_command(self, text: str) -> Optional[Dict[str, str]]:
        """
        从语音文本中提取导航命令

        Args:
            text: 识别到的语音文本

        Returns:
            包含命令类型和参数的字典，如果不是导航命令则返回None
        """
        if not text:
            return None

        text = text.strip().lower()

        # 导航到某地的命令
        nav_patterns = [
            r'导航\s*[到去]\s*(.+?)(?:请|吗|呢|吧|$)',
            r'去\s*(.+?)(?:请|吗|呢|吧|$)',
            r'带我\s*[到去]\s*(.+?)(?:请|吗|呢|吧|$)',
            r'我要\s*去\s*(.+?)(?:请|吗|呢|吧|$)'
        ]

        for pattern in nav_patterns:
            match = re.search(pattern, text)
            if match:
                destination = match.group(1).strip()
                # 清理目的地名称
                destination = re.sub(r'^(附近\s*的?\s*|去\s*)+', '', destination).strip()
                if destination:
                    return {
                        'type': 'set_destination',
                        'destination': destination,
                        'original_text': text
                    }

        # 其他导航命令
        navigation_commands = {
            '开始导航': 'start_navigation',
            '结束导航': 'stop_navigation',
            '停止导航': 'stop_navigation',
            '暂停导航': 'pause_navigation',
            '继续导航': 'resume_navigation',
            '重新规划': 'recalculate_route',
            '当前位置': 'announce_location',
            '我在哪里': 'announce_location',
            '剩余距离': 'announce_remaining',
            '还有多远': 'announce_remaining',
            '下一个路口': 'announce_next_turn',
            '导航状态': 'announce_status',
            '导航帮助': 'navigation_help'
        }

        for cmd_text, cmd_type in navigation_commands.items():
            if cmd_text in text:
                return {
                    'type': cmd_type,
                    'original_text': text
                }

        return None

    def extract_poi_query(self, text: str) -> Optional[Dict[str, str]]:
        """
        从语音文本中提取POI查询命令

        Args:
            text: 识别到的语音文本

        Returns:
            包含查询类型和关键词的字典，如果不是POI查询则返回None
        """
        if not text:
            return None

        text = text.strip().lower()

        # POI查询模式
        poi_patterns = [
            r'附近\s*有\s*什么\s*(.+?)(?:吗|呢|$)',
            r'附近\s*的\s*(.+?)(?:在哪里|有哪些|吗|呢|$)',
            r'查找\s*(.+?)(?:吗|呢|$)',
            r'找\s*(.+?)(?:吗|呢|$)',
            r'搜索\s*(.+?)(?:吗|呢|$)',
            r'(.+?)\s*在哪里(?:吗|呢|$)'
        ]

        for pattern in poi_patterns:
            match = re.search(pattern, text)
            if match:
                keyword = match.group(1).strip()
                if keyword:
                    return {
                        'type': 'poi_search',
                        'keyword': keyword,
                        'original_text': text
                    }

        return None

    def is_system_command(self, text: str) -> Optional[str]:
        """
        检查是否为系统控制命令

        Args:
            text: 识别到的语音文本

        Returns:
            命令类型，如果不是系统命令则返回None
        """
        if not text:
            return None

        text = text.strip().lower()

        system_commands = {
            '开始检测': 'start_detection',
            '停止检测': 'stop_detection',
            '开始录制': 'start_recording',
            '停止录制': 'stop_recording',
            '退出系统': 'exit_system',
            '帮助': 'help',
            '系统帮助': 'help',
            '暂停': 'pause',
            '继续': 'resume'
        }

        for cmd_text, cmd_type in system_commands.items():
            if cmd_text in text:
                return cmd_type

        return None

    def parse_voice_command(self, text: str) -> Dict[str, any]:
        """
        解析语音命令的统一接口

        Args:
            text: 识别到的语音文本

        Returns:
            解析结果字典
        """
        result = {
            'original_text': text,
            'command_type': 'unknown',
            'parsed_command': None
        }

        if not text:
            return result

        # 检查导航命令
        nav_cmd = self.extract_navigation_command(text)
        if nav_cmd:
            result['command_type'] = 'navigation'
            result['parsed_command'] = nav_cmd
            return result

        # 检查POI查询
        poi_cmd = self.extract_poi_query(text)
        if poi_cmd:
            result['command_type'] = 'poi'
            result['parsed_command'] = poi_cmd
            return result

        # 检查系统命令
        sys_cmd = self.is_system_command(text)
        if sys_cmd:
            result['command_type'] = 'system'
            result['parsed_command'] = {'type': sys_cmd}
            return result

        return result

    def extract_accessibility_command(self, text: str) -> Optional[Dict[str, str]]:
        """
        提取无障碍相关的语音命令 - 专为盲人用户设计

        Args:
            text: 识别到的语音文本

        Returns:
            包含命令类型和参数的字典
        """
        if not text:
            return None

        text = text.strip().lower()

        # 无障碍友好的命令模式
        accessibility_patterns = [
            # 重复指令
            (r'(再说一遍|重复|没听清|重复一遍)', 'repeat_last'),

            # 详细信息请求
            (r'(详细说明|具体信息|更多信息)', 'detailed_info'),

            # 周围环境查询
            (r'(周围有什么|附近环境|环境描述)', 'describe_environment'),

            # 安全查询
            (r'(安全吗|有危险吗|可以走吗)', 'safety_check'),

            # 方向确认
            (r'(哪个方向|往哪走|怎么走)', 'direction_help'),

            # 距离查询
            (r'(还有多远|距离多少|多长时间)', 'distance_query'),

            # 紧急停止
            (r'(停|停止|等等|暂停)', 'emergency_stop'),

            # 帮助请求
            (r'(帮助|救命|不知道怎么办)', 'help_request'),

            # 确认命令
            (r'(是的|对的|确认|好的)', 'confirm'),
            (r'(不是|不对|取消|不要)', 'cancel'),

            # 速度调整
            (r'(说慢点|慢一点|太快了)', 'slow_down'),
            (r'(说快点|快一点|太慢了)', 'speed_up'),
        ]

        for pattern, command_type in accessibility_patterns:
            if re.search(pattern, text):
                return {
                    'type': 'accessibility',
                    'command': command_type,
                    'original_text': text
                }

        return None

    def parse_enhanced_voice_command(self, text: str) -> Dict[str, Any]:
        """
        增强的语音命令解析 - 为盲人用户优化

        Args:
            text: 识别到的语音文本

        Returns:
            解析结果字典
        """
        result = {
            'original_text': text,
            'command_type': 'unknown',
            'parsed_command': None,
            'confidence': 0.0
        }

        if not text:
            return result

        # 检查无障碍命令
        accessibility_cmd = self.extract_accessibility_command(text)
        if accessibility_cmd:
            result['command_type'] = 'accessibility'
            result['parsed_command'] = accessibility_cmd
            result['confidence'] = 0.9
            return result

        # 检查原有的命令类型
        original_result = self.parse_voice_command(text)
        if original_result['command_type'] != 'unknown':
            result = original_result
            result['confidence'] = 0.8
            return result

        # 模糊匹配 - 为语音识别错误提供容错
        fuzzy_result = self._fuzzy_command_match(text)
        if fuzzy_result:
            result.update(fuzzy_result)

        return result

    def _fuzzy_command_match(self, text: str) -> Optional[Dict[str, Any]]:
        """
        模糊命令匹配 - 处理语音识别错误

        Args:
            text: 可能有错误的语音文本

        Returns:
            匹配结果
        """
        # 常见的语音识别错误和纠正
        corrections = {
            '导行': '导航',
            '带我': '带我去',
            '找找': '查找',
            '附件': '附近',
            '银行': '银行',
            '餐厅': '餐厅',
            '超市': '超市',
            '医院': '医院',
            '地铁': '地铁站',
            '公交': '公交站',
        }

        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)

        # 如果纠正后的文本不同，重新解析
        if corrected_text != text:
            corrected_result = self.parse_voice_command(corrected_text)
            if corrected_result['command_type'] != 'unknown':
                corrected_result['confidence'] = 0.6  # 降低置信度因为是纠正后的
                return corrected_result

        return None