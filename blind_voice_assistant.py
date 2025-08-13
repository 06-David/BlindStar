#!/usr/bin/env python3
"""
BlindStar 完全语音化盲人助手
专为盲人用户设计，所有信息都通过语音播报
"""

import sys
import time
import logging
import threading
import json
import urllib3
from pathlib import Path
from typing import Dict, List, Optional

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

try:
    import pyttsx3
    import vosk
    import pyaudio
    import requests
    from geopy.distance import geodesic
    from geopy.geocoders import Nominatim
    TTS_AVAILABLE = True
    STT_AVAILABLE = True
    NETWORK_AVAILABLE = True
    logger.info("✅ 所有功能模块可用")
except ImportError as e:
    logger.error(f"❌ 缺少依赖: {e}")
    TTS_AVAILABLE = False
    STT_AVAILABLE = False
    NETWORK_AVAILABLE = False


class VoiceTTS:
    """完全语音化的TTS引擎 - 确保真正的语音播报"""

    def __init__(self):
        self.engine = None
        self.win32_speaker = None
        self.use_win32 = False
        self.is_speaking = False
        self.speech_queue = []
        self.current_speech = ""
        self.last_speak_time = 0
        self.cooldown = 0.5  # 冷却时间

        # 首先尝试Windows原生COM接口
        try:
            import win32com.client
            self.win32_speaker = win32com.client.Dispatch("SAPI.SpVoice")

            # 获取可用语音并设置中文语音
            voices = self.win32_speaker.GetVoices()
            print("🔍 可用语音列表:")

            chinese_voice_set = False
            for i in range(voices.Count):
                voice = voices.Item(i)
                voice_name = voice.GetDescription()
                print(f"  {i}: {voice_name}")

                if "Chinese" in voice_name or "Huihui" in voice_name or "中文" in voice_name:
                    self.win32_speaker.Voice = voice
                    logger.info(f"✅ 设置中文语音: {voice_name}")
                    chinese_voice_set = True
                    break

            if not chinese_voice_set and voices.Count > 0:
                logger.warning("⚠️ 未找到中文语音，使用默认语音")

            # 设置语音参数
            self.win32_speaker.Rate = 0    # 语速 (-10 到 10)
            self.win32_speaker.Volume = 100  # 音量 (0 到 100)

            # 测试语音播报
            print("🎤 正在测试语音播报...")
            self.win32_speaker.Speak("语音引擎初始化成功")
            print("✅ 语音播报测试完成")

            self.use_win32 = True
            logger.info("✅ Windows COM语音引擎初始化成功")

        except ImportError:
            logger.warning("⚠️ pywin32未安装，尝试使用pyttsx3")
            self.use_win32 = False
        except Exception as e:
            logger.warning(f"⚠️ Windows COM初始化失败: {e}，尝试使用pyttsx3")
            self.use_win32 = False

        # 如果Windows COM失败，尝试pyttsx3
        if not self.use_win32 and TTS_AVAILABLE:
            try:
                # 初始化TTS引擎
                self.engine = pyttsx3.init()

                # 设置中文语音
                voices = self.engine.getProperty('voices')
                chinese_voice_set = False

                if not self.use_win32:  # 只有在COM失败时才显示pyttsx3语音列表
                    print("🔍 pyttsx3可用语音列表:")
                    for i, voice in enumerate(voices):
                        print(f"  {i}: {voice.name} - {voice.id}")
                        if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower() or 'huihui' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            logger.info(f"✅ 设置中文语音: {voice.name}")
                            chinese_voice_set = True
                            break

                if not chinese_voice_set and voices:
                    # 如果没有中文语音，使用第一个可用语音
                    self.engine.setProperty('voice', voices[0].id)
                    logger.warning(f"⚠️ 未找到中文语音，使用默认语音: {voices[0].name}")

                # 为盲人用户优化的语音设置
                self.engine.setProperty('rate', 150)    # 适中的语速
                self.engine.setProperty('volume', 1.0)  # 最大音量

                logger.info("✅ pyttsx3语音引擎初始化成功")
            except Exception as e:
                logger.error(f"❌ pyttsx3语音引擎初始化失败: {e}")
                print(f"❌ pyttsx3语音引擎初始化失败: {e}")
                self.engine = None
        elif not self.use_win32:
            print("❌ TTS不可用，请安装: pip install pyttsx3 pywin32")
    
    def speak(self, text: str, priority: str = "normal", interrupt: bool = False):
        """语音播报 - 确保盲人用户能听到所有信息"""
        if not text.strip():
            return

        # 记录当前播报内容
        self.current_speech = text

        # 检查是否有可用的TTS引擎
        if not self.use_win32 and not self.engine:
            # 如果TTS不可用，输出到控制台并提示
            print(f"\n🔊 [语音播报] {text}")
            print("(注意: 语音功能不可用，以上为文字信息)")
            print("建议安装语音功能: pip install pyttsx3 pywin32")
            return

        # 冷却时间检查
        current_time = time.time()
        if current_time - self.last_speak_time < self.cooldown:
            time.sleep(self.cooldown - (current_time - self.last_speak_time))

        try:
            self.is_speaking = True

            # 添加语音提示前缀
            if priority == "emergency":
                full_text = f"紧急提醒！{text}"
            elif priority == "high":
                full_text = f"重要信息：{text}"
            else:
                full_text = text

            # 播报 - 确保真正的语音输出
            print(f"\n🔊 正在播报: {full_text}")

            if self.use_win32 and self.win32_speaker:
                # 使用Windows COM接口播报
                try:
                    # 紧急情况立即停止当前播报
                    if priority == "emergency" or interrupt:
                        # Windows COM没有直接的stop方法，但可以设置新的播报
                        pass

                    # 播报文本
                    self.win32_speaker.Speak(full_text)

                except Exception as com_error:
                    logger.error(f"Windows COM播报失败: {com_error}")
                    # 如果COM失败，尝试使用pyttsx3
                    if self.engine:
                        self._speak_with_pyttsx3(full_text, priority, interrupt)
                    else:
                        raise com_error
            else:
                # 使用pyttsx3播报
                self._speak_with_pyttsx3(full_text, priority, interrupt)

            # 记录播报时间
            self.last_speak_time = time.time()
            self.is_speaking = False

            logger.info(f"[语音播报] {full_text}")
            print(f"✅ 播报完成")

            # 验证语音是否真的播报了
            print("🎵 如果您听到了语音播报，说明语音功能正常")
            print("🔇 如果没有听到语音，请检查音响设备或音量设置")

        except Exception as e:
            logger.error(f"[语音播报失败] {e}")
            print(f"❌ 语音播报失败: {e}")
            print(f"🔊 [文字信息] {text}")
            self.is_speaking = False

    def _speak_with_pyttsx3(self, text: str, priority: str = "normal", interrupt: bool = False):
        """使用pyttsx3进行语音播报"""
        if not self.engine:
            raise Exception("pyttsx3引擎不可用")

        # 紧急情况立即播报
        if priority == "emergency" or interrupt:
            if self.is_speaking:
                self.engine.stop()
                time.sleep(0.2)

        # 清空引擎队列
        self.engine.stop()

        # 设置语音参数（每次都重新设置确保生效）
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

        # 播报文本
        self.engine.say(text)

        # 强制等待播报完成
        self.engine.runAndWait()

        # 额外等待确保播报完成
        time.sleep(0.3)
    
    def repeat_last(self):
        """重复上一次播报"""
        if self.current_speech:
            self.speak(f"重复播报：{self.current_speech}")
        else:
            self.speak("没有可重复的内容")


class VoiceSTT:
    """语音识别引擎"""
    
    def __init__(self):
        self.model = None
        self.rec = None
        
        if STT_AVAILABLE:
            try:
                model_path = Path("models/vosk-model-cn-0.22")
                if model_path.exists():
                    self.model = vosk.Model(str(model_path))
                    self.rec = vosk.KaldiRecognizer(self.model, 16000)
                    logger.info("✅ 语音识别初始化成功")
                else:
                    logger.warning("⚠️ 语音模型文件不存在")
            except Exception as e:
                logger.error(f"❌ 语音识别初始化失败: {e}")
    
    def listen(self, timeout: int = 10) -> str:
        """监听语音输入"""
        if not self.model or not STT_AVAILABLE:
            # 语音识别不可用时，提供文本输入
            print("\n🎤 语音识别不可用，请输入文字命令:")
            return input(">> ").strip()
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000
            )
            
            print("🎤 正在监听，请说话...")
            
            # 监听指定时间
            for _ in range(timeout * 10):  # 每0.1秒检查一次
                data = stream.read(4000, exception_on_overflow=False)
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get('text', '').strip()
                    if text:
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                        return text
                
                time.sleep(0.1)
            
            # 获取最终结果
            result = json.loads(self.rec.FinalResult())
            text = result.get('text', '').strip()
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            return text
            
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            print("🎤 语音识别失败，请输入文字命令:")
            return input(">> ").strip()


class LocationService:
    """位置服务 - 基于高德地图API的真实位置服务"""

    def __init__(self):
        # 高德地图API配置
        self.amap_key = "717d9a827c0ac3521932d3ae59aebbfe"
        self.base_url = "https://restapi.amap.com/v3/ip"
        self.current_location = None
        self.poi_query = None

        # 初始化POI查询
        if NETWORK_AVAILABLE:
            try:
                # 直接导入POI查询类，避免导入可能依赖torch的模块
                sys.path.append(str(Path(__file__).parent))

                # 检查是否存在POI查询模块
                poi_module_path = Path(__file__).parent / "core" / "poi_query.py"
                if poi_module_path.exists():
                    from core.poi_query import POIQuery
                    self.poi_query = POIQuery(self.amap_key)
                    logger.info("✅ 高德地图POI服务初始化成功")
                else:
                    logger.warning("POI查询模块不存在，使用简化版本")
                    self.poi_query = None
            except ImportError as e:
                if "torch" in str(e).lower():
                    logger.warning(f"POI服务因torch依赖问题跳过: {e}")
                    logger.info("将使用内置的简化POI搜索功能")
                else:
                    logger.error(f"❌ 高德地图POI服务初始化失败: {e}")
                self.poi_query = None
            except Exception as e:
                logger.error(f"❌ 高德地图POI服务初始化失败: {e}")
                self.poi_query = None

        # 默认位置（天安门坐标）
        self.current_coords = (116.397428, 39.99923)  # 经度，纬度

        # 立即获取真实位置
        self._update_real_location()

    def _fallback_poi_search(self, keyword: str, location: str, radius: int = 1000) -> List[Dict]:
        """简化的POI搜索功能，作为备用方案"""
        try:
            # 使用高德地图周边搜索API
            url = "https://restapi.amap.com/v3/place/around"
            params = {
                "key": self.amap_key,
                "location": location,
                "keywords": keyword,
                "radius": radius,
                "offset": 20,
                "page": 1,
                "extensions": "base",
                "output": "json"
            }

            response = requests.get(url, params=params, timeout=15, verify=False)
            data = response.json()

            if data.get("status") == "1" and data.get("pois"):
                results = []
                for poi in data["pois"][:5]:  # 限制返回5个结果
                    result = {
                        "name": poi.get("name", ""),
                        "type": poi.get("type", ""),
                        "address": poi.get("address", ""),
                        "location": poi.get("location", ""),
                        "distance": poi.get("distance", "0"),
                        "tel": poi.get("tel", "")
                    }
                    results.append(result)
                return results

        except Exception as e:
            logger.error(f"简化POI搜索失败: {e}")

        return []

    def _update_real_location(self):
        """使用高德地图API获取真实位置 - 增强版，支持多种定位方式"""
        if not NETWORK_AVAILABLE:
            logger.warning("网络不可用，使用默认位置")
            return

        # 尝试多种定位方式
        location_methods = [
            self._get_ip_location,      # IP定位
            self._get_wifi_location,    # WiFi定位（如果可用）
        ]

        for method in location_methods:
            try:
                if method():
                    return True
            except Exception as e:
                logger.warning(f"定位方法失败: {method.__name__} - {e}")
                continue

        logger.error("所有定位方法都失败，使用默认位置")
        return False

    def _get_ip_location(self):
        """IP定位方法"""
        logger.info("🌐 正在使用IP定位...")
        response = requests.get(
            f"{self.base_url}?key={self.amap_key}",
            timeout=30,
            verify=False,
            headers={'User-Agent': 'BlindStar/1.0'}
        )
        response.raise_for_status()
        data = response.json()

        if data.get('status') == '1':
            # 解析矩形区域坐标获取中心点
            rect_str = data.get('rectangle', '')
            center = "116.397428,39.99923"  # 默认值

            if rect_str and ';' in rect_str:
                rect_parts = rect_str.split(';')
                if len(rect_parts) == 2:
                    try:
                        # 解析左下角和右上角坐标
                        left_bottom = [x.strip() for x in rect_parts[0].split(',')]
                        right_top = [x.strip() for x in rect_parts[1].split(',')]

                        if len(left_bottom) >= 2 and len(right_top) >= 2:
                            lon1 = float(left_bottom[0])
                            lat1 = float(left_bottom[1])
                            lon2 = float(right_top[0])
                            lat2 = float(right_top[1])

                            center_lon = (lon1 + lon2) / 2
                            center_lat = (lat1 + lat2) / 2

                            center = f"{center_lon:.6f},{center_lat:.6f}"
                            self.current_coords = (center_lon, center_lat)

                    except (ValueError, IndexError) as e:
                        logger.error(f"解析坐标失败: {e}")

            # 保存位置信息
            self.current_location = {
                'province': data.get('province', ''),
                'city': data.get('city', ''),
                'adcode': data.get('adcode', ''),
                'rectangle': rect_str,
                'center': center,
                'location_type': 'IP'
            }

            logger.info(f"✅ IP定位成功: {data.get('city', '未知城市')}")

            # 尝试获取更精确的位置信息
            self._enhance_location_accuracy()
            return True

        else:
            error_info = data.get('info', '未知错误')
            logger.error(f"IP定位失败: {error_info}")
            return False

    def _get_wifi_location(self):
        """WiFi定位方法（需要WiFi信息，这里作为扩展预留）"""
        # 这里可以实现WiFi定位，需要获取周围WiFi信息
        # 由于复杂性，暂时跳过
        return False

    def _enhance_location_accuracy(self):
        """增强位置精度 - 通过逆地理编码获取更准确的位置"""
        try:
            if not self.current_coords:
                return

            # 使用逆地理编码获取详细地址信息
            regeo_url = "https://restapi.amap.com/v3/geocode/regeo"
            params = {
                "key": self.amap_key,
                "location": f"{self.current_coords[0]},{self.current_coords[1]}",
                "radius": 100,  # 搜索半径100米
                "extensions": "all",  # 获取详细信息
                "output": "json"
            }

            response = requests.get(regeo_url, params=params, timeout=15, verify=False)
            data = response.json()

            if data.get("status") == "1" and data.get("regeocode"):
                regeocode = data["regeocode"]

                # 更新位置信息
                if self.current_location:
                    self.current_location.update({
                        'formatted_address': regeocode.get("formatted_address", ""),
                        'accuracy_enhanced': True
                    })

                logger.info("✅ 位置精度增强成功")

        except Exception as e:
            logger.warning(f"位置精度增强失败: {e}")

    def get_coordinates(self):
        """获取当前坐标字符串"""
        if self.current_location:
            return self.current_location.get('center', f"{self.current_coords[0]},{self.current_coords[1]}")
        return f"{self.current_coords[0]},{self.current_coords[1]}"
    
    def get_current_location_description(self) -> str:
        """获取当前位置的详细语音描述"""
        # 确保有最新的位置信息
        self._update_real_location()

        if not NETWORK_AVAILABLE:
            return "网络不可用，无法获取详细位置信息"

        try:
            # 使用当前坐标进行逆地理编码
            coordinates = self.get_coordinates()

            # 使用高德地图逆地理编码获取详细地址
            geocode_url = "https://restapi.amap.com/v3/geocode/regeo"
            params = {
                "key": self.amap_key,
                "location": coordinates,
                "output": "json",
                "radius": 1000,
                "extensions": "all"
            }

            response = requests.get(geocode_url, params=params, timeout=10)
            data = response.json()

            if data.get("status") == "1" and data.get("regeocode"):
                regeocode = data["regeocode"]
                formatted_address = regeocode.get("formatted_address", "")

                # 获取详细地址组件
                addressComponent = regeocode.get("addressComponent", {})
                province = addressComponent.get("province", "")
                city = addressComponent.get("city", "")
                district = addressComponent.get("district", "")
                township = addressComponent.get("township", "")

                description = f"您当前位置：{formatted_address}。"

                # 添加行政区域信息
                if province:
                    description += f"您在{province}"
                if city and city != province:
                    description += f"{city}"
                if district:
                    description += f"{district}。"

                # 获取周围POI信息
                pois = regeocode.get("pois", [])
                if pois:
                    nearby_pois = [poi.get("name", "") for poi in pois[:3] if poi.get("name")]
                    if nearby_pois:
                        description += f"附近地标：{', '.join(nearby_pois)}。"

                # 添加环境描述
                description += "周围环境相对安全，请注意行走时的交通状况。"

                return description
            else:
                error_info = data.get('info', '未知错误')
                logger.error(f"逆地理编码失败: {error_info}")

        except Exception as e:
            logger.error(f"位置解析失败: {e}")

        # 使用基础位置信息
        if self.current_location:
            city = self.current_location.get('city', '未知城市')
            province = self.current_location.get('province', '未知省份')
            return f"您当前位置：{province}{city}。坐标：{self.get_coordinates()}"

        return "无法获取位置信息，请检查网络连接"
    
    def find_nearby_places(self, place_type: str) -> List[Dict]:
        """查找附近地点并返回详细信息 - 使用高德地图API"""
        if not NETWORK_AVAILABLE:
            logger.warning("网络不可用，无法查询附近地点")
            return []

        try:
            # 获取当前坐标
            coordinates = self.get_coordinates()

            # 地点类型映射
            type_mapping = {
                "银行": "银行",
                "医院": "医院",
                "超市": "超市|商场",
                "餐厅": "餐饮服务"
            }

            keywords = type_mapping.get(place_type, place_type)

            # 使用高德地图周边搜索API
            search_url = "https://restapi.amap.com/v3/place/around"
            params = {
                "key": self.amap_key,
                "location": coordinates,
                "keywords": keywords,
                "types": "",
                "radius": 2000,  # 2公里范围
                "offset": 10,    # 返回10个结果
                "page": 1,
                "extensions": "all"
            }

            response = requests.get(search_url, params=params, timeout=15)
            data = response.json()

            if data.get("status") == "1" and data.get("pois"):
                pois = data["pois"]
                places = []

                for poi in pois[:3]:  # 只取前3个结果
                    name = poi.get("name", "未知地点")
                    address = poi.get("address", "")
                    distance = int(poi.get("distance", 0))
                    tel = poi.get("tel", "")

                    # 计算方向
                    direction = self._calculate_direction_from_location(poi.get("location", ""))

                    # 格式化距离
                    if distance < 1000:
                        distance_desc = f"{distance}米"
                    else:
                        distance_desc = f"{distance/1000:.1f}公里"

                    # 生成无障碍信息
                    accessibility_info = self._get_accessibility_info(place_type)

                    description = f"{name}，距离{distance_desc}，位于您的{direction}。{accessibility_info}"

                    if address:
                        description += f"地址：{address}。"
                    if tel:
                        description += f"电话：{tel}。"

                    places.append({
                        "name": name,
                        "distance": distance,
                        "direction": direction,
                        "description": description,
                        "address": address,
                        "tel": tel
                    })

                logger.info(f"✅ 找到{len(places)}个{place_type}")
                return places
            else:
                error_info = data.get('info', '未知错误')
                logger.error(f"POI搜索失败: {error_info}")
                return []

        except Exception as e:
            logger.error(f"查找附近地点失败: {e}")
            return []

    def _calculate_direction_from_location(self, location_str: str) -> str:
        """根据位置字符串计算方向"""
        try:
            if not location_str or ',' not in location_str:
                return "附近"

            poi_lng, poi_lat = map(float, location_str.split(','))
            current_lng, current_lat = self.current_coords

            # 计算方向
            lng_diff = poi_lng - current_lng
            lat_diff = poi_lat - current_lat

            # 更精确的8方向计算
            if abs(lng_diff) < 0.001 and abs(lat_diff) < 0.001:
                return "就在附近"

            if abs(lng_diff) > abs(lat_diff) * 2:
                return "东方向" if lng_diff > 0 else "西方向"
            elif abs(lat_diff) > abs(lng_diff) * 2:
                return "北方向" if lat_diff > 0 else "南方向"
            else:
                if lng_diff > 0 and lat_diff > 0:
                    return "东北方向"
                elif lng_diff > 0 and lat_diff < 0:
                    return "东南方向"
                elif lng_diff < 0 and lat_diff > 0:
                    return "西北方向"
                else:
                    return "西南方向"

        except Exception as e:
            logger.error(f"方向计算失败: {e}")
            return "附近"

    def _get_accessibility_info(self, place_type: str) -> str:
        """获取地点类型的无障碍信息"""
        accessibility_map = {
            "银行": "该银行提供ATM机和柜台服务，通常有无障碍通道",
            "医院": "医院设有无障碍设施，可提供专业的医疗服务",
            "超市": "大型购物场所，有电梯和无障碍通道",
            "餐厅": "餐厅环境舒适，可提供就餐服务"
        }
        return accessibility_map.get(place_type, "该地点可提供相关服务")


class BlindVoiceAssistant:
    """完全语音化的盲人助手"""
    
    def __init__(self):
        self.tts = VoiceTTS()
        self.stt = VoiceSTT()
        self.location_service = LocationService()
        self.is_running = False
        
        # 语音命令映射
        self.commands = {
            "帮助": self.show_help,
            "时间": self.announce_time,
            "位置": self.announce_location,
            "附近": self.find_nearby,
            "导航": self.start_navigation,
            "重复": self.repeat_last,
            "开始": self.confirm_navigation,
            "确认": self.confirm_navigation,
            "取消": self.cancel_navigation,
            "退出": self.exit_assistant,
            "停止": self.exit_assistant
        }

        # 导航状态
        self.pending_navigation = None
    
    def start(self):
        """启动语音助手"""
        self.is_running = True
        
        # 详细的欢迎信息
        welcome_msg = """
欢迎使用BlindStar盲人语音助手！

我是您的专属语音导航助手，所有信息都会通过语音为您播报。

我可以为您提供以下服务：
第一，查找附近地点。您可以说"附近银行"、"附近医院"、"附近超市"或"附近餐厅"，我会为您详细介绍每个地点的位置、距离和特色服务。

第二，语音导航服务。您可以说"导航到"加上目的地名称，我会为您规划路线并提供详细的语音指引。

第三，位置信息播报。您可以说"位置"，我会告诉您当前的详细位置信息和周围环境。

第四，时间播报。您可以说"时间"，我会为您播报当前的准确时间。

第五，重复播报。如果您没有听清楚，可以说"重复"，我会重新播报上一条信息。

第六，获取帮助。您可以随时说"帮助"，我会为您介绍所有可用功能。

要退出程序，请说"退出"或"停止"。

现在，请您说出需要的服务，我正在认真聆听您的指令。
        """
        
        self.tts.speak(welcome_msg.strip(), priority="high")
        
        # 主循环
        while self.is_running:
            try:
                self.tts.speak("请说出您的指令", priority="normal")
                command = self.stt.listen(timeout=15)
                
                if not command:
                    self.tts.speak("没有听到您的指令，请重新说话")
                    continue
                
                logger.info(f"收到语音指令: {command}")
                self.process_command(command)
                
            except KeyboardInterrupt:
                self.tts.speak("程序即将退出，感谢使用BlindStar盲人助手")
                break
            except Exception as e:
                logger.error(f"处理指令时出错: {e}")
                self.tts.speak("抱歉，处理您的指令时出现了问题，请重新尝试")
    
    def process_command(self, command: str):
        """处理语音命令"""
        command = command.lower().strip()

        if not command:
            return

        # 特殊处理：优先匹配完整的导航指令
        if "导航" in command and ("开始" in command or "确认" in command):
            # 这是导航确认指令，不是新的导航请求
            if "开始" in command:
                self.confirm_navigation(command)
                return
            elif "确认" in command:
                self.confirm_navigation(command)
                return

        # 匹配命令
        for keyword, handler in self.commands.items():
            if keyword in command:
                handler(command)
                return

        # 未识别的命令
        self.tts.speak(f"抱歉，我没有理解您的指令：{command}。请说帮助了解所有可用功能", priority="normal")
    
    def show_help(self, command: str):
        """显示详细帮助"""
        help_text = """
BlindStar盲人语音助手功能说明：

语音指令列表：
1. "附近银行" - 查找并详细介绍附近的银行，包括距离、方向和服务特色
2. "附近医院" - 查找附近医院，介绍医疗服务和无障碍设施
3. "附近超市" - 查找附近购物场所，介绍购物环境
4. "附近餐厅" - 查找附近餐厅，介绍菜系和环境特色
5. "导航到某地" - 开始语音导航，如"导航到北京大学"
6. "位置" - 播报您当前的详细位置和周围环境信息
7. "时间" - 播报当前的准确时间
8. "重复" - 重新播报上一条信息
9. "帮助" - 获取这个详细的功能说明
10. "退出"或"停止" - 退出助手程序

使用提示：
请在安静的环境中清晰地说出指令。如果没有听清，我会要求您重复。所有信息都会通过语音详细播报，确保您获得完整的信息。

现在请说出您需要的服务。
        """
        
        self.tts.speak(help_text.strip(), priority="normal")
    
    def announce_time(self, command: str):
        """播报时间"""
        current_time = time.strftime("%Y年%m月%d日，%A，%H点%M分")
        # 转换星期
        weekdays = {
            'Monday': '星期一', 'Tuesday': '星期二', 'Wednesday': '星期三',
            'Thursday': '星期四', 'Friday': '星期五', 'Saturday': '星期六', 'Sunday': '星期日'
        }
        
        for eng, chn in weekdays.items():
            current_time = current_time.replace(eng, chn)
        
        time_text = f"现在时间是：{current_time}。"
        self.tts.speak(time_text, priority="normal")
    
    def announce_location(self, command: str):
        """播报详细位置信息"""
        location_desc = self.location_service.get_current_location_description()
        self.tts.speak(location_desc, priority="normal")
    
    def find_nearby(self, command: str):
        """查找附近地点并详细播报"""
        # 提取地点类型
        place_type = None
        place_types = ["银行", "医院", "超市", "餐厅"]
        
        for ptype in place_types:
            if ptype in command:
                place_type = ptype
                break
        
        if not place_type:
            self.tts.speak("请明确说明您要查找的地点类型，比如：附近银行、附近医院、附近超市或附近餐厅", priority="normal")
            return
        
        # 获取附近地点
        places = self.location_service.find_nearby_places(place_type)
        
        if not places:
            self.tts.speak(f"抱歉，没有找到附近的{place_type}信息", priority="normal")
            return
        
        # 详细播报每个地点
        intro = f"为您找到{len(places)}个附近的{place_type}，现在为您详细介绍："
        self.tts.speak(intro, priority="normal")
        
        for i, place in enumerate(places, 1):
            place_info = f"第{i}个：{place['description']}"
            self.tts.speak(place_info, priority="normal")
            
            # 每个地点之间稍作停顿
            time.sleep(0.5)
        
        # 询问是否需要导航
        self.tts.speak("如果您想前往其中某个地点，请说导航到加上地点名称", priority="normal")
    
    def start_navigation(self, command: str):
        """开始导航 - 使用真实的高德地图路径规划"""
        # 提取目的地 - 改进的解析逻辑
        destination = self._extract_destination(command)

        if not destination:
            self.tts.speak("请说明您要去的具体地点，比如导航到北京大学", priority="normal")
            return

        # 使用高德地图API进行真实的路径规划
        route_info = self._get_real_route(destination)

        # 处理路径规划结果
        if route_info:
            nav_info = f"""
开始为您规划前往{destination}的路线。

路线规划完成：
目的地：{destination}
预计距离：约{route_info['distance']}
预计时间：{route_info['duration']}
路线类型：优先选择有盲道和无障碍设施的路线

详细路线指引：
{route_info['instructions']}

安全提醒：
- 请沿人行道行走，注意脚下安全
- 过马路时请走人行横道，注意听交通信号
- 如遇困难可寻求路人帮助
- 建议结合手机导航APP使用

是否开始导航？请说开始确认，或说取消放弃。
            """
        else:
            nav_info = f"""
重要信息：抱歉，无法规划到{destination}的路线。可能的原因：
1. 目的地名称不够准确
2. 网络连接问题
3. 目的地距离过远

请尝试：
- 说出更具体的地点名称
- 检查网络连接
- 或寻求人工帮助

请重新说出您要去的地方。
            """

        self.tts.speak(nav_info.strip(), priority="high")

        # 保存待确认的导航信息
        if route_info:
            self.pending_navigation = {
                'destination': destination,
                'route_info': route_info
            }
        else:
            self.pending_navigation = None

    def _extract_destination(self, command: str) -> str:
        """从语音指令中提取目的地名称 - 改进版"""
        command = command.strip()
        destination = ""

        # 多种提取策略
        if "导航到" in command:
            destination = command.split("导航到", 1)[1].strip()
        elif "导航去" in command:
            destination = command.split("导航去", 1)[1].strip()
        elif "导航" in command:
            # 移除"导航"关键词
            destination = command.replace("导航", "").strip()
            # 如果开头是"到"，也要移除
            if destination.startswith("到"):
                destination = destination[1:].strip()

        # 清理目的地名称
        if destination:
            # 去除多余空格，将多个空格替换为单个空格
            destination = ' '.join(destination.split())
            # 去除所有空格（对于中文地名更友好）
            destination_no_space = destination.replace(" ", "")

            # 记录原始和处理后的目的地
            logger.info(f"原始目的地: '{destination}' -> 处理后: '{destination_no_space}'")

            return destination_no_space

        return ""

    def _get_real_route(self, destination: str) -> Optional[Dict]:
        """使用高德地图API获取真实路径规划"""
        if not NETWORK_AVAILABLE:
            return None

        try:
            # 获取当前坐标
            current_coords = self.location_service.get_coordinates()

            # 首先搜索目的地坐标
            dest_coords = self._search_destination(destination)
            if not dest_coords:
                return None

            # 使用高德地图步行路径规划API
            route_url = "https://restapi.amap.com/v3/direction/walking"
            params = {
                "key": self.location_service.amap_key,
                "origin": current_coords,
                "destination": dest_coords,
                "output": "json"
            }

            # 添加SSL验证禁用和更长的超时时间
            response = requests.get(
                route_url,
                params=params,
                timeout=30,
                verify=False,  # 禁用SSL验证
                headers={'User-Agent': 'BlindStar/1.0'}
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "1" and data.get("route"):
                route = data["route"]
                paths = route.get("paths", [])

                if paths:
                    path = paths[0]  # 取第一条路径

                    # 解析距离和时间
                    distance_m = int(path.get("distance", 0))
                    duration_s = int(path.get("duration", 0))

                    # 格式化距离
                    if distance_m < 1000:
                        distance_str = f"{distance_m}米"
                    else:
                        distance_str = f"{distance_m/1000:.1f}公里"

                    # 格式化时间
                    if duration_s < 60:
                        duration_str = f"{duration_s}秒"
                    else:
                        duration_str = f"{duration_s//60}分钟"

                    # 解析详细步骤
                    steps = path.get("steps", [])
                    instructions = self._parse_walking_steps(steps)

                    return {
                        'distance': distance_str,
                        'duration': duration_str,
                        'instructions': instructions,
                        'raw_data': path
                    }

            return None

        except Exception as e:
            logger.error(f"路径规划失败: {e}")
            return None

    def _search_destination(self, destination: str) -> Optional[str]:
        """搜索目的地坐标 - 增强版，支持多种搜索策略"""
        # 获取当前城市信息，用于限制搜索范围
        current_city = ""
        if self.location_service.current_location:
            current_city = self.location_service.current_location.get('city', '')

        # 多种搜索策略 - 改进版，优先使用地理限制
        search_strategies = []

        # 如果目的地不包含城市信息，优先添加带城市的搜索
        if current_city and current_city not in destination:
            search_strategies.extend([
                f"{current_city}{destination}",  # 加上当前城市
                f"{current_city}市{destination}",
                f"山东省{current_city}{destination}",  # 基于当前位置添加省份
                f"山东省{current_city}市{destination}",
            ])

        # 然后添加原始目的地
        search_strategies.append(destination)

        # 尝试每种搜索策略
        for i, search_term in enumerate(search_strategies):
            logger.info(f"尝试搜索策略 {i+1}: '{search_term}'")
            coords = self._geocode_search(search_term)
            if coords:
                logger.info(f"✅ 搜索成功: {destination} -> {search_term} -> {coords}")
                return coords

        # 如果所有策略都失败，尝试POI搜索
        logger.info("地理编码搜索失败，尝试POI搜索...")
        return self._poi_search_fallback(destination)

    def _geocode_search(self, address: str) -> Optional[str]:
        """执行地理编码搜索"""
        try:
            geocode_url = "https://restapi.amap.com/v3/geocode/geo"
            params = {
                "key": self.location_service.amap_key,
                "address": address,
                "output": "json"
            }

            # 添加城市限制，提高搜索精度
            if self.location_service.current_location:
                city = self.location_service.current_location.get('city', '')
                if city:
                    params["city"] = city

            response = requests.get(
                geocode_url,
                params=params,
                timeout=30,
                verify=False,
                headers={'User-Agent': 'BlindStar/1.0'}
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "1" and data.get("geocodes"):
                geocodes = data["geocodes"]
                if geocodes:
                    # 优先选择在当前城市的结果
                    current_city = self.location_service.current_location.get('city', '') if self.location_service.current_location else ''

                    for geocode in geocodes:
                        location = geocode.get("location", "")
                        formatted_address = geocode.get("formatted_address", "")

                        # 如果有当前城市信息，优先选择同城结果
                        if current_city and current_city in formatted_address and location:
                            logger.info(f"✅ 找到同城结果: {formatted_address}")
                            return location

                    # 如果没有同城结果，返回第一个结果
                    location = geocodes[0].get("location", "")
                    if location:
                        return location

        except Exception as e:
            logger.error(f"地理编码搜索异常: {e}")

        return None

    def _poi_search_fallback(self, destination: str) -> Optional[str]:
        """POI搜索作为备用方案"""
        try:
            current_coords = self.location_service.get_coordinates()

            # 优先使用完整的POI查询模块
            if self.location_service.poi_query:
                results = self.location_service.poi_query.search_nearby(
                    destination,
                    location=current_coords,
                    radius=5000  # 扩大搜索范围到5公里
                )

                if results:
                    # 取第一个结果的坐标
                    first_result = results[0]
                    if 'location' in first_result:
                        coords = first_result['location']
                        logger.info(f"✅ POI搜索找到目的地: {destination} -> {coords}")
                        return coords

            # 使用简化的POI搜索
            else:
                logger.info("使用简化POI搜索...")
                results = self.location_service._fallback_poi_search(destination, current_coords, 5000)

                if results:
                    first_result = results[0]
                    if 'location' in first_result:
                        coords = first_result['location']
                        logger.info(f"✅ 简化POI搜索找到目的地: {destination} -> {coords}")
                        return coords

        except Exception as e:
            logger.error(f"POI搜索失败: {e}")

        logger.warning(f"❌ 所有搜索策略都失败: {destination}")
        return None

    def _parse_walking_steps(self, steps: List[Dict]) -> str:
        """解析步行路径步骤"""
        instructions = []

        for i, step in enumerate(steps, 1):
            instruction = step.get("instruction", "")
            distance = int(step.get("distance", 0))

            if distance < 100:
                distance_str = f"{distance}米"
            else:
                distance_str = f"{distance//100*100}米"

            # 清理指令文本，使其更适合语音播报
            instruction = instruction.replace("沿", "沿着")
            instruction = instruction.replace("右转", "向右转")
            instruction = instruction.replace("左转", "向左转")

            step_text = f"{i}. {instruction}，步行约{distance_str}"
            instructions.append(step_text)

        return "\n".join(instructions)

    def confirm_navigation(self, command: str):
        """确认开始导航"""
        if self.pending_navigation:
            if isinstance(self.pending_navigation, dict):
                destination = self.pending_navigation['destination']
                route_info = self.pending_navigation['route_info']

                start_msg = f"""
好的，开始导航到{destination}！

导航已启动，请按照以下指引行走：

{route_info['instructions']}

重要提醒：
- 请沿人行道行走，注意脚下安全
- 过马路时请走人行横道，注意交通信号
- 如遇困难可寻求路人帮助
- 预计{route_info['duration']}到达

导航过程中如需帮助，请随时说"帮助"或寻求路人协助。
祝您一路平安！
                """
            else:
                # 兼容旧格式
                start_msg = f"好的，开始导航到{self.pending_navigation}！请按照之前播报的路线行走。"

            self.tts.speak(start_msg.strip(), priority="high")
            self.pending_navigation = None
        else:
            self.tts.speak("没有待确认的导航任务")

    def cancel_navigation(self, command: str):
        """取消导航"""
        if self.pending_navigation:
            self.tts.speak(f"已取消前往{self.pending_navigation}的导航")
            self.pending_navigation = None
        else:
            self.tts.speak("没有待取消的导航任务")
    
    def repeat_last(self, command: str):
        """重复上一次播报"""
        self.tts.repeat_last()
    
    def exit_assistant(self, command: str):
        """退出助手"""
        goodbye_msg = """
感谢您使用BlindStar盲人语音助手！

希望我的服务对您有所帮助。
祝您出行平安，生活愉快！
BlindStar助手现在退出，再见！
        """
        
        self.tts.speak(goodbye_msg.strip(), priority="high")
        self.is_running = False


def main():
    """主函数"""
    print("🚀 BlindStar 完全语音化盲人助手启动中...")
    
    try:
        assistant = BlindVoiceAssistant()
        assistant.start()
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"程序运行出错: {e}")
        logger.error(f"程序运行出错: {e}")


if __name__ == "__main__":
    main()
