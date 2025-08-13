import logging
import time
import requests
import math
from typing import Dict, List, Tuple, Optional
from .poi_query import POIQuery
from .tts_engine import TTSEngine
import re  # 新增正则表达式库

logger = logging.getLogger(__name__)


def calculate_bearing(origin: Tuple[float, float], destination: Tuple[float, float]) -> float:
    """
    计算从起点到终点的方位角（度数）

    Args:
        origin: 起点坐标 (经度, 纬度)
        destination: 终点坐标 (经度, 纬度)

    Returns:
        方位角（0-360度，0度为正北方向）
    """
    lon1, lat1 = math.radians(origin[0]), math.radians(origin[1])
    lon2, lat2 = math.radians(destination[0]), math.radians(destination[1])

    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def calculate_distance(origin: Tuple[float, float], destination: Tuple[float, float]) -> float:
    """
    计算两点间的直线距离（米）
    使用Haversine公式

    Args:
        origin: 起点坐标 (经度, 纬度)
        destination: 终点坐标 (经度, 纬度)

    Returns:
        距离（米）
    """
    R = 6371000  # 地球半径（米）

    lon1, lat1 = math.radians(origin[0]), math.radians(origin[1])
    lon2, lat2 = math.radians(destination[0]), math.radians(destination[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    distance = R * c
    return distance


def bearing_to_direction(bearing: float) -> str:
    """
    将方位角转换为方向描述（8个基本方向）

    Args:
        bearing: 方位角（0-360度）

    Returns:
        方向描述（如"东北方向"）
    """
    directions = [
        "正北", "东北", "正东", "东南",
        "正南", "西南", "正西", "西北"
    ]

    # 每个方向占45度
    index = int((bearing + 22.5) / 45) % 8
    return directions[index]


class AmapNavigation:
    """高德地图导航服务封装"""

    def __init__(self, api_key: str, api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://restapi.amap.com/v3"
        self.current_route = None
        self.is_navigating = False
        self.current_step_index = 0
        self.last_update_time = 0
        self.update_interval = 5  # 导航状态更新间隔(秒)
        self.poi = None  # POI查询模块

    def set_poi_module(self, poi: POIQuery):
        """设置POI查询模块"""
        self.poi = poi

    def plan_route(self, origin: Tuple[float, float],
                   destination: Tuple[float, float],
                   mode: str = "walking") -> Optional[Dict]:
        """
        路径规划（安全优先）

        Args:
            origin: 起点坐标 (经度, 纬度)
            destination: 终点坐标 (经度, 纬度)
            mode: 出行模式 (walking, transit)

        Returns:
            路径规划结果字典，包含路径点、距离、时间等信息
        """
        endpoint = "/direction/walking" if mode == "walking" else "/direction/transit"
        url = f"{self.base_url}{endpoint}?key={self.api_key}"

        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "strategy": "0" if mode == "transit" else "safe"
        }

        if mode == "transit":
            params["nightflag"] = "1"  # 夜间安全模式

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["status"] == "1":
                route = data["route"]
                self.current_route = self._parse_route(route)
                return self.current_route
            else:
                logger.error(f"路径规划失败: {data.get('info', '未知错误')}")
                return None

        except Exception as e:
            logger.error(f"路径规划请求异常: {e}")
            return None

    def _parse_route(self, route_data: Dict) -> Dict:
        """解析路径规划结果"""
        path = route_data["paths"][0]
        steps = []

        for step in path["steps"]:
            # 清理HTML标签
            instruction = step["instruction"].replace("<b>", "").replace("</b>", "")
            steps.append({
                "distance": float(step["distance"]),
                "duration": float(step["duration"]),
                "action": self._extract_action(instruction),
                "road": self._extract_road_name(instruction),
                "polyline": step["polyline"],
                "instruction": instruction
            })

        return {
            "distance": float(path["distance"]),
            "duration": float(path["duration"]),
            "steps": steps,
            "origin": route_data["origin"],
            "destination": route_data["destination"]
        }

    def _extract_action(self, instruction: str) -> str:
        """从导航指令中提取动作类型"""
        if "左转" in instruction:
            return "左转"
        elif "右转" in instruction:
            return "右转"
        elif "掉头" in instruction:
            return "掉头"
        elif "到达" in instruction:
            return "到达目的地"
        return "向前"

    def _extract_road_name(self, instruction: str) -> str:
        """从导航指令中提取道路名称"""
        # 简化的道路名称提取逻辑
        if "进入" in instruction:
            start = instruction.find("进入") + 2
            end = instruction.find("，", start)
            return instruction[start:end] if end != -1 else instruction[start:]
        return instruction.split("，")[0]

    def start_navigation(self, route: Dict):
        """开始实时导航"""
        if not route or "steps" not in route:
            raise ValueError("无效的路径规划结果")

        self.current_route = route
        self.is_navigating = True
        self.current_step_index = 0
        self.last_update_time = time.time()
        logger.info("导航已开始")

    def update_navigation(self, current_location: Tuple[float, float]) -> Optional[Dict]:
        """
        更新导航状态

        Args:
            current_location: 当前位置坐标 (经度, 纬度)

        Returns:
            当前导航指令信息
        """
        if not self.is_navigating or not self.current_route:
            return None

        # 控制更新频率
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return None

        self.last_update_time = current_time
        current_step = self.current_route["steps"][self.current_step_index]

        # 简化距离计算（实际应使用Haversine公式）
        # 这里仅返回当前步骤信息
        return {
            "action": current_step["action"],
            "distance": current_step["distance"],
            "road": current_step["road"],
            "instruction": current_step["instruction"],
            "step_index": self.current_step_index,
            "total_steps": len(self.current_route["steps"])
        }

    def stop_navigation(self):
        """停止导航"""
        self.is_navigating = False
        self.current_route = None
        logger.info("导航已停止")

    def next_step(self):
        """前进到下一步"""
        if self.is_navigating and self.current_route:
            if self.current_step_index < len(self.current_route["steps"]) - 1:
                self.current_step_index += 1
                return True
        return False


class NavigationEventHandler:
    """导航事件处理器"""

    def __init__(self, tts_engine: TTSEngine):
        self.tts = tts_engine

    def on_route_calculated(self, route: Dict):
        """路径规划完成事件"""
        distance_km = route["distance"] / 1000
        duration_min = route["duration"] / 60
        msg = f"路径规划完成，总距离{distance_km:.1f}公里，预计耗时{duration_min:.0f}分钟"
        self.tts.speak(msg)
        return msg

    def on_navigation_point(self, point_info: Dict):
        """导航关键点事件"""
        distance = point_info["distance"]
        action = point_info["action"]
        road = point_info.get("road", "")

        if distance < 1000:
            distance_desc = f"{int(distance)}米"
        else:
            distance_desc = f"{distance / 1000:.1f}公里"

        msg = f"前方{distance_desc}{action}"
        if road:
            msg += f"，进入{road}"

        self.tts.speak(msg)
        return msg

    def on_arrive_destination(self):
        """到达目的地事件"""
        msg = "已到达目的地"
        self.tts.speak(msg)
        return msg

    def on_navigation_start(self):
        """导航开始事件"""
        msg = "导航已开始，请按照提示行进"
        self.tts.speak(msg)
        return msg


class VoiceNavigationHandler:
    """语音导航命令处理器（增强版）"""

    NAVIGATION_COMMANDS = {
        "开始导航": "start_navigation",
        "结束导航": "stop_navigation",
        "暂停导航": "pause_navigation",
        "继续导航": "resume_navigation",
        "重新规划": "recalculate_route",
        "当前位置": "announce_location",
        "剩余距离": "announce_remaining",
        "下一个路口": "announce_next_turn",
        "导航到": "set_destination",
        "导航去": "set_destination"
    }

    def __init__(self, nav_system: AmapNavigation, tts_engine: TTSEngine):
        self.nav = nav_system
        self.tts = tts_engine
        self.event_handler = NavigationEventHandler(tts_engine)
        self.is_paused = False
        self.destination = None  # 存储目的地
        self.poi = None

    def set_poi_module(self, poi: POIQuery):
        """设置POI查询模块"""
        self.poi = poi

    def handle_command(self, command: str) -> bool:
        """处理导航语音命令（增强容错性）"""
        # 清理命令中的多余空格，提高匹配准确性
        cleaned_command = re.sub(r'\s+', '', command.strip())

        # 检查是否是导航到某地的命令（增强匹配，支持空格）
        # 使用正则表达式匹配，支持空格
        nav_pattern = r'导航\s*[到去]\s*(.+?)(?:请|吗|呢|吧|$)'
        nav_match = re.search(nav_pattern, command)

        if nav_match:
            dest_name = nav_match.group(1).strip()
            # 进一步清理目的地名称
            # 去掉开头的"附近"、"去"等词，支持多次清理
            dest_name = re.sub(r'^(附近\s*的?\s*|去\s*)+', '', dest_name).strip()
            # 去掉中间多余的"去"
            dest_name = re.sub(r'\s+去\s+', ' ', dest_name).strip()

            if dest_name:
                logging.info(f"[导航处理器] 提取到目的地: '{dest_name}'")
                self.set_destination(dest_name)
                return True

        # 检查其他导航命令（同时检查原始命令和清理后的命令）
        for cmd_key, cmd_type in self.NAVIGATION_COMMANDS.items():
            # 清理命令字典中的空格进行匹配
            cleaned_cmd_key = re.sub(r'\s+', '', cmd_key)
            original_match = cmd_key in command
            cleaned_match = cleaned_cmd_key in cleaned_command

            if original_match or cleaned_match:
                return self._handle_nav_command(cmd_type)

        return False

    def _handle_nav_command(self, cmd_type: str) -> bool:
        """处理具体导航命令"""
        if cmd_type == "start_navigation":
            self._start_navigation()
        elif cmd_type == "stop_navigation":
            self._stop_navigation()
        elif cmd_type == "pause_navigation":
            self._pause_navigation()
        elif cmd_type == "resume_navigation":
            self._resume_navigation()
        elif cmd_type == "recalculate_route":
            self._recalculate_route()
        elif cmd_type == "announce_location":
            self._announce_location()
        elif cmd_type == "announce_remaining":
            self._announce_remaining()
        elif cmd_type == "announce_next_turn":
            self._announce_next_turn()
        else:
            logging.warning(f"[导航处理器] 未知命令类型: {cmd_type}")
            return False

        return True

    def set_destination(self, destination_name: str):
        """设置导航目的地（增强版，包含方位信息）"""
        if not self.poi:
            self.tts.speak("POI查询功能未启用，无法设置目的地")
            return

        try:
            # 搜索目的地
            results = self.poi.search_nearby(destination_name)
            if not results:
                self.tts.speak(f"没有找到{destination_name}的位置")
                return

            # 选择第一个结果（最近的）
            destination_poi = results[0]
            self.destination = (
                float(destination_poi['longitude']),
                float(destination_poi['latitude'])
            )

            # 获取当前位置并计算方位信息
            current_location = self._get_current_location()
            if current_location:
                logging.info(f"当前位置: {current_location}")
                logging.info(f"目的地位置: {self.destination}")

                # 计算方位和距离
                bearing = calculate_bearing(current_location, self.destination)
                distance = calculate_distance(current_location, self.destination)
                direction = bearing_to_direction(bearing)

                logging.info(f"计算距离: {distance}米")

                # 检查最大导航范围（10公里）
                MAX_NAVIGATION_DISTANCE = 10000  # 10公里
                if distance > MAX_NAVIGATION_DISTANCE:
                    distance_km = distance / 1000
                    message = f"目的地{destination_poi['name']}距离过远（{distance_km:.1f}公里），超出最大导航范围10公里，请选择更近的目的地"
                    self.tts.speak(message)
                    logging.warning(f"目的地超出范围: {destination_poi['name']} - {distance_km:.1f}公里")
                    self.destination = None  # 清除目的地
                    return

                # 格式化距离描述
                if distance < 1000:
                    distance_desc = f"{int(distance)}米"
                else:
                    distance_desc = f"{distance/1000:.1f}公里"

                # 播报详细信息
                message = f"已设置目的地为{destination_poi['name']}，位于您的{direction}，直线距离约{distance_desc}"
                self.tts.speak(message)
                logging.info(f"目的地设置: {destination_poi['name']} - {self.destination} - {direction} - {distance_desc}")
            else:
                # 如果无法获取当前位置，只播报基本信息
                self.tts.speak(f"已设置目的地为{destination_poi['name']}")
                logging.info(f"目的地设置: {destination_poi['name']} - {self.destination}")

        except Exception as e:
            logging.error(f"设置目的地失败: {e}")
            self.tts.speak("设置目的地失败，请稍后重试")

    def _start_navigation(self):
        """开始导航到设置的目的地（完整流程，包含方位指引）"""
        if not self.destination:
            self.tts.speak("请先设置目的地，例如说'导航到天安门'")
            return

        # 获取当前位置（实际应用中应使用GPS）
        origin = self._get_current_location()
        if not origin:
            self.tts.speak("无法获取当前位置")
            return

        # 计算初始方位信息
        bearing = calculate_bearing(origin, self.destination)
        distance = calculate_distance(origin, self.destination)
        direction = bearing_to_direction(bearing)

        # 格式化距离
        if distance < 1000:
            distance_desc = f"{int(distance)}米"
        else:
            distance_desc = f"{distance/1000:.1f}公里"

        # 规划路径
        route = self.nav.plan_route(origin, self.destination)
        if route:
            self.nav.start_navigation(route)

            # 播报详细的导航开始信息
            start_message = f"导航开始，目的地位于{direction}，直线距离{distance_desc}。"

            # 如果有路径规划，添加路径信息
            if route.get("steps") and len(route["steps"]) > 0:
                first_step = route["steps"][0]
                if first_step.get("instruction"):
                    start_message += f"请{first_step['instruction']}"

            self.tts.speak(start_message)
            logging.info(f"导航开始: {direction} - {distance_desc}")
        else:
            self.tts.speak("路径规划失败，请重试")

    def _stop_navigation(self):
        """结束导航"""
        if self.nav.is_navigating:
            self.nav.stop_navigation()
            self.tts.speak("导航已结束")
        else:
            self.tts.speak("当前没有进行导航")

    def _pause_navigation(self):
        """暂停导航"""
        if self.nav.is_navigating and not self.is_paused:
            self.is_paused = True
            self.tts.speak("导航已暂停")

    def _resume_navigation(self):
        """继续导航"""
        if self.nav.is_navigating and self.is_paused:
            self.is_paused = False
            self.tts.speak("继续导航")

    def _recalculate_route(self):
        """重新规划路径"""
        if self.nav.is_navigating:
            current_location = self._get_current_location()
            if current_location and self.destination:
                route = self.nav.plan_route(current_location, self.destination)
                if route:
                    self.nav.start_navigation(route)
                    self.tts.speak("已重新规划路径")

    def _announce_location(self):
        """播报当前位置（增强版，包含目的地方位信息）"""
        if self.poi:
            try:
                # 通过POI模块获取当前位置
                current_location = self.poi.get_current_location_by_ip()
                if current_location:
                    # 解析经纬度
                    lng, lat = current_location.split(',')
                    location_text = f"当前位置：经度{lng}，纬度{lat}"

                    # 如果有设置目的地，播报相对方位信息
                    if self.destination:
                        origin = (float(lng), float(lat))
                        bearing = calculate_bearing(origin, self.destination)
                        distance = calculate_distance(origin, self.destination)
                        direction = bearing_to_direction(bearing)

                        if distance < 1000:
                            distance_desc = f"{int(distance)}米"
                        else:
                            distance_desc = f"{distance/1000:.1f}公里"

                        location_text += f"。目的地位于您的{direction}，距离约{distance_desc}"

                    self.tts.speak(location_text)
                else:
                    self.tts.speak("无法获取当前位置信息")
            except Exception as e:
                logging.error(f"获取位置失败: {e}")
                self.tts.speak("获取位置信息失败")
        else:
            self.tts.speak("位置服务未启用")

    def _announce_remaining(self):
        """播报剩余距离"""
        if self.nav.is_navigating and self.nav.current_route:
            remaining = self.nav.current_route["distance"]
            if remaining < 1000:
                self.tts.speak(f"距离目的地还有{int(remaining)}米")
            else:
                self.tts.speak(f"距离目的地还有{remaining / 1000:.1f}公里")
        else:
            self.tts.speak("当前没有进行导航，请先设置目的地并开始导航")

    def _announce_next_turn(self):
        """播报下一个路口"""
        if self.nav.is_navigating and self.nav.current_route:
            next_step = self.nav.current_route["steps"][self.nav.current_step_index]
            self.event_handler.on_navigation_point(next_step)
        else:
            self.tts.speak("当前没有进行导航，请先设置目的地并开始导航")

    def _get_current_location(self) -> Optional[Tuple[float, float]]:
        """获取当前位置（通过POI模块）"""
        if self.poi:
            try:
                # 通过POI模块获取当前位置
                current_location = self.poi.get_current_location_by_ip()
                if current_location:
                    # 解析经纬度
                    lng, lat = current_location.split(',')
                    return (float(lng), float(lat))
            except Exception as e:
                logging.error(f"获取当前位置失败: {e}")

        # 如果无法获取位置，返回None
        return None
