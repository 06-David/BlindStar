#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlindStar 盲人导航系统主程序
集成定位、搜索、路径规划、实时导航和语音交互功能
"""

import sys
import os
import logging
import time
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blindstar.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BlindStarApp:
    """BlindStar主应用程序"""
    
    def __init__(self):
        self.location_service = None
        self.poi_service = None
        self.routing_service = None
        self.navigation_service = None
        self.tts_service = None
        self.speech_service = None

        self.current_location = None
        self.is_running = False
        self.voice_mode = False  # 语音模式标志
        
        logger.info("🌟 BlindStar盲人导航系统启动")
    
    def initialize_services(self) -> bool:
        """初始化所有服务"""
        try:
            logger.info("🚀 正在初始化BlindStar服务...")
            
            # 初始化语音服务
            from src.voice.tts_service import get_tts_service
            from src.voice.speech_recognition_service import get_speech_recognition_service

            self.tts_service = get_tts_service()
            self.speech_service = get_speech_recognition_service()

            # 设置语音服务之间的关联
            self.speech_service.set_tts_service(self.tts_service)

            # 检查语音识别是否可用
            self.voice_mode = self.speech_service.is_available()

            if self.voice_mode:
                self.tts_service.speak_info("BlindStar系统正在启动，语音识别已启用，请稍候")
                logger.info("✅ 语音识别模式已启用")
            else:
                self.tts_service.speak_info("BlindStar系统正在启动，将使用键盘输入模式，请稍候")
                logger.warning("⚠️ 语音识别不可用，使用键盘输入模式")
            
            # 初始化定位服务
            from src.navigation.enhanced_location import get_location_service
            self.location_service = get_location_service()
            
            # 初始化POI搜索服务
            from src.navigation.enhanced_poi import get_poi_service
            self.poi_service = get_poi_service()
            
            # 初始化路径规划服务
            from src.navigation.enhanced_routing import get_routing_service
            self.routing_service = get_routing_service()
            
            # 初始化实时导航服务
            from src.navigation.realtime_navigation import get_realtime_navigation_service
            self.navigation_service = get_realtime_navigation_service(
                self.location_service, self.routing_service, self.tts_service
            )
            
            logger.info("✅ 所有服务初始化完成")

            if self.voice_mode:
                self.tts_service.speak_info("BlindStar系统启动完成，语音识别已就绪。您可以直接说出命令，或按回车键查看菜单")
                # 启动语音监听
                self.speech_service.start_listening()
            else:
                self.tts_service.speak_info("BlindStar系统启动完成，请使用键盘输入")

            return True
            
        except Exception as e:
            logger.error(f"❌ 服务初始化失败: {e}")
            if self.tts_service:
                self.tts_service.speak_info("系统初始化失败")
            return False
    
    def get_current_location(self) -> bool:
        """获取当前位置"""
        try:
            self.tts_service.speak_info("正在获取您的当前位置...")

            self.current_location = self.location_service.get_current_location()

            if self.current_location:
                logger.info(f"📍 当前位置: {self.current_location.address}")
                self.tts_service.speak_info(f"定位成功。您当前位置是：{self.current_location.address}")

                # 如果定位精度不够，提示用户可以使用位置校正
                if self.current_location.confidence < 0.8:
                    tip_text = "当前定位精度有限，如需更精确定位请选择位置校正功能"
                    logger.info(f"💡 {tip_text}")
                    self.tts_service.speak_info(tip_text)

                return True
            else:
                logger.error("❌ 定位失败")
                self.tts_service.speak_info("定位失败，建议使用位置校正功能手动输入准确地址")
                return False

        except Exception as e:
            logger.error(f"❌ 获取位置异常: {e}")
            self.tts_service.speak_info("获取位置时出现错误")
            return False

    def correct_location(self) -> bool:
        """位置校正功能"""
        try:
            logger.info("🔧 开始位置校正...")
            self.tts_service.speak_info("位置校正功能。请输入您的准确地址")

            print("\n🔧 位置校正功能")
            print("请输入您的准确地址（例如：内蒙古自治区呼和浩特市赛罕区桥靠西街26号）")

            address = self.get_user_text_input("请说出或输入您的准确地址")

            if not address:
                self.tts_service.speak_info("地址不能为空")
                return False

            self.tts_service.speak_info(f"正在校正位置到：{address}")

            corrected_location = self.location_service.correct_location(address)

            if corrected_location:
                self.current_location = corrected_location
                success_text = f"位置校正成功。校正后地址：{corrected_location.address}"
                logger.info(f"✅ {success_text}")
                self.tts_service.speak_info(success_text)
                return True
            else:
                error_text = "位置校正失败，请检查地址格式是否正确"
                logger.error(f"❌ {error_text}")
                self.tts_service.speak_info(error_text)
                return False

        except Exception as e:
            logger.error(f"❌ 位置校正异常: {e}")
            self.tts_service.speak_info("位置校正时出现错误")
            return False
    
    def search_nearby_poi(self, keyword: str) -> list:
        """搜索附近POI"""
        try:
            if not self.current_location:
                self.tts_service.speak_info("请先获取当前位置")
                return []
            
            self.tts_service.speak_info(f"正在搜索附近的{keyword}...")
            
            # 使用坐标字符串而不是对象
            location_str = f"{self.current_location.longitude},{self.current_location.latitude}"
            
            from src.navigation.enhanced_poi import SortRule
            search_result = self.poi_service.search_around(
                location=location_str,
                keywords=keyword,
                radius=1000,
                page_size=5,
                sort_rule=SortRule.BLIND_FRIENDLY
            )
            
            if search_result.pois:
                logger.info(f"✅ 找到{len(search_result.pois)}个{keyword}")
                
                # 播报搜索结果
                result_text = f"找到{len(search_result.pois)}个{keyword}："
                for i, poi in enumerate(search_result.pois, 1):
                    logger.info(f"   {i}. {poi.name} - 距离{poi.distance:.0f}米")
                    result_text += f"{i}、{poi.name}，距离{poi.distance:.0f}米；"
                
                self.tts_service.speak_info(result_text)
                return search_result.pois
            else:
                logger.warning(f"⚠️ 附近没有找到{keyword}")
                self.tts_service.speak_info(f"附近没有找到{keyword}")
                return []
                
        except Exception as e:
            logger.error(f"❌ 搜索POI异常: {e}")
            self.tts_service.speak_info("搜索时出现错误")
            return []
    
    def start_navigation_to_poi(self, poi) -> bool:
        """开始导航到POI"""
        try:
            if not self.current_location:
                self.tts_service.speak_info("请先获取当前位置")
                return False
            
            logger.info(f"🧭 开始导航到: {poi.name}")
            
            # 简单坐标类
            class Coordinates:
                def __init__(self, longitude, latitude):
                    self.longitude = longitude
                    self.latitude = latitude
            
            destination = Coordinates(poi.location.longitude, poi.location.latitude)
            
            success = self.navigation_service.start_navigation(destination, poi.name)
            
            if success:
                logger.info("✅ 导航启动成功")
                return True
            else:
                logger.error("❌ 导航启动失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 启动导航异常: {e}")
            self.tts_service.speak_info("启动导航时出现错误")
            return False
    
    def show_main_menu(self):
        """显示主菜单"""
        # 检查是否正在导航
        is_navigating = (self.navigation_service and
                        hasattr(self.navigation_service, 'state') and
                        self.navigation_service.state.value == 'navigating')

        if self.voice_mode:
            if is_navigating:
                # 导航状态下的简化提示
                menu_text = """
🧭 正在导航中 - 语音命令：
• "导航状态" - 查看导航信息
• "停止导航" - 停止当前导航
• "退出" - 退出系统

🎤 请说出命令："""
                print(menu_text)
                # 导航时不播报冗长的菜单，只简单提示
                # self.tts_service.speak_info("正在导航中，可说：导航状态、停止导航、退出")
            else:
                # 非导航状态下的完整菜单
                menu_text = """
🎤 BlindStar盲人导航系统 - 语音命令模式：

语音命令示例：
• "获取位置" 或 "定位" - 获取当前位置
• "位置校正" 或 "校正位置" - 手动输入准确地址
• "搜索医院" 或 "找医院" - 搜索附近医院
• "搜索银行" 或 "找银行" - 搜索附近银行
• "搜索超市" 或 "找超市" - 搜索附近超市
• "搜索餐厅" 或 "找餐厅" - 搜索附近餐厅
• "导航状态" 或 "查看状态" - 查看导航状态
• "停止导航" - 停止当前导航
• "帮助" - 查看使用说明
• "退出" 或 "再见" - 退出系统

您也可以按回车键查看数字菜单，或直接说出命令："""

                print(menu_text)
                self.tts_service.speak_info("语音命令模式已启用。您可以直接说出命令，比如：获取位置、搜索医院、找银行等。也可以按回车键查看完整菜单")
        else:
            if is_navigating:
                # 导航状态下的简化菜单
                menu_text = """
🧭 正在导航中：
7. 查看导航状态
8. 停止导航
0. 退出系统

请输入选项编号："""
                print(menu_text)
                self.tts_service.speak_info("正在导航中，可选择：7导航状态，8停止导航，0退出")
            else:
                # 非导航状态下的完整菜单
                menu_text = """
BlindStar盲人导航系统主菜单：
1. 获取当前位置
2. 位置校正（手动输入准确地址）
3. 搜索附近医院
4. 搜索附近银行
5. 搜索附近超市
6. 搜索附近餐厅
7. 查看导航状态
8. 停止导航
9. 帮助
0. 退出系统

请输入选项编号："""

                print(menu_text)
                self.tts_service.speak_info("请选择功能：1获取位置，2位置校正，3搜索医院，4搜索银行，5搜索超市，6搜索餐厅，7导航状态，8停止导航，9帮助，0退出")

    def get_user_input(self, prompt: str = "") -> str:
        """获取用户输入（支持语音和键盘）"""
        if self.voice_mode:
            # 语音模式：先尝试获取语音命令，如果没有则使用键盘输入
            print(f"\n🎤 {prompt if prompt else '请说出您的命令，或按回车键使用键盘输入：'}")

            # 等待语音命令（非阻塞）
            voice_command = self.speech_service.get_voice_command(timeout=3.0)
            if voice_command:
                logger.info(f"🎤 收到语音命令: {voice_command}")
                return voice_command

            # 如果没有语音命令，使用键盘输入
            try:
                keyboard_input = input("键盘输入: ").strip()
                if keyboard_input:
                    return keyboard_input
                else:
                    # 如果键盘也没有输入，再次尝试语音
                    self.tts_service.speak_info("请说出您的命令")
                    voice_command = self.speech_service.get_voice_command(timeout=10.0)
                    return voice_command if voice_command else ""
            except KeyboardInterrupt:
                return "0"  # 退出
        else:
            # 键盘模式
            try:
                return input(prompt if prompt else "请输入: ").strip()
            except KeyboardInterrupt:
                return "0"  # 退出

    def get_user_choice(self, prompt: str, max_choice: int = 5) -> str:
        """获取用户选择（支持语音和键盘）"""
        if self.voice_mode:
            print(f"\n🎤 {prompt}")

            # 暂停主语音监听，避免音频流冲突
            was_listening = False
            if self.speech_service and hasattr(self.speech_service, 'is_listening'):
                was_listening = self.speech_service.is_listening
                if was_listening:
                    logger.info("🎤 暂停主语音监听进入选择模式")
                    # 注意：这里不调用pause_listening，因为我们要继续使用主监听循环
                    # 只是切换到选择模式

            try:
                # 先尝试语音选择
                voice_choice = self.speech_service.get_voice_choice(prompt, max_choice)
                if voice_choice:
                    return voice_choice

                # 语音失败，使用键盘输入
                try:
                    return input("键盘输入选择: ").strip()
                except KeyboardInterrupt:
                    return ""
            finally:
                # 恢复正常命令模式
                if was_listening and self.speech_service:
                    logger.info("🎤 恢复主语音监听命令模式")
                    # 确保退出选择模式
                    if hasattr(self.speech_service, 'choice_mode'):
                        self.speech_service.choice_mode = False
                        self.speech_service.choice_result = None
                        self.speech_service.choice_max = 0
        else:
            # 键盘模式
            try:
                return input(f"{prompt}: ").strip()
            except KeyboardInterrupt:
                return ""

    def get_user_text_input(self, prompt: str) -> str:
        """获取用户文本输入（如地址）"""
        if self.voice_mode:
            print(f"\n🎤 {prompt}")

            # 先尝试语音输入
            voice_input = self.speech_service.get_voice_input(prompt)
            if voice_input:
                return voice_input

            # 语音失败，使用键盘输入
            try:
                return input("键盘输入: ").strip()
            except KeyboardInterrupt:
                return ""
        else:
            # 键盘模式
            try:
                return input(f"{prompt}: ").strip()
            except KeyboardInterrupt:
                return ""
    
    def handle_user_input(self, choice: str) -> bool:
        """处理用户输入"""
        try:
            choice = choice.strip()
            
            if choice == "1":
                self.get_current_location()

            elif choice == "2":
                self.correct_location()

            elif choice == "3":
                pois = self.search_nearby_poi("医院")
                if pois:
                    self._handle_poi_selection(pois, "医院")

            elif choice == "4":
                pois = self.search_nearby_poi("银行")
                if pois:
                    self._handle_poi_selection(pois, "银行")
                    
            elif choice == "5":
                pois = self.search_nearby_poi("超市")
                if pois:
                    self._handle_poi_selection(pois, "超市")

            elif choice == "6":
                pois = self.search_nearby_poi("餐厅")
                if pois:
                    self._handle_poi_selection(pois, "餐厅")
                    
            elif choice == "7":
                self._show_navigation_status()

            elif choice == "8":
                self._stop_navigation()

            elif choice == "9":
                self._show_help()

            elif choice == "0":
                return False  # 退出
                
            else:
                self.tts_service.speak_info("无效选项，请重新选择")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ 处理用户输入异常: {e}")
            if self.tts_service:
                self.tts_service.speak_info("处理输入时出现错误")
            return True
    
    def _handle_poi_selection(self, pois: list, category: str):
        """处理POI选择"""
        try:
            if len(pois) == 1:
                # 只有一个结果，直接导航
                self.start_navigation_to_poi(pois[0])
            else:
                # 多个结果，让用户选择
                self.tts_service.speak_info(f"找到多个{category}，请选择：")
                
                for i, poi in enumerate(pois, 1):
                    print(f"{i}. {poi.name} - 距离{poi.distance:.0f}米")
                
                self.tts_service.speak_info("请输入编号选择目的地，或输入0返回主菜单")
                
                while True:
                    try:
                        selection = self.get_user_choice(f"请选择{category}编号（1-{len(pois)}），或说'返回'", len(pois))

                        if selection == "0" or "返回" in selection:
                            break

                        index = int(selection) - 1
                        if 0 <= index < len(pois):
                            selected_poi = pois[index]
                            self.start_navigation_to_poi(selected_poi)
                            break
                        else:
                            self.tts_service.speak_info("无效编号，请重新选择")

                    except (ValueError, TypeError):
                        self.tts_service.speak_info("请说出或输入有效数字")
                        
        except Exception as e:
            logger.error(f"❌ 处理POI选择异常: {e}")
            self.tts_service.speak_info("选择过程中出现错误")
    
    def _show_navigation_status(self):
        """显示导航状态"""
        try:
            status = self.navigation_service.get_navigation_status()
            
            if status["state"] == "idle":
                self.tts_service.speak_info("当前没有进行导航")
            elif status["state"] == "navigating":
                current_step = status.get("current_step", 0)
                total_steps = status.get("total_steps", 0)
                instruction = status.get("current_instruction", "")
                
                status_text = f"正在导航中。当前第{current_step}步，共{total_steps}步。当前指令：{instruction}"
                self.tts_service.speak_info(status_text)
            elif status["state"] == "completed":
                self.tts_service.speak_info("导航已完成")
            else:
                self.tts_service.speak_info(f"导航状态：{status['state']}")
                
        except Exception as e:
            logger.error(f"❌ 获取导航状态异常: {e}")
            self.tts_service.speak_info("获取导航状态时出现错误")
    
    def _stop_navigation(self):
        """停止导航"""
        try:
            self.navigation_service.stop_navigation()
            logger.info("🛑 导航已停止")
            
        except Exception as e:
            logger.error(f"❌ 停止导航异常: {e}")
            self.tts_service.speak_info("停止导航时出现错误")
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
BlindStar盲人导航系统帮助：

主要功能：
- 智能定位：获取您的精确位置
- POI搜索：搜索附近的医院、银行、超市等设施
- 路径规划：规划最优的步行路径
- 实时导航：提供逐步的语音导航指令
- 语音交互：全程语音播报和指导

使用方法：
1. 首先获取当前位置
2. 搜索您需要的设施类型
3. 选择目标设施开始导航
4. 跟随语音指令到达目的地

注意事项：
- 请确保网络连接正常
- 导航过程中请注意安全
- 如有问题可随时停止导航

技术支持：BlindStar团队
"""
        
        print(help_text)
        self.tts_service.speak_info("BlindStar支持智能定位、设施搜索、路径规划和实时导航。使用时请先获取位置，然后搜索目标，选择后开始导航。详细信息请查看屏幕显示")
    
    def run(self):
        """运行主程序"""
        try:
            # 初始化服务
            if not self.initialize_services():
                return
            
            self.is_running = True
            
            # 主循环
            while self.is_running:
                try:
                    self.show_main_menu()

                    # 获取用户输入（支持语音和键盘）
                    choice = self.get_user_input()

                    if not choice:
                        continue

                    if not self.handle_user_input(choice):
                        break

                    print("\n" + "="*50 + "\n")

                except KeyboardInterrupt:
                    logger.info("用户中断程序")
                    if self.tts_service:
                        self.tts_service.speak_info("程序已退出，再见")
                    break
                except Exception as e:
                    logger.error(f"❌ 主循环异常: {e}")
                    if self.tts_service:
                        self.tts_service.speak_info("程序出现错误，请重试")
            
            # 清理资源
            self._cleanup()
            
        except Exception as e:
            logger.error(f"❌ 程序运行异常: {e}")
            if self.tts_service:
                self.tts_service.speak_info("程序运行出现严重错误")
    
    def _cleanup(self):
        """清理资源"""
        try:
            logger.info("🧹 正在清理资源...")
            
            if self.navigation_service:
                self.navigation_service.stop_navigation()

            if self.speech_service:
                self.speech_service.stop_listening()

            if self.tts_service:
                self.tts_service.speak_info("感谢使用BlindStar盲人导航系统，再见")
                time.sleep(2)  # 等待语音播放完成
            
            logger.info("✅ 资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 资源清理异常: {e}")

def main():
    """主函数"""
    try:
        print("🌟 欢迎使用BlindStar盲人导航系统 🌟")
        print("=" * 50)
        
        app = BlindStarApp()
        app.run()
        
    except Exception as e:
        logger.error(f"❌ 程序启动失败: {e}")
        print(f"程序启动失败: {e}")

if __name__ == "__main__":
    main()
