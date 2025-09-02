#!/usr/bin/env python3
"""
BlindStar增强导航功能演示脚本
展示如何使用所有新增的导航功能
"""

import sys
import time
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_enhanced_location():
    """演示增强定位功能"""
    logger.info("🌍 === 增强定位功能演示 ===")
    
    try:
        from src.navigation.enhanced_location import get_location_service
        
        location_service = get_location_service()
        
        # 获取当前位置
        logger.info("📍 获取当前位置...")
        current_location = location_service.get_current_location()
        
        logger.info(f"当前位置: {current_location.address}")
        logger.info(f"坐标: {current_location.longitude}, {current_location.latitude}")
        logger.info(f"定位类型: {current_location.location_type}")
        logger.info(f"置信度: {current_location.confidence:.2f}")
        
        # 地理编码示例
        logger.info("🗺️ 地理编码示例...")
        beijing_location = location_service.geocode_address("北京天安门广场")
        if beijing_location:
            logger.info(f"天安门广场坐标: {beijing_location.longitude}, {beijing_location.latitude}")
        
        return current_location
        
    except Exception as e:
        logger.error(f"增强定位功能演示失败: {e}")
        return None


def demo_enhanced_poi_search(location):
    """演示增强POI搜索功能"""
    logger.info("🏪 === 增强POI搜索功能演示 ===")
    
    try:
        from src.navigation.enhanced_poi import get_poi_service, SortRule
        
        poi_service = get_poi_service()
        
        # 搜索附近的银行
        logger.info("🏦 搜索附近的银行...")
        search_result = poi_service.search_text(
            keywords="银行",
            location=location.to_coordinates() if location else "116.397428,39.99923",
            radius=1000,
            sort_rule=SortRule.BLIND_FRIENDLY
        )
        
        logger.info(f"找到 {len(search_result.pois)} 个银行")
        for i, poi in enumerate(search_result.pois[:3]):  # 显示前3个
            logger.info(f"  {i+1}. {poi.name}")
            logger.info(f"     地址: {poi.address}")
            logger.info(f"     距离: {poi.distance:.0f}米")
            logger.info(f"     无障碍评分: {poi.accessibility_score:.2f}")
            logger.info(f"     说明: {poi.accessibility_notes}")
        
        # 输入提示示例
        logger.info("💡 输入提示示例...")
        tips = poi_service.get_input_tips("医院", limit=5)
        logger.info("输入'医院'的提示:")
        for tip in tips:
            logger.info(f"  - {tip['name']} ({tip['district']})")
        
        return search_result.pois[0] if search_result.pois else None
        
    except Exception as e:
        logger.error(f"增强POI搜索功能演示失败: {e}")
        return None


def demo_enhanced_routing(origin_location, destination_poi):
    """演示增强路径规划功能"""
    logger.info("🗺️ === 增强路径规划功能演示 ===")
    
    try:
        from src.navigation.enhanced_routing import get_routing_service, RouteStrategy
        
        routing_service = get_routing_service()
        
        if not origin_location or not destination_poi:
            logger.warning("缺少起点或终点信息，使用默认坐标")
            origin = "116.397428,39.99923"  # 天安门
            destination = "116.407526,39.90403"  # 前门
        else:
            origin = origin_location.to_coordinates()
            destination = destination_poi.location.to_coordinates()
        
        # 规划盲人友好路径
        logger.info("🚶 规划盲人友好路径...")
        route_info = routing_service.plan_route(
            origin=origin,
            destination=destination,
            strategy=RouteStrategy.SAFE_PRIORITY
        )
        
        if route_info:
            logger.info(f"路径规划成功!")
            logger.info(f"总距离: {route_info.distance:.0f}米")
            logger.info(f"预计时间: {route_info.duration//60}分{route_info.duration%60}秒")
            logger.info(f"人行道覆盖率: {route_info.sidewalk_coverage:.1%}")
            logger.info(f"盲道覆盖率: {route_info.blind_path_coverage:.1%}")
            logger.info(f"安全评分: {route_info.overall_safety_score:.2f}")
            
            logger.info("📋 路径步骤:")
            for i, step in enumerate(route_info.steps[:3]):  # 显示前3步
                logger.info(f"  {i+1}. {step.instruction}")
                logger.info(f"     距离: {step.distance:.0f}米")
                logger.info(f"     道路: {step.road_name}")
                logger.info(f"     危险等级: {step.hazard_level}")
                if step.accessibility_note:
                    logger.info(f"     无障碍说明: {step.accessibility_note}")
        else:
            logger.warning("路径规划失败")
        
        return route_info
        
    except Exception as e:
        logger.error(f"增强路径规划功能演示失败: {e}")
        return None


def demo_traffic_service(location):
    """演示交通信息服务"""
    logger.info("🚦 === 交通信息服务演示 ===")
    
    try:
        from src.navigation.traffic_service import get_traffic_service
        
        traffic_service = get_traffic_service()
        
        # 获取交通状态
        logger.info("🚗 获取周边交通状态...")
        traffic_report = traffic_service.get_traffic_status(
            location=location.to_coordinates() if location else "116.397428,39.99923",
            radius=1000
        )
        
        if traffic_report:
            logger.info(f"区域: {traffic_report.area_name}")
            logger.info(f"整体状态: {traffic_report.overall_status.value}")
            logger.info(f"出行建议: {traffic_report.travel_advice}")
            
            if traffic_report.traffic_info:
                logger.info("道路状况:")
                for info in traffic_report.traffic_info[:3]:  # 显示前3条
                    logger.info(f"  - {info.road_name}: {info.blind_friendly_description}")
                    logger.info(f"    安全等级: {info.safety_level}/5")
                    logger.info(f"    推荐通行: {'是' if info.recommended else '否'}")
        
        # 获取交通事件
        logger.info("⚠️ 获取交通事件...")
        incidents = traffic_service.get_traffic_incidents(
            location=location.to_coordinates() if location else "116.397428,39.99923",
            radius=2000
        )
        
        if incidents:
            logger.info(f"发现 {len(incidents)} 个交通事件:")
            for incident in incidents[:2]:  # 显示前2个
                logger.info(f"  - {incident.audio_description}")
                logger.info(f"    安全警告: {incident.safety_warning}")
                logger.info(f"    替代建议: {incident.alternative_route}")
        else:
            logger.info("当前区域无特殊交通事件")
        
    except Exception as e:
        logger.error(f"交通信息服务演示失败: {e}")


def demo_cache_and_performance():
    """演示缓存和性能优化"""
    logger.info("⚡ === 缓存和性能优化演示 ===")
    
    try:
        from src.core.cache_manager import get_cache_manager
        from src.core.amap_client import get_amap_client
        
        cache_manager = get_cache_manager()
        api_client = get_amap_client()
        
        # 缓存统计
        logger.info("📊 缓存统计:")
        stats = cache_manager.get_stats()
        logger.info(f"  总缓存条目: {stats['total_entries']}")
        logger.info(f"  内存使用: {stats['memory_cache']['memory_usage_mb']:.2f}MB")
        logger.info(f"  磁盘使用: {stats['disk_cache']['disk_usage_mb']:.2f}MB")
        
        # API调用统计
        if hasattr(api_client, 'config') and api_client.config:
            api_stats = api_client.config.get_api_stats()
            logger.info("📈 API调用统计:")
            for endpoint, stats in api_stats.items():
                if stats['total_calls'] > 0:
                    logger.info(f"  {endpoint}:")
                    logger.info(f"    总调用: {stats['total_calls']}")
                    logger.info(f"    成功: {stats['success_count']}")
                    logger.info(f"    失败: {stats['error_count']}")
        
        # 性能测试
        logger.info("🏃 性能测试...")
        test_data = {"performance_test": list(range(100))}
        
        # 测试缓存写入性能
        start_time = time.time()
        cache_manager.set("perf_test", test_data, 300)
        write_time = time.time() - start_time
        
        # 测试缓存读取性能
        start_time = time.time()
        cached_data = cache_manager.get("perf_test")
        read_time = time.time() - start_time
        
        logger.info(f"缓存写入时间: {write_time*1000:.2f}ms")
        logger.info(f"缓存读取时间: {read_time*1000:.2f}ms")
        logger.info(f"性能提升: {write_time/read_time:.1f}x")
        
    except Exception as e:
        logger.error(f"缓存和性能优化演示失败: {e}")


def demo_fallback_and_error_handling():
    """演示降级策略和错误处理"""
    logger.info("🛡️ === 降级策略和错误处理演示 ===")
    
    try:
        from src.core.fallback_manager import get_fallback_manager, ErrorType
        
        fallback_manager = get_fallback_manager()
        
        # 模拟服务调用
        logger.info("📞 模拟服务调用...")
        fallback_manager.record_success("demo_service", "test_endpoint")
        fallback_manager.record_error("demo_service", ErrorType.NETWORK_ERROR, "模拟网络错误", "test_endpoint")
        
        # 检查服务状态
        status = fallback_manager.get_service_status("demo_service")
        is_available = fallback_manager.is_service_available("demo_service")
        
        logger.info(f"服务状态: {status.value}")
        logger.info(f"服务可用: {'是' if is_available else '否'}")
        
        # 获取降级数据示例
        logger.info("📦 获取降级数据示例...")
        fallback_data = fallback_manager.get_fallback_data(
            "poi_service", 
            "search", 
            keyword="银行", 
            location="116.397428,39.99923"
        )
        
        if fallback_data:
            logger.info(f"找到离线POI数据: {len(fallback_data)} 条")
        else:
            logger.info("暂无离线POI数据")
        
        # 健康报告
        logger.info("📋 系统健康报告:")
        health_report = fallback_manager.get_health_report()
        logger.info(f"  整体健康状态: {health_report['overall_health']}")
        logger.info(f"  总错误数: {health_report['total_errors']}")
        logger.info(f"  最近错误数: {health_report['recent_errors']}")
        
        for service_name, service_info in health_report['services'].items():
            logger.info(f"  {service_name}:")
            logger.info(f"    状态: {service_info['status']}")
            logger.info(f"    成功率: {service_info['success_rate']:.1%}")
            logger.info(f"    可用性: {service_info['availability']:.1%}")
        
    except Exception as e:
        logger.error(f"降级策略和错误处理演示失败: {e}")


def main():
    """主演示函数"""
    logger.info("🚀 BlindStar增强导航功能演示开始")
    logger.info("=" * 60)
    
    try:
        # 1. 增强定位功能演示
        current_location = demo_enhanced_location()
        time.sleep(1)
        
        # 2. 增强POI搜索功能演示
        destination_poi = demo_enhanced_poi_search(current_location)
        time.sleep(1)
        
        # 3. 增强路径规划功能演示
        route_info = demo_enhanced_routing(current_location, destination_poi)
        time.sleep(1)
        
        # 4. 交通信息服务演示
        demo_traffic_service(current_location)
        time.sleep(1)
        
        # 5. 缓存和性能优化演示
        demo_cache_and_performance()
        time.sleep(1)
        
        # 6. 降级策略和错误处理演示
        demo_fallback_and_error_handling()
        
        logger.info("=" * 60)
        logger.info("🎉 BlindStar增强导航功能演示完成")
        logger.info("💡 所有功能已成功集成，为盲人用户提供更好的导航体验")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        logger.error("请检查相关模块是否正确安装和配置")


if __name__ == "__main__":
    main()
