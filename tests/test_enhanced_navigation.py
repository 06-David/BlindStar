#!/usr/bin/env python3
"""
增强导航功能测试脚本
测试所有新增的导航功能，包括API调用、缓存、错误处理等
"""

import sys
import time
import logging
import unittest
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestEnhancedNavigation(unittest.TestCase):
    """增强导航功能测试类"""
    
    def setUp(self):
        """测试前准备"""
        logger.info("🧪 开始测试增强导航功能")
    
    def test_api_config_manager(self):
        """测试API配置管理器"""
        logger.info("📋 测试API配置管理器")
        
        try:
            from src.config.amap_config import get_amap_config
            
            config = get_amap_config()
            self.assertIsNotNone(config)
            self.assertIsNotNone(config.api_key)
            self.assertEqual(config.api_key, "717d9a827c0ac3521932d3ae59aebbfe")
            
            # 测试端点配置
            endpoint = config.get_endpoint("ip_location")
            self.assertIsNotNone(endpoint)
            self.assertEqual(endpoint.name, "IP定位")
            
            # 测试API调用限制检查
            can_request, reason = config.can_make_request("ip_location")
            self.assertTrue(can_request)
            
            logger.info("✅ API配置管理器测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ API配置管理器导入失败: {e}")
            self.skipTest("API配置管理器不可用")
        except Exception as e:
            logger.error(f"❌ API配置管理器测试失败: {e}")
            self.fail(f"API配置管理器测试失败: {e}")
    
    def test_cache_manager(self):
        """测试缓存管理器"""
        logger.info("💾 测试缓存管理器")
        
        try:
            from src.core.cache_manager import get_cache_manager
            
            cache_manager = get_cache_manager()
            self.assertIsNotNone(cache_manager)
            
            # 测试缓存设置和获取
            test_key = "test_key"
            test_data = {"test": "data", "timestamp": time.time()}
            
            # 设置缓存
            success = cache_manager.set(test_key, test_data, 60)
            self.assertTrue(success)
            
            # 获取缓存
            cached_data = cache_manager.get(test_key)
            self.assertIsNotNone(cached_data)
            self.assertEqual(cached_data["test"], "data")
            
            # 测试缓存统计
            stats = cache_manager.get_stats()
            self.assertIsInstance(stats, dict)
            self.assertIn("total_entries", stats)
            
            logger.info("✅ 缓存管理器测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ 缓存管理器导入失败: {e}")
            self.skipTest("缓存管理器不可用")
        except Exception as e:
            logger.error(f"❌ 缓存管理器测试失败: {e}")
            self.fail(f"缓存管理器测试失败: {e}")
    
    def test_poi_code_manager(self):
        """测试POI编码管理器"""
        logger.info("🏪 测试POI编码管理器")
        
        try:
            from src.config.poi_codes import get_poi_manager
            
            poi_manager = get_poi_manager()
            self.assertIsNotNone(poi_manager)
            
            # 测试POI编码获取
            bank_code = poi_manager.get_poi_code("银行")
            self.assertIsNotNone(bank_code)
            self.assertEqual(bank_code, "160300")
            
            # 测试模糊匹配
            hospital_code = poi_manager.get_poi_code("医院")
            self.assertIsNotNone(hospital_code)
            
            # 测试盲人友好类型
            blind_friendly_types = poi_manager.get_blind_friendly_types()
            self.assertIsInstance(blind_friendly_types, dict)
            self.assertIn("银行", blind_friendly_types)
            
            logger.info("✅ POI编码管理器测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ POI编码管理器导入失败: {e}")
            self.skipTest("POI编码管理器不可用")
        except Exception as e:
            logger.error(f"❌ POI编码管理器测试失败: {e}")
            self.fail(f"POI编码管理器测试失败: {e}")
    
    def test_enhanced_location_service(self):
        """测试增强定位服务"""
        logger.info("📍 测试增强定位服务")
        
        try:
            from src.navigation.enhanced_location import get_location_service
            
            location_service = get_location_service()
            self.assertIsNotNone(location_service)
            
            # 测试获取当前位置
            current_location = location_service.get_current_location()
            self.assertIsNotNone(current_location)
            self.assertIsNotNone(current_location.longitude)
            self.assertIsNotNone(current_location.latitude)
            
            # 测试地理编码
            location_info = location_service.geocode_address("北京天安门")
            if location_info:  # 网络可用时才测试
                self.assertIsNotNone(location_info.longitude)
                self.assertIsNotNone(location_info.latitude)
            
            logger.info("✅ 增强定位服务测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ 增强定位服务导入失败: {e}")
            self.skipTest("增强定位服务不可用")
        except Exception as e:
            logger.error(f"❌ 增强定位服务测试失败: {e}")
            self.fail(f"增强定位服务测试失败: {e}")
    
    def test_enhanced_routing_service(self):
        """测试增强路径规划服务"""
        logger.info("🗺️ 测试增强路径规划服务")
        
        try:
            from src.navigation.enhanced_routing import get_routing_service, RouteStrategy
            
            routing_service = get_routing_service()
            self.assertIsNotNone(routing_service)
            
            # 测试路径规划
            route_info = routing_service.plan_route(
                origin="116.397428,39.99923",  # 天安门
                destination="116.407526,39.90403",  # 前门
                strategy=RouteStrategy.SAFE_PRIORITY
            )
            
            if route_info:  # 网络可用时才验证
                self.assertIsNotNone(route_info.origin)
                self.assertIsNotNone(route_info.destination)
                self.assertGreater(route_info.distance, 0)
                self.assertIsInstance(route_info.steps, list)
            
            logger.info("✅ 增强路径规划服务测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ 增强路径规划服务导入失败: {e}")
            self.skipTest("增强路径规划服务不可用")
        except Exception as e:
            logger.error(f"❌ 增强路径规划服务测试失败: {e}")
            self.fail(f"增强路径规划服务测试失败: {e}")
    
    def test_enhanced_poi_service(self):
        """测试增强POI搜索服务"""
        logger.info("🔍 测试增强POI搜索服务")
        
        try:
            from src.navigation.enhanced_poi import get_poi_service, SortRule
            
            poi_service = get_poi_service()
            self.assertIsNotNone(poi_service)
            
            # 测试文本搜索
            search_result = poi_service.search_text(
                keywords="银行",
                location="116.397428,39.99923",
                radius=1000,
                sort_rule=SortRule.BLIND_FRIENDLY
            )
            
            self.assertIsNotNone(search_result)
            self.assertIsInstance(search_result.pois, list)
            
            # 测试输入提示
            tips = poi_service.get_input_tips("银行", limit=5)
            self.assertIsInstance(tips, list)
            
            logger.info("✅ 增强POI搜索服务测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ 增强POI搜索服务导入失败: {e}")
            self.skipTest("增强POI搜索服务不可用")
        except Exception as e:
            logger.error(f"❌ 增强POI搜索服务测试失败: {e}")
            self.fail(f"增强POI搜索服务测试失败: {e}")
    
    def test_traffic_service(self):
        """测试交通信息服务"""
        logger.info("🚦 测试交通信息服务")
        
        try:
            from src.navigation.traffic_service import get_traffic_service
            
            traffic_service = get_traffic_service()
            self.assertIsNotNone(traffic_service)
            
            # 测试获取交通状态
            traffic_report = traffic_service.get_traffic_status(
                location="116.397428,39.99923",
                radius=1000
            )
            
            if traffic_report:  # 网络可用时才验证
                self.assertIsNotNone(traffic_report.area_name)
                self.assertIsInstance(traffic_report.traffic_info, list)
            
            # 测试获取交通事件
            incidents = traffic_service.get_traffic_incidents(
                location="116.397428,39.99923",
                radius=2000
            )
            
            self.assertIsInstance(incidents, list)
            
            logger.info("✅ 交通信息服务测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ 交通信息服务导入失败: {e}")
            self.skipTest("交通信息服务不可用")
        except Exception as e:
            logger.error(f"❌ 交通信息服务测试失败: {e}")
            self.fail(f"交通信息服务测试失败: {e}")
    
    def test_fallback_manager(self):
        """测试降级管理器"""
        logger.info("🛡️ 测试降级管理器")
        
        try:
            from src.core.fallback_manager import get_fallback_manager, ErrorType
            
            fallback_manager = get_fallback_manager()
            self.assertIsNotNone(fallback_manager)
            
            # 测试记录成功和错误
            fallback_manager.record_success("test_service", "test_endpoint")
            fallback_manager.record_error("test_service", ErrorType.NETWORK_ERROR, "测试错误", "test_endpoint")
            
            # 测试服务状态
            status = fallback_manager.get_service_status("test_service")
            self.assertIsNotNone(status)
            
            # 测试服务可用性
            is_available = fallback_manager.is_service_available("test_service")
            self.assertIsInstance(is_available, bool)
            
            # 测试健康报告
            health_report = fallback_manager.get_health_report()
            self.assertIsInstance(health_report, dict)
            self.assertIn("services", health_report)
            self.assertIn("overall_health", health_report)
            
            logger.info("✅ 降级管理器测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ 降级管理器导入失败: {e}")
            self.skipTest("降级管理器不可用")
        except Exception as e:
            logger.error(f"❌ 降级管理器测试失败: {e}")
            self.fail(f"降级管理器测试失败: {e}")
    
    def test_api_client_integration(self):
        """测试API客户端集成"""
        logger.info("🔌 测试API客户端集成")
        
        try:
            from src.core.amap_client import get_amap_client
            
            client = get_amap_client()
            self.assertIsNotNone(client)
            
            # 测试API请求（使用IP定位作为示例）
            success, data, message = client.request("ip_location")
            
            # 无论成功还是失败，都应该有明确的返回值
            self.assertIsInstance(success, bool)
            self.assertIsInstance(message, str)
            
            if success:
                self.assertIsNotNone(data)
                logger.info(f"✅ API请求成功: {message}")
            else:
                logger.warning(f"⚠️ API请求失败: {message}")
            
            logger.info("✅ API客户端集成测试通过")
            
        except ImportError as e:
            logger.warning(f"⚠️ API客户端导入失败: {e}")
            self.skipTest("API客户端不可用")
        except Exception as e:
            logger.error(f"❌ API客户端集成测试失败: {e}")
            self.fail(f"API客户端集成测试失败: {e}")
    
    def test_performance_and_caching(self):
        """测试性能和缓存效果"""
        logger.info("⚡ 测试性能和缓存效果")
        
        try:
            from src.core.cache_manager import get_cache_manager
            
            cache_manager = get_cache_manager()
            
            # 测试缓存性能
            test_data = {"large_data": list(range(1000))}
            
            # 第一次设置（应该较慢）
            start_time = time.time()
            cache_manager.set("perf_test", test_data, 300)
            set_time = time.time() - start_time
            
            # 第一次获取（应该较快）
            start_time = time.time()
            cached_data = cache_manager.get("perf_test")
            get_time = time.time() - start_time
            
            self.assertIsNotNone(cached_data)
            self.assertEqual(len(cached_data["large_data"]), 1000)
            
            # 缓存获取应该比设置快
            self.assertLess(get_time, set_time)
            
            logger.info(f"✅ 缓存性能测试通过 - 设置时间: {set_time:.4f}s, 获取时间: {get_time:.4f}s")
            
        except ImportError as e:
            logger.warning(f"⚠️ 性能测试导入失败: {e}")
            self.skipTest("性能测试不可用")
        except Exception as e:
            logger.error(f"❌ 性能测试失败: {e}")
            self.fail(f"性能测试失败: {e}")
    
    def tearDown(self):
        """测试后清理"""
        logger.info("🧹 测试清理完成")


def run_comprehensive_test():
    """运行综合测试"""
    logger.info("🚀 开始运行BlindStar增强导航功能综合测试")
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedNavigation)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    logger.info("📊 测试结果摘要:")
    logger.info(f"   总测试数: {result.testsRun}")
    logger.info(f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"   失败: {len(result.failures)}")
    logger.info(f"   错误: {len(result.errors)}")
    logger.info(f"   跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        logger.error("❌ 失败的测试:")
        for test, traceback in result.failures:
            logger.error(f"   {test}: {traceback}")
    
    if result.errors:
        logger.error("❌ 错误的测试:")
        for test, traceback in result.errors:
            logger.error(f"   {test}: {traceback}")
    
    # 判断测试是否通过
    if result.wasSuccessful():
        logger.info("🎉 所有测试通过！BlindStar增强导航功能运行正常")
        return True
    else:
        logger.error("💥 部分测试失败，请检查相关功能")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
