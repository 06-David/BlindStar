#!/usr/bin/env python3
"""
降级策略和错误处理管理器
提供API服务降级、离线模式、错误恢复等功能
"""

import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import pickle

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """服务状态"""
    HEALTHY = "healthy"         # 健康
    DEGRADED = "degraded"       # 降级
    OFFLINE = "offline"         # 离线
    MAINTENANCE = "maintenance" # 维护中


class ErrorType(Enum):
    """错误类型"""
    NETWORK_ERROR = "network_error"       # 网络错误
    API_LIMIT = "api_limit"              # API限制
    TIMEOUT = "timeout"                  # 超时
    AUTH_ERROR = "auth_error"            # 认证错误
    SERVER_ERROR = "server_error"        # 服务器错误
    DATA_ERROR = "data_error"            # 数据错误
    UNKNOWN = "unknown"                  # 未知错误


@dataclass
class ErrorRecord:
    """错误记录"""
    error_type: ErrorType
    message: str
    timestamp: float
    endpoint: str = ""
    retry_count: int = 0
    resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class ServiceHealth:
    """服务健康状态"""
    service_name: str
    status: ServiceStatus = ServiceStatus.HEALTHY
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[ErrorRecord] = None
    last_success: float = 0.0
    degraded_since: float = 0.0
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        total = self.error_count + self.success_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    def get_availability(self) -> float:
        """获取可用性"""
        if self.status == ServiceStatus.HEALTHY:
            return 1.0
        elif self.status == ServiceStatus.DEGRADED:
            return 0.5
        else:
            return 0.0


class OfflineDataManager:
    """离线数据管理器"""
    
    def __init__(self, offline_dir: str = "data/offline"):
        self.offline_dir = Path(offline_dir)
        self.offline_dir.mkdir(parents=True, exist_ok=True)
        
        # 离线数据文件
        self.poi_data_file = self.offline_dir / "poi_data.json"
        self.location_data_file = self.offline_dir / "location_data.json"
        self.route_data_file = self.offline_dir / "route_data.json"
        
        # 加载离线数据
        self.offline_poi_data = self._load_offline_data(self.poi_data_file)
        self.offline_location_data = self._load_offline_data(self.location_data_file)
        self.offline_route_data = self._load_offline_data(self.route_data_file)
        
        logger.info("✅ 离线数据管理器初始化完成")
    
    def _load_offline_data(self, file_path: Path) -> Dict[str, Any]:
        """加载离线数据"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载离线数据失败 {file_path}: {e}")
        return {}
    
    def save_poi_data(self, keyword: str, location: str, data: List[Dict[str, Any]]):
        """保存POI数据到离线存储"""
        try:
            cache_key = f"{keyword}_{location}"
            self.offline_poi_data[cache_key] = {
                'data': data,
                'timestamp': time.time(),
                'keyword': keyword,
                'location': location
            }
            
            with open(self.poi_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.offline_poi_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存离线POI数据失败: {e}")
    
    def get_offline_poi_data(self, keyword: str, location: str) -> Optional[List[Dict[str, Any]]]:
        """获取离线POI数据"""
        try:
            cache_key = f"{keyword}_{location}"
            if cache_key in self.offline_poi_data:
                cached_data = self.offline_poi_data[cache_key]
                # 检查数据是否过期（7天）
                if time.time() - cached_data['timestamp'] < 7 * 24 * 3600:
                    return cached_data['data']
        except Exception as e:
            logger.error(f"获取离线POI数据失败: {e}")
        return None
    
    def save_location_data(self, ip: str, data: Dict[str, Any]):
        """保存位置数据到离线存储"""
        try:
            self.offline_location_data[ip] = {
                'data': data,
                'timestamp': time.time()
            }
            
            with open(self.location_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.offline_location_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存离线位置数据失败: {e}")
    
    def get_offline_location_data(self, ip: str) -> Optional[Dict[str, Any]]:
        """获取离线位置数据"""
        try:
            if ip in self.offline_location_data:
                cached_data = self.offline_location_data[ip]
                # 检查数据是否过期（1天）
                if time.time() - cached_data['timestamp'] < 24 * 3600:
                    return cached_data['data']
        except Exception as e:
            logger.error(f"获取离线位置数据失败: {e}")
        return None


class FallbackManager:
    """降级策略和错误处理管理器"""
    
    def __init__(self):
        self.services = {}  # 服务健康状态
        self.error_history = []  # 错误历史
        self.offline_manager = OfflineDataManager()
        self.lock = threading.RLock()
        
        # 降级策略配置
        self.degradation_config = {
            'error_threshold': 5,        # 错误阈值
            'success_rate_threshold': 0.7,  # 成功率阈值
            'recovery_threshold': 10,    # 恢复阈值
            'offline_threshold': 20,     # 离线阈值
            'health_check_interval': 60  # 健康检查间隔
        }
        
        # 启动健康检查
        self._start_health_monitor()
        
        logger.info("✅ 降级策略管理器初始化完成")
    
    def record_success(self, service_name: str, endpoint: str = ""):
        """记录成功调用"""
        with self.lock:
            if service_name not in self.services:
                self.services[service_name] = ServiceHealth(service_name)
            
            service = self.services[service_name]
            service.success_count += 1
            service.last_success = time.time()
            
            # 检查是否可以恢复
            self._check_service_recovery(service)
    
    def record_error(self, service_name: str, error_type: ErrorType, message: str, endpoint: str = ""):
        """记录错误"""
        with self.lock:
            if service_name not in self.services:
                self.services[service_name] = ServiceHealth(service_name)
            
            service = self.services[service_name]
            service.error_count += 1
            
            # 创建错误记录
            error_record = ErrorRecord(
                error_type=error_type,
                message=message,
                timestamp=time.time(),
                endpoint=endpoint
            )
            
            service.last_error = error_record
            self.error_history.append(error_record)
            
            # 限制错误历史长度
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]
            
            # 检查是否需要降级
            self._check_service_degradation(service)
    
    def get_service_status(self, service_name: str) -> ServiceStatus:
        """获取服务状态"""
        with self.lock:
            if service_name in self.services:
                return self.services[service_name].status
            return ServiceStatus.HEALTHY
    
    def is_service_available(self, service_name: str) -> bool:
        """检查服务是否可用"""
        status = self.get_service_status(service_name)
        return status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    def get_fallback_data(self, service_name: str, operation: str, **kwargs) -> Optional[Any]:
        """获取降级数据"""
        try:
            if service_name == "poi_service" and operation == "search":
                keyword = kwargs.get('keyword', '')
                location = kwargs.get('location', '')
                return self.offline_manager.get_offline_poi_data(keyword, location)
            
            elif service_name == "location_service" and operation == "ip_location":
                ip = kwargs.get('ip', '')
                return self.offline_manager.get_offline_location_data(ip)
            
            elif service_name == "routing_service" and operation == "plan_route":
                # 返回简化的路径规划结果
                return self._get_simple_route_fallback(**kwargs)
            
            return None
            
        except Exception as e:
            logger.error(f"获取降级数据失败: {e}")
            return None
    
    def save_successful_data(self, service_name: str, operation: str, data: Any, **kwargs):
        """保存成功的数据用于离线使用"""
        try:
            if service_name == "poi_service" and operation == "search":
                keyword = kwargs.get('keyword', '')
                location = kwargs.get('location', '')
                if isinstance(data, list):
                    self.offline_manager.save_poi_data(keyword, location, data)
            
            elif service_name == "location_service" and operation == "ip_location":
                ip = kwargs.get('ip', '')
                if isinstance(data, dict):
                    self.offline_manager.save_location_data(ip, data)
                    
        except Exception as e:
            logger.error(f"保存成功数据失败: {e}")
    
    def _check_service_degradation(self, service: ServiceHealth):
        """检查服务是否需要降级"""
        try:
            # 基于错误数量降级
            if service.error_count >= self.degradation_config['offline_threshold']:
                if service.status != ServiceStatus.OFFLINE:
                    service.status = ServiceStatus.OFFLINE
                    logger.warning(f"🔴 服务 {service.service_name} 已离线")
            
            elif service.error_count >= self.degradation_config['error_threshold']:
                success_rate = service.get_success_rate()
                if success_rate < self.degradation_config['success_rate_threshold']:
                    if service.status == ServiceStatus.HEALTHY:
                        service.status = ServiceStatus.DEGRADED
                        service.degraded_since = time.time()
                        logger.warning(f"🟡 服务 {service.service_name} 已降级")
                        
        except Exception as e:
            logger.error(f"检查服务降级失败: {e}")
    
    def _check_service_recovery(self, service: ServiceHealth):
        """检查服务是否可以恢复"""
        try:
            if service.status == ServiceStatus.HEALTHY:
                return
            
            success_rate = service.get_success_rate()
            recent_success_count = 0
            
            # 检查最近的成功次数
            current_time = time.time()
            if current_time - service.last_success < 300:  # 5分钟内有成功
                recent_success_count = service.success_count
            
            # 恢复条件
            if (success_rate >= self.degradation_config['success_rate_threshold'] and 
                recent_success_count >= self.degradation_config['recovery_threshold']):
                
                service.status = ServiceStatus.HEALTHY
                service.degraded_since = 0.0
                logger.info(f"✅ 服务 {service.service_name} 已恢复")
                
        except Exception as e:
            logger.error(f"检查服务恢复失败: {e}")
    
    def _get_simple_route_fallback(self, **kwargs) -> Dict[str, Any]:
        """获取简化的路径规划降级数据"""
        origin = kwargs.get('origin', '')
        destination = kwargs.get('destination', '')
        
        return {
            'status': '1',
            'route': {
                'paths': [{
                    'distance': '未知',
                    'duration': '未知',
                    'steps': [{
                        'instruction': f'从{origin}前往{destination}的详细路线暂时无法获取，请使用其他导航工具或寻求帮助',
                        'distance': '0',
                        'duration': '0',
                        'road': '未知道路',
                        'action': '直行'
                    }]
                }]
            }
        }
    
    def _start_health_monitor(self):
        """启动健康监控"""
        def health_check():
            while True:
                try:
                    time.sleep(self.degradation_config['health_check_interval'])
                    self._perform_health_check()
                except Exception as e:
                    logger.error(f"健康检查失败: {e}")
        
        health_thread = threading.Thread(target=health_check, daemon=True)
        health_thread.start()
    
    def _perform_health_check(self):
        """执行健康检查"""
        with self.lock:
            current_time = time.time()
            
            for service in self.services.values():
                # 如果长时间没有活动，重置计数器
                if current_time - service.last_success > 3600:  # 1小时
                    service.error_count = max(0, service.error_count - 1)
                    service.success_count = max(0, service.success_count - 1)
                
                # 检查是否可以从降级状态恢复
                if service.status == ServiceStatus.DEGRADED:
                    if current_time - service.degraded_since > 1800:  # 30分钟后尝试恢复
                        service.status = ServiceStatus.HEALTHY
                        logger.info(f"✅ 服务 {service.service_name} 自动恢复")
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        with self.lock:
            report = {
                'services': {},
                'overall_health': 'healthy',
                'total_errors': len(self.error_history),
                'recent_errors': 0
            }
            
            degraded_count = 0
            offline_count = 0
            current_time = time.time()
            
            for service_name, service in self.services.items():
                report['services'][service_name] = {
                    'status': service.status.value,
                    'success_rate': service.get_success_rate(),
                    'availability': service.get_availability(),
                    'error_count': service.error_count,
                    'success_count': service.success_count
                }
                
                if service.status == ServiceStatus.DEGRADED:
                    degraded_count += 1
                elif service.status == ServiceStatus.OFFLINE:
                    offline_count += 1
            
            # 计算最近错误数
            report['recent_errors'] = sum(
                1 for error in self.error_history 
                if current_time - error.timestamp < 3600
            )
            
            # 确定整体健康状态
            if offline_count > 0:
                report['overall_health'] = 'critical'
            elif degraded_count > 0:
                report['overall_health'] = 'degraded'
            
            return report


# 全局降级管理器实例
_global_fallback_manager = None

def get_fallback_manager() -> FallbackManager:
    """获取全局降级管理器实例"""
    global _global_fallback_manager
    if _global_fallback_manager is None:
        _global_fallback_manager = FallbackManager()
    return _global_fallback_manager
