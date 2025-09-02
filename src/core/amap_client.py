#!/usr/bin/env python3
"""
高德地图API客户端
统一的API请求管理，包含缓存、限流、错误处理等功能
"""

import time
import json
import hashlib
import logging
import requests
import threading
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .cache_manager import get_cache_manager
    from ..config.amap_config import get_amap_config, APIEndpoint
except ImportError:
    try:
        # 尝试绝对导入
        from src.config.amap_config import get_amap_config, APIEndpoint
        from src.core.cache_manager import get_cache_manager
    except ImportError:
        # 如果无法导入，使用简化版本
        logger = logging.getLogger(__name__)
        logger.warning("无法导入完整配置，使用简化版本")
        get_amap_config = None
        APIEndpoint = None
        get_cache_manager = None

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class APICache:
    """API响应缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/amap"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        self._lock = threading.RLock()
        
        # 内存缓存限制
        self.max_memory_cache_size = 1000
        self.memory_cache_ttl = 300  # 5分钟
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 排序参数以确保一致性
        sorted_params = sorted(params.items())
        cache_string = f"{endpoint}:{json.dumps(sorted_params, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict[str, Any], ttl: int = 300) -> Optional[Dict[str, Any]]:
        """获取缓存数据"""
        cache_key = self._get_cache_key(endpoint, params)
        current_time = time.time()
        
        with self._lock:
            # 先检查内存缓存
            if cache_key in self._memory_cache:
                cache_data = self._memory_cache[cache_key]
                if current_time - cache_data['timestamp'] < ttl:
                    logger.debug(f"🎯 内存缓存命中: {endpoint}")
                    return cache_data['data']
                else:
                    # 过期删除
                    del self._memory_cache[cache_key]
            
            # 检查文件缓存
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if current_time - cache_data['timestamp'] < ttl:
                        logger.debug(f"💾 文件缓存命中: {endpoint}")
                        # 加载到内存缓存
                        self._add_to_memory_cache(cache_key, cache_data['data'])
                        return cache_data['data']
                    else:
                        # 过期删除
                        cache_file.unlink()
                        
                except Exception as e:
                    logger.warning(f"读取缓存文件失败: {e}")
                    cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, endpoint: str, params: Dict[str, Any], data: Dict[str, Any], ttl: int = 300):
        """设置缓存数据"""
        cache_key = self._get_cache_key(endpoint, params)
        current_time = time.time()
        
        cache_data = {
            'timestamp': current_time,
            'data': data,
            'ttl': ttl
        }
        
        with self._lock:
            # 添加到内存缓存
            self._add_to_memory_cache(cache_key, data)
            
            # 保存到文件缓存（异步）
            def save_to_file():
                try:
                    cache_file = self.cache_dir / f"{cache_key}.json"
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"保存缓存文件失败: {e}")
            
            # 在后台线程中保存
            threading.Thread(target=save_to_file, daemon=True).start()
    
    def _add_to_memory_cache(self, cache_key: str, data: Dict[str, Any]):
        """添加到内存缓存"""
        # 检查内存缓存大小限制
        if len(self._memory_cache) >= self.max_memory_cache_size:
            # 删除最旧的缓存项
            oldest_key = min(self._memory_cache.keys(), 
                           key=lambda k: self._memory_cache[k]['timestamp'])
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = {
            'timestamp': time.time(),
            'data': data
        }
    
    def clear(self, endpoint: Optional[str] = None):
        """清除缓存"""
        with self._lock:
            if endpoint:
                # 清除特定端点的缓存
                keys_to_remove = [k for k in self._memory_cache.keys() 
                                if k.startswith(endpoint)]
                for key in keys_to_remove:
                    del self._memory_cache[key]
            else:
                # 清除所有缓存
                self._memory_cache.clear()
                
                # 清除文件缓存
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()


class AmapAPIClient:
    """高德地图API客户端"""
    
    def __init__(self, api_key: str = "请填写你的key"):
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3"
        self._session = None
        self._setup_session()

        # 使用智能缓存管理器
        if get_cache_manager:
            try:
                self.cache_manager = get_cache_manager()
                logger.info("✅ 使用智能缓存管理器")
            except Exception as e:
                logger.warning(f"缓存管理器初始化失败: {e}")
                self.cache_manager = None
                self.cache = APICache()
        else:
            self.cache_manager = None
            self.cache = APICache()

        # 如果有完整配置，使用配置管理器
        if get_amap_config:
            try:
                self.config = get_amap_config()
                self.api_key = self.config.api_key
                logger.info("✅ 使用完整配置管理器")
            except Exception as e:
                logger.warning(f"配置管理器初始化失败，使用简化版本: {e}")
                self.config = None
        else:
            self.config = None

        logger.info("✅ 高德地图API客户端初始化完成")
    
    def _setup_session(self):
        """设置HTTP会话"""
        self._session = requests.Session()
        
        # 设置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # 设置默认请求头
        self._session.headers.update({
            'User-Agent': 'BlindStar_POI/2.0 (Navigation Assistant)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
    
    def request(self, endpoint_name: str, url: str = None, **params) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        发起API请求
        
        Args:
            endpoint_name: 端点名称
            url: 请求URL（如果不提供则使用默认）
            **params: 请求参数
            
        Returns:
            (是否成功, 响应数据, 错误信息)
        """
        # 如果有配置管理器，使用配置管理器
        if self.config:
            endpoint = self.config.get_endpoint(endpoint_name)
            # 检查调用限制
            if self.config:
                can_request, reason = self.config.can_make_request(endpoint_name)
                if not can_request:
                    logger.warning(f"🚫 API调用被限制: {endpoint_name} - {reason}")
                    return False, None, f"API调用限制: {reason}"
            
            if endpoint:
                request_url = endpoint.url
                timeout = endpoint.timeout
                cache_ttl = endpoint.cache_ttl
            else:
                request_url = url or f"{self.base_url}/{endpoint_name}"
                timeout = 30
                cache_ttl = 300
        else:
            request_url = url or f"{self.base_url}/{endpoint_name}"
            timeout = 30
            cache_ttl = 300
        
        # 构建请求参数
        request_params = {
            'key': self.api_key,
            'output': 'json'
        }
        request_params.update(params)
        
        # 检查缓存
        cache_key = f"{endpoint_name}_{hashlib.md5(json.dumps(request_params, sort_keys=True).encode()).hexdigest()}"

        if self.cache_manager:
            cached_data = self.cache_manager.get(cache_key)
        else:
            cached_data = self.cache.get(endpoint_name, request_params, cache_ttl)

        if cached_data:
            if self.config:
                self.config.record_call(endpoint_name, True)
            return True, cached_data, "缓存命中"
        
        # 发起请求（增强版本，带重试和监控）
        return self._make_request_with_retry(endpoint_name, request_url, request_params, timeout, cache_key, cache_ttl)

    def _make_request_with_retry(self, endpoint_name: str, request_url: str,
                                request_params: dict, timeout: int,
                                cache_key: str, cache_ttl: int) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """带重试机制的API请求"""
        max_retries = 3
        retry_delays = [1, 2, 4]  # 指数退避

        for attempt in range(max_retries):
            try:
                logger.debug(f"🌐 发起API请求: {endpoint_name} (尝试 {attempt + 1}/{max_retries})")

                # 记录请求开始时间
                start_time = time.time()

                response = self._session.get(
                    url=request_url,
                    params=request_params,
                    timeout=timeout,
                    verify=False  # 禁用SSL验证
                )

                # 记录响应时间
                response_time = time.time() - start_time

                response.raise_for_status()
                data = response.json()

                # 检查API响应状态
                if data.get('status') == '1':
                    # 成功响应，缓存结果
                    if self.cache_manager:
                        self.cache_manager.set(cache_key, data, cache_ttl)
                    else:
                        self.cache.set(endpoint_name, request_params, data, cache_ttl)

                    if self.config:
                        self.config.record_call(endpoint_name, True)

                    logger.debug(f"✅ API请求成功: {endpoint_name} (响应时间: {response_time:.2f}s)")
                    return True, data, "请求成功"
                else:
                    # API返回错误
                    error_msg = data.get('info', '未知错误')
                    error_code = data.get('infocode', 'N/A')

                    # 检查是否是可重试的错误
                    if self._is_retryable_error(error_code):
                        if attempt < max_retries - 1:
                            delay = retry_delays[attempt]
                            logger.warning(f"⚠️ API错误可重试: {endpoint_name} - {error_msg} (错误码: {error_code})，{delay}秒后重试")
                            time.sleep(delay)
                            continue

                    logger.error(f"❌ API返回错误: {endpoint_name} - {error_msg} (错误码: {error_code})")
                    if self.config:
                        self.config.record_call(endpoint_name, False)
                    return False, None, f"API错误: {error_msg} (错误码: {error_code})"

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(f"⏰ 请求超时，{delay}秒后重试: {endpoint_name}")
                    time.sleep(delay)
                    continue
                else:
                    error_msg = f"请求超时: {endpoint_name}"
                    logger.error(f"⏰ {error_msg}")
                    if self.config:
                        self.config.record_call(endpoint_name, False)
                    return False, None, error_msg

            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(f"🌐 连接错误，{delay}秒后重试: {endpoint_name}")
                    time.sleep(delay)
                    continue
                else:
                    error_msg = f"网络连接失败: {endpoint_name}"
                    logger.error(f"🌐 {error_msg}")
                    if self.config:
                        self.config.record_call(endpoint_name, False)
                    return False, None, error_msg

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(f"🌐 网络请求异常，{delay}秒后重试: {endpoint_name} - {str(e)}")
                    time.sleep(delay)
                    continue
                else:
                    error_msg = f"网络请求失败: {endpoint_name} - {str(e)}"
                    logger.error(f"🌐 {error_msg}")
                    if self.config:
                        self.config.record_call(endpoint_name, False)
                    return False, None, error_msg

            except json.JSONDecodeError as e:
                error_msg = f"响应解析失败: {endpoint_name} - {str(e)}"
                logger.error(f"📄 {error_msg}")
                if self.config:
                    self.config.record_call(endpoint_name, False)
                return False, None, error_msg

            except Exception as e:
                error_msg = f"未知错误: {endpoint_name} - {str(e)}"
                logger.error(f"❓ {error_msg}")
                if self.config:
                    self.config.record_call(endpoint_name, False)
                return False, None, error_msg

        # 所有重试都失败
        error_msg = f"API请求失败，已重试{max_retries}次: {endpoint_name}"
        logger.error(f"❌ {error_msg}")
        if self.config:
            self.config.record_call(endpoint_name, False)
        return False, None, error_msg

    def _is_retryable_error(self, error_code: str) -> bool:
        """判断错误是否可重试"""
        # 高德地图API可重试的错误码
        retryable_codes = [
            '10001',  # key不正确或过期
            '10002',  # 没有权限使用相应的服务或者请求接口的路径拼写错误
            '10003',  # 访问已超出日访问量
            '10004',  # 单位时间内访问过于频繁
            '10005',  # IP白名单出错，发送请求的服务器IP不在IP白名单内
            '20001',  # 请求参数非法
            '20002',  # 缺少必填参数
            '20003',  # 请求协议非法
            '30001',  # 引擎返回数据异常
            '30002',  # 请求服务响应错误
            '30003'   # 规划点（包括起点、终点、途经点）不在中国陆地范围内
        ]

        # 网络相关错误通常可重试
        network_retryable_codes = ['10004', '30001', '30002']

        return error_code in network_retryable_codes

    def get_api_health_status(self) -> dict:
        """获取API健康状态"""
        try:
            if self.config:
                stats = self.config.get_api_stats("ip_location")
                return {
                    "status": "healthy" if stats.get("success_rate", 0) > 0.8 else "degraded",
                    "success_rate": stats.get("success_rate", 0),
                    "total_calls": stats.get("total_calls", 0),
                    "failed_calls": stats.get("failed_calls", 0),
                    "last_call_time": stats.get("last_call_time", "未知")
                }
            else:
                return {"status": "unknown", "message": "无配置管理器"}
        except Exception as e:
            logger.error(f"❌ 获取API健康状态失败: {e}")
            return {"status": "error", "message": str(e)}

    def test_api_connectivity(self) -> dict:
        """测试API连通性"""
        try:
            logger.info("🔍 测试高德地图API连通性...")

            # 测试IP定位API
            success, data, error = self.request("ip_location")

            if success:
                logger.info("✅ API连通性测试成功")
                return {
                    "status": "success",
                    "message": "API连通性正常",
                    "response_data": data
                }
            else:
                logger.warning(f"⚠️ API连通性测试失败: {error}")
                return {
                    "status": "failed",
                    "message": f"API连通性测试失败: {error}",
                    "error": error
                }

        except Exception as e:
            logger.error(f"❌ API连通性测试异常: {e}")
            return {
                "status": "error",
                "message": f"API连通性测试异常: {str(e)}",
                "error": str(e)
            }


# 全局客户端实例
_global_client = None

def get_amap_client() -> AmapAPIClient:
    """获取全局API客户端实例"""
    global _global_client
    if _global_client is None:
        _global_client = AmapAPIClient()
    return _global_client
