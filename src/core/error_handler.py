#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高德地图API错误处理器
基于官方错误码说明文档的完整错误处理机制
"""

import logging
from typing import Dict, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"           # 轻微错误，可以重试
    MEDIUM = "medium"     # 中等错误，需要调整参数
    HIGH = "high"         # 严重错误，需要人工干预
    CRITICAL = "critical" # 致命错误，服务不可用

class ErrorCategory(Enum):
    """错误类别"""
    AUTH = "auth"         # 认证相关
    QUOTA = "quota"       # 配额相关
    PARAM = "param"       # 参数相关
    SERVICE = "service"   # 服务相关
    NETWORK = "network"   # 网络相关
    UNKNOWN = "unknown"   # 未知错误

class AmapErrorHandler:
    """高德地图API错误处理器"""
    
    # 错误码映射表
    ERROR_CODE_MAP = {
        # 认证相关错误
        "10001": {
            "message": "API密钥不正确或已过期",
            "severity": ErrorSeverity.CRITICAL,
            "category": ErrorCategory.AUTH,
            "solution": "请检查API密钥是否正确，或联系管理员更新密钥",
            "retry": False
        },
        "10002": {
            "message": "没有权限使用相应的服务或请求路径错误",
            "severity": ErrorSeverity.HIGH,
            "category": ErrorCategory.AUTH,
            "solution": "请检查API权限配置或请求URL是否正确",
            "retry": False
        },
        "10009": {
            "message": "请求密钥与绑定平台不符",
            "severity": ErrorSeverity.HIGH,
            "category": ErrorCategory.AUTH,
            "solution": "请使用正确平台的API密钥",
            "retry": False
        },
        "10012": {
            "message": "权限不足，服务请求被拒绝",
            "severity": ErrorSeverity.HIGH,
            "category": ErrorCategory.AUTH,
            "solution": "请联系管理员开通相应服务权限",
            "retry": False
        },
        "10013": {
            "message": "API密钥已被删除",
            "severity": ErrorSeverity.CRITICAL,
            "category": ErrorCategory.AUTH,
            "solution": "请联系管理员重新申请API密钥",
            "retry": False
        },
        
        # 配额相关错误
        "10003": {
            "message": "日访问量已超限",
            "severity": ErrorSeverity.HIGH,
            "category": ErrorCategory.QUOTA,
            "solution": "请等待次日0:00自动解封，或联系管理员升级配额",
            "retry": False
        },
        "10004": {
            "message": "单位时间内访问过于频繁",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.QUOTA,
            "solution": "请降低请求频率，系统将在下一分钟自动解封",
            "retry": True
        },
        "10010": {
            "message": "IP访问超限",
            "severity": ErrorSeverity.HIGH,
            "category": ErrorCategory.QUOTA,
            "solution": "请联系管理员设置IP白名单或提交工单",
            "retry": False
        },
        "10014": {
            "message": "QPS已超出限制",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.QUOTA,
            "solution": "请降低请求频率",
            "retry": True
        },
        "40000": {
            "message": "服务余额已耗尽",
            "severity": ErrorSeverity.CRITICAL,
            "category": ErrorCategory.QUOTA,
            "solution": "请联系管理员充值服务余额",
            "retry": False
        },
        "40002": {
            "message": "购买的服务已到期",
            "severity": ErrorSeverity.CRITICAL,
            "category": ErrorCategory.QUOTA,
            "solution": "请联系管理员续费服务",
            "retry": False
        },
        
        # 参数相关错误
        "20000": {
            "message": "请求参数非法",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.PARAM,
            "solution": "请检查请求参数是否符合API规范",
            "retry": False
        },
        "20001": {
            "message": "缺少必填参数",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.PARAM,
            "solution": "请检查并补充必填参数",
            "retry": False
        },
        "20002": {
            "message": "请求协议非法",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.PARAM,
            "solution": "请检查HTTP请求方法是否正确",
            "retry": False
        },
        "20011": {
            "message": "查询坐标在海外但没有海外权限",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.PARAM,
            "solution": "请确保查询坐标在中国境内，或申请海外服务权限",
            "retry": False
        },
        "20012": {
            "message": "查询信息存在非法内容",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.PARAM,
            "solution": "请检查查询内容是否包含敏感词汇",
            "retry": False
        },
        
        # 路径规划相关错误
        "20800": {
            "message": "规划点不在中国陆地范围内",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.PARAM,
            "solution": "请确保起点和终点都在中国大陆范围内",
            "retry": False
        },
        "20801": {
            "message": "规划点附近搜不到路",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.PARAM,
            "solution": "请调整起点或终点位置到有道路的区域",
            "retry": False
        },
        "20802": {
            "message": "路线计算失败",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.SERVICE,
            "solution": "请尝试调整起点或终点位置",
            "retry": True
        },
        "20803": {
            "message": "起点终点距离过长",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.PARAM,
            "solution": "请缩短起点和终点之间的距离",
            "retry": False
        },
        
        # 服务相关错误
        "10016": {
            "message": "服务器负载过高",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.SERVICE,
            "solution": "请稍后重试",
            "retry": True
        },
        "10017": {
            "message": "所请求的资源不可用",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.SERVICE,
            "solution": "请稍后重试或联系技术支持",
            "retry": True
        },
        "20003": {
            "message": "其他未知错误",
            "severity": ErrorSeverity.MEDIUM,
            "category": ErrorCategory.UNKNOWN,
            "solution": "请重试或联系技术支持",
            "retry": True
        }
    }
    
    @classmethod
    def handle_error(cls, infocode: str, info: str = "") -> Tuple[str, bool, str]:
        """
        处理高德地图API错误
        
        Args:
            infocode: 错误码
            info: 错误信息
            
        Returns:
            Tuple[错误描述, 是否可重试, 解决方案]
        """
        try:
            if infocode == "10000":
                return "请求成功", False, ""
            
            error_info = cls.ERROR_CODE_MAP.get(infocode)
            
            if error_info:
                message = error_info["message"]
                can_retry = error_info["retry"]
                solution = error_info["solution"]
                severity = error_info["severity"]
                category = error_info["category"]
                
                # 记录错误日志
                log_level = cls._get_log_level(severity)
                logger.log(log_level, f"高德API错误 [{infocode}]: {message}")
                logger.log(log_level, f"错误类别: {category.value}, 严重程度: {severity.value}")
                logger.log(log_level, f"解决方案: {solution}")
                
                return message, can_retry, solution
            else:
                # 未知错误码
                message = f"未知错误码: {infocode}, 错误信息: {info}"
                logger.warning(message)
                return message, True, "请重试或联系技术支持"
                
        except Exception as e:
            logger.error(f"错误处理器异常: {e}")
            return f"错误处理失败: {str(e)}", False, "请联系技术支持"
    
    @classmethod
    def _get_log_level(cls, severity: ErrorSeverity) -> int:
        """根据错误严重程度获取日志级别"""
        level_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return level_map.get(severity, logging.WARNING)
    
    @classmethod
    def is_auth_error(cls, infocode: str) -> bool:
        """判断是否为认证相关错误"""
        error_info = cls.ERROR_CODE_MAP.get(infocode)
        return error_info and error_info["category"] == ErrorCategory.AUTH
    
    @classmethod
    def is_quota_error(cls, infocode: str) -> bool:
        """判断是否为配额相关错误"""
        error_info = cls.ERROR_CODE_MAP.get(infocode)
        return error_info and error_info["category"] == ErrorCategory.QUOTA
    
    @classmethod
    def is_param_error(cls, infocode: str) -> bool:
        """判断是否为参数相关错误"""
        error_info = cls.ERROR_CODE_MAP.get(infocode)
        return error_info and error_info["category"] == ErrorCategory.PARAM
    
    @classmethod
    def should_retry(cls, infocode: str) -> bool:
        """判断是否应该重试"""
        error_info = cls.ERROR_CODE_MAP.get(infocode)
        return error_info and error_info["retry"]
    
    @classmethod
    def get_user_friendly_message(cls, infocode: str, service_name: str = "导航服务") -> str:
        """获取用户友好的错误消息"""
        try:
            error_info = cls.ERROR_CODE_MAP.get(infocode)
            
            if not error_info:
                return f"{service_name}暂时不可用，请稍后重试"
            
            category = error_info["category"]
            severity = error_info["severity"]
            
            if category == ErrorCategory.AUTH:
                return f"{service_name}认证失败，请联系管理员"
            elif category == ErrorCategory.QUOTA:
                if severity == ErrorSeverity.CRITICAL:
                    return f"{service_name}已达使用上限，请联系管理员"
                else:
                    return f"{service_name}使用频率过高，请稍后重试"
            elif category == ErrorCategory.PARAM:
                return f"请求参数有误，请检查输入信息"
            elif category == ErrorCategory.SERVICE:
                return f"{service_name}暂时不可用，请稍后重试"
            else:
                return f"{service_name}出现未知错误，请重试或联系技术支持"
                
        except Exception as e:
            logger.error(f"生成用户友好消息失败: {e}")
            return f"{service_name}出现错误，请重试"

# 全局错误处理器实例
error_handler = AmapErrorHandler()
