#!/usr/bin/env python3
"""
智能缓存管理器
提供多层缓存、智能过期、压缩存储等功能
"""

import time
import json
import hashlib
import logging
import threading
import pickle
import gzip
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import OrderedDict
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = 0.0
    size: int = 0
    compressed: bool = False
    
    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.timestamp > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_access = time.time()


class MemoryCache:
    """内存缓存"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.current_memory = 0
        
        logger.info(f"✅ 内存缓存初始化完成，最大条目数: {max_size}，最大内存: {max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.update_access()
                    # 移到末尾（LRU）
                    self.cache.move_to_end(key)
                    return entry.data
                else:
                    # 过期删除
                    self._remove_entry(key)
        return None
    
    def set(self, key: str, data: Any, ttl: int = 300) -> bool:
        """设置缓存数据"""
        with self.lock:
            try:
                # 计算数据大小
                data_size = len(pickle.dumps(data))
                
                # 检查是否需要压缩
                compressed = False
                if data_size > 1024:  # 大于1KB的数据进行压缩
                    compressed_data = gzip.compress(pickle.dumps(data))
                    if len(compressed_data) < data_size * 0.8:  # 压缩率超过20%才使用
                        data = compressed_data
                        data_size = len(compressed_data)
                        compressed = True
                
                # 检查内存限制
                if data_size > self.max_memory_bytes:
                    logger.warning(f"数据过大，无法缓存: {data_size} bytes")
                    return False
                
                # 清理空间
                while (len(self.cache) >= self.max_size or 
                       self.current_memory + data_size > self.max_memory_bytes):
                    if not self.cache:
                        break
                    # 删除最久未使用的条目
                    oldest_key = next(iter(self.cache))
                    self._remove_entry(oldest_key)
                
                # 添加新条目
                entry = CacheEntry(
                    key=key,
                    data=data,
                    timestamp=time.time(),
                    ttl=ttl,
                    size=data_size,
                    compressed=compressed
                )
                
                self.cache[key] = entry
                self.current_memory += data_size
                
                return True
                
            except Exception as e:
                logger.error(f"设置内存缓存失败: {e}")
                return False
    
    def _remove_entry(self, key: str):
        """删除缓存条目"""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size
            del self.cache[key]
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            return {
                'total_entries': len(self.cache),
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'memory_usage_percent': (self.current_memory / self.max_memory_bytes) * 100,
                'hit_rate': self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self) -> float:
        """计算缓存命中率"""
        if not self.cache:
            return 0.0
        
        total_access = sum(entry.access_count for entry in self.cache.values())
        if total_access == 0:
            return 0.0
        
        # 简化的命中率计算
        return min(100.0, (len(self.cache) / total_access) * 100)


class DiskCache:
    """磁盘缓存"""
    
    def __init__(self, cache_dir: str = "cache/disk", max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.db_path = self.cache_dir / "cache_index.db"
        self.lock = threading.RLock()
        
        self._init_database()
        logger.info(f"✅ 磁盘缓存初始化完成，缓存目录: {cache_dir}，最大大小: {max_size_mb}MB")
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        filename TEXT,
                        timestamp REAL,
                        ttl INTEGER,
                        size INTEGER,
                        access_count INTEGER DEFAULT 0,
                        last_access REAL
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"初始化缓存数据库失败: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        'SELECT filename, timestamp, ttl FROM cache_entries WHERE key = ?',
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        filename, timestamp, ttl = row
                        
                        # 检查是否过期
                        if time.time() - timestamp > ttl:
                            self._remove_entry(key)
                            return None
                        
                        # 读取文件
                        cache_file = self.cache_dir / filename
                        if cache_file.exists():
                            with open(cache_file, 'rb') as f:
                                data = pickle.load(f)
                            
                            # 更新访问信息
                            conn.execute(
                                'UPDATE cache_entries SET access_count = access_count + 1, last_access = ? WHERE key = ?',
                                (time.time(), key)
                            )
                            conn.commit()
                            
                            return data
                        else:
                            # 文件不存在，删除索引
                            self._remove_entry(key)
                
                return None
                
            except Exception as e:
                logger.error(f"获取磁盘缓存失败: {e}")
                return None
    
    def set(self, key: str, data: Any, ttl: int = 3600) -> bool:
        """设置缓存数据"""
        with self.lock:
            try:
                # 生成文件名
                filename = hashlib.md5(key.encode()).hexdigest() + '.cache'
                cache_file = self.cache_dir / filename
                
                # 序列化数据
                serialized_data = pickle.dumps(data)
                data_size = len(serialized_data)
                
                # 检查磁盘空间
                self._cleanup_if_needed(data_size)
                
                # 写入文件
                with open(cache_file, 'wb') as f:
                    f.write(serialized_data)
                
                # 更新数据库
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute(
                        'INSERT OR REPLACE INTO cache_entries (key, filename, timestamp, ttl, size, last_access) VALUES (?, ?, ?, ?, ?, ?)',
                        (key, filename, time.time(), ttl, data_size, time.time())
                    )
                    conn.commit()
                
                return True
                
            except Exception as e:
                logger.error(f"设置磁盘缓存失败: {e}")
                return False
    
    def _cleanup_if_needed(self, new_data_size: int):
        """根据需要清理缓存"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 获取当前总大小
                cursor = conn.execute('SELECT SUM(size) FROM cache_entries')
                current_size = cursor.fetchone()[0] or 0
                
                # 如果需要清理
                if current_size + new_data_size > self.max_size_bytes:
                    # 删除最久未访问的条目
                    cursor = conn.execute(
                        'SELECT key, filename FROM cache_entries ORDER BY last_access ASC'
                    )
                    
                    for key, filename in cursor.fetchall():
                        self._remove_entry(key)
                        
                        # 重新检查大小
                        cursor2 = conn.execute('SELECT SUM(size) FROM cache_entries')
                        current_size = cursor2.fetchone()[0] or 0
                        
                        if current_size + new_data_size <= self.max_size_bytes:
                            break
                            
        except Exception as e:
            logger.error(f"清理磁盘缓存失败: {e}")
    
    def _remove_entry(self, key: str):
        """删除缓存条目"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('SELECT filename FROM cache_entries WHERE key = ?', (key,))
                row = cursor.fetchone()
                
                if row:
                    filename = row[0]
                    cache_file = self.cache_dir / filename
                    
                    # 删除文件
                    if cache_file.exists():
                        cache_file.unlink()
                    
                    # 删除数据库记录
                    conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"删除磁盘缓存条目失败: {e}")
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            try:
                # 删除所有缓存文件
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                
                # 清空数据库
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute('DELETE FROM cache_entries')
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"清空磁盘缓存失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('SELECT COUNT(*), SUM(size), SUM(access_count) FROM cache_entries')
                count, total_size, total_access = cursor.fetchone()
                
                return {
                    'total_entries': count or 0,
                    'disk_usage_mb': (total_size or 0) / (1024 * 1024),
                    'disk_usage_percent': ((total_size or 0) / self.max_size_bytes) * 100,
                    'total_access': total_access or 0
                }
                
        except Exception as e:
            logger.error(f"获取磁盘缓存统计失败: {e}")
            return {'total_entries': 0, 'disk_usage_mb': 0, 'disk_usage_percent': 0, 'total_access': 0}


class SmartCacheManager:
    """智能缓存管理器"""
    
    def __init__(self, 
                 memory_cache_size: int = 1000,
                 memory_cache_mb: int = 100,
                 disk_cache_mb: int = 500,
                 cache_dir: str = "cache"):
        """
        初始化智能缓存管理器
        
        Args:
            memory_cache_size: 内存缓存最大条目数
            memory_cache_mb: 内存缓存最大大小(MB)
            disk_cache_mb: 磁盘缓存最大大小(MB)
            cache_dir: 缓存目录
        """
        self.memory_cache = MemoryCache(memory_cache_size, memory_cache_mb)
        self.disk_cache = DiskCache(f"{cache_dir}/disk", disk_cache_mb)
        
        # 缓存策略配置
        self.memory_ttl_threshold = 300  # 5分钟以下的数据优先使用内存缓存
        self.disk_ttl_threshold = 3600   # 1小时以上的数据使用磁盘缓存
        
        logger.info("✅ 智能缓存管理器初始化完成")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据（智能多层查找）"""
        # 先查内存缓存
        data = self.memory_cache.get(key)
        if data is not None:
            return data
        
        # 再查磁盘缓存
        data = self.disk_cache.get(key)
        if data is not None:
            # 热数据提升到内存缓存
            self.memory_cache.set(key, data, self.memory_ttl_threshold)
            return data
        
        return None
    
    def set(self, key: str, data: Any, ttl: int = 300) -> bool:
        """设置缓存数据（智能存储策略）"""
        success = False
        
        # 根据TTL选择存储策略
        if ttl <= self.memory_ttl_threshold:
            # 短期数据存储到内存
            success = self.memory_cache.set(key, data, ttl)
        elif ttl >= self.disk_ttl_threshold:
            # 长期数据存储到磁盘
            success = self.disk_cache.set(key, data, ttl)
        else:
            # 中期数据同时存储到内存和磁盘
            memory_success = self.memory_cache.set(key, data, ttl)
            disk_success = self.disk_cache.set(key, data, ttl)
            success = memory_success or disk_success
        
        return success
    
    def delete(self, key: str):
        """删除缓存数据"""
        # 从两个缓存中都删除
        self.memory_cache._remove_entry(key)
        self.disk_cache._remove_entry(key)
    
    def clear(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        self.disk_cache.clear()
        logger.info("🗑️ 所有缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        return {
            'memory_cache': memory_stats,
            'disk_cache': disk_stats,
            'total_entries': memory_stats['total_entries'] + disk_stats['total_entries'],
            'total_memory_mb': memory_stats['memory_usage_mb'] + disk_stats['disk_usage_mb']
        }
    
    def optimize(self):
        """优化缓存（清理过期数据、整理碎片等）"""
        try:
            # 清理过期的内存缓存
            with self.memory_cache.lock:
                expired_keys = []
                for key, entry in self.memory_cache.cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self.memory_cache._remove_entry(key)
            
            # 清理过期的磁盘缓存
            with self.disk_cache.lock:
                try:
                    with sqlite3.connect(str(self.disk_cache.db_path)) as conn:
                        cursor = conn.execute(
                            'SELECT key FROM cache_entries WHERE ? - timestamp > ttl',
                            (time.time(),)
                        )
                        expired_keys = [row[0] for row in cursor.fetchall()]
                        
                        for key in expired_keys:
                            self.disk_cache._remove_entry(key)
                            
                except Exception as e:
                    logger.error(f"清理过期磁盘缓存失败: {e}")
            
            logger.info("✅ 缓存优化完成")
            
        except Exception as e:
            logger.error(f"缓存优化失败: {e}")


# 全局缓存管理器实例
_global_cache_manager = None

def get_cache_manager() -> SmartCacheManager:
    """获取全局缓存管理器实例"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = SmartCacheManager()
    return _global_cache_manager
