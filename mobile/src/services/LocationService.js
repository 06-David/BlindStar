/**
 * BlindStar定位服务
 * 处理GPS定位、位置跟踪和位置相关功能
 */

import Geolocation from '@react-native-community/geolocation';
import { PermissionsAndroid, Platform } from 'react-native';
import Config from '../config/Config';
import ApiService from './ApiService';

class LocationService {
  constructor() {
    this.watchId = null;
    this.currentLocation = null;
    this.lastUpdateTime = 0;
    this.locationCallbacks = new Set();
    this.isTracking = false;
    
    // 配置Geolocation
    if (Platform.OS === 'android') {
      Geolocation.setRNConfiguration({
        skipPermissionRequests: false,
        authorizationLevel: 'whenInUse',
      });
    }
  }

  /**
   * 请求位置权限
   */
  async requestLocationPermission() {
    try {
      if (Platform.OS === 'android') {
        const granted = await PermissionsAndroid.requestMultiple([
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
          PermissionsAndroid.PERMISSIONS.ACCESS_COARSE_LOCATION,
        ]);
        
        return (
          granted[PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION] === 'granted' ||
          granted[PermissionsAndroid.PERMISSIONS.ACCESS_COARSE_LOCATION] === 'granted'
        );
      }
      
      // iOS权限在Info.plist中配置，这里直接返回true
      return true;
    } catch (error) {
      console.error('❌ 位置权限请求失败:', error);
      return false;
    }
  }

  /**
   * 获取当前位置（单次）
   */
  async getCurrentPosition() {
    return new Promise((resolve, reject) => {
      const options = Config.LOCATION.GPS_OPTIONS;
      
      Geolocation.getCurrentPosition(
        (position) => {
          const location = this._processPosition(position);
          this.currentLocation = location;
          console.log('📍 获取当前位置成功:', location);
          resolve(location);
        },
        (error) => {
          console.error('❌ 获取当前位置失败:', error);
          reject(this._handleLocationError(error));
        },
        options
      );
    });
  }

  /**
   * 开始位置跟踪
   */
  async startTracking() {
    try {
      // 检查权限
      const hasPermission = await this.requestLocationPermission();
      if (!hasPermission) {
        throw new Error('位置权限被拒绝');
      }

      if (this.isTracking) {
        console.log('⚠️ 位置跟踪已在运行');
        return;
      }

      const options = Config.LOCATION.GPS_OPTIONS;
      
      this.watchId = Geolocation.watchPosition(
        (position) => {
          const location = this._processPosition(position);
          this._updateLocation(location);
        },
        (error) => {
          console.error('❌ 位置跟踪错误:', error);
          this._notifyLocationError(this._handleLocationError(error));
        },
        options
      );

      this.isTracking = true;
      console.log('🎯 开始位置跟踪');
      
    } catch (error) {
      console.error('❌ 启动位置跟踪失败:', error);
      throw error;
    }
  }

  /**
   * 停止位置跟踪
   */
  stopTracking() {
    if (this.watchId !== null) {
      Geolocation.clearWatch(this.watchId);
      this.watchId = null;
      this.isTracking = false;
      console.log('🛑 停止位置跟踪');
    }
  }

  /**
   * 添加位置更新回调
   */
  addLocationCallback(callback) {
    this.locationCallbacks.add(callback);
    
    // 如果有当前位置，立即调用回调
    if (this.currentLocation) {
      callback(this.currentLocation);
    }
  }

  /**
   * 移除位置更新回调
   */
  removeLocationCallback(callback) {
    this.locationCallbacks.delete(callback);
  }

  /**
   * 获取当前位置（缓存）
   */
  getCurrentLocation() {
    return this.currentLocation;
  }

  /**
   * 计算两点间距离（米）
   */
  calculateDistance(lat1, lng1, lat2, lng2) {
    const R = 6371e3; // 地球半径（米）
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δφ = (lat2 - lat1) * Math.PI / 180;
    const Δλ = (lng2 - lng1) * Math.PI / 180;

    const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
  }

  /**
   * 计算方位角（度）
   */
  calculateBearing(lat1, lng1, lat2, lng2) {
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δλ = (lng2 - lng1) * Math.PI / 180;

    const y = Math.sin(Δλ) * Math.cos(φ2);
    const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);

    const θ = Math.atan2(y, x);
    return (θ * 180 / Math.PI + 360) % 360; // 转换为0-360度
  }

  /**
   * 获取方向描述
   */
  getDirectionDescription(bearing) {
    const directions = [
      '正北', '东北偏北', '东北', '东北偏东',
      '正东', '东南偏东', '东南', '东南偏南',
      '正南', '西南偏南', '西南', '西南偏西',
      '正西', '西北偏西', '西北', '西北偏北'
    ];
    
    const index = Math.round(bearing / 22.5) % 16;
    return directions[index];
  }

  /**
   * 地理编码（地址转坐标）
   */
  async geocodeAddress(address) {
    try {
      const result = await ApiService.geocodeAddress({ address });
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error.message);
      }
    } catch (error) {
      console.error('❌ 地理编码失败:', error);
      throw error;
    }
  }

  /**
   * 逆地理编码（坐标转地址）
   */
  async reverseGeocode(latitude, longitude) {
    try {
      const location = `${longitude},${latitude}`;
      const result = await ApiService.reverseGeocode({ location });
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error.message);
      }
    } catch (error) {
      console.error('❌ 逆地理编码失败:', error);
      throw error;
    }
  }

  /**
   * 处理位置数据
   */
  _processPosition(position) {
    const { latitude, longitude, accuracy, speed, heading } = position.coords;
    const timestamp = position.timestamp;
    
    return {
      latitude,
      longitude,
      accuracy,
      speed: speed || 0,
      heading: heading || 0,
      timestamp,
      formattedCoords: `${longitude},${latitude}`,
    };
  }

  /**
   * 更新位置
   */
  _updateLocation(newLocation) {
    const now = Date.now();
    const timeDiff = now - this.lastUpdateTime;
    
    // 检查是否需要更新
    if (this.currentLocation) {
      const distance = this.calculateDistance(
        this.currentLocation.latitude,
        this.currentLocation.longitude,
        newLocation.latitude,
        newLocation.longitude
      );
      
      // 距离和时间阈值检查
      if (distance < Config.LOCATION.UPDATE_THRESHOLD.DISTANCE && 
          timeDiff < Config.LOCATION.UPDATE_THRESHOLD.TIME) {
        return; // 不需要更新
      }
    }

    this.currentLocation = newLocation;
    this.lastUpdateTime = now;
    
    // 通知所有回调
    this.locationCallbacks.forEach(callback => {
      try {
        callback(newLocation);
      } catch (error) {
        console.error('❌ 位置回调执行失败:', error);
      }
    });
    
    console.log('📍 位置已更新:', newLocation.formattedCoords);
  }

  /**
   * 处理位置错误
   */
  _handleLocationError(error) {
    const errorMessages = {
      1: '位置权限被拒绝',
      2: '位置信息不可用',
      3: '位置请求超时',
    };
    
    return {
      code: error.code,
      message: errorMessages[error.code] || error.message || '位置获取失败',
      originalError: error,
    };
  }

  /**
   * 通知位置错误
   */
  _notifyLocationError(error) {
    this.locationCallbacks.forEach(callback => {
      try {
        if (callback.onError) {
          callback.onError(error);
        }
      } catch (err) {
        console.error('❌ 位置错误回调执行失败:', err);
      }
    });
  }

  /**
   * 清理资源
   */
  cleanup() {
    this.stopTracking();
    this.locationCallbacks.clear();
    this.currentLocation = null;
  }
}

// 导出单例
export default new LocationService();
