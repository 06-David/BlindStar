/**
 * BlindStar导航主界面
 * 盲人友好的导航界面设计
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Vibration,
  ScrollView,
  Dimensions,
} from 'react-native';

// 导入服务
import LocationService from '../services/LocationService';
import VoiceService from '../services/VoiceService';
import ApiService from '../services/ApiService';
import Config from '../config/Config';

const { width, height } = Dimensions.get('window');

const NavigationScreen = () => {
  const [currentLocation, setCurrentLocation] = useState(null);
  const [isNavigating, setIsNavigating] = useState(false);
  const [navigationSession, setNavigationSession] = useState(null);
  const [nearbyPOIs, setNearbyPOIs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    initializeScreen();
    
    return () => {
      cleanup();
    };
  }, []);

  /**
   * 初始化界面
   */
  const initializeScreen = async () => {
    try {
      // 开始位置跟踪
      await LocationService.startTracking();
      
      // 添加位置更新回调
      LocationService.addLocationCallback(handleLocationUpdate);
      
      // 获取当前位置
      const location = LocationService.getCurrentLocation();
      if (location) {
        setCurrentLocation(location);
        await searchNearbyPOIs(location);
      }
      
      // 播报界面说明
      await VoiceService.speak(
        '导航界面已加载，您可以搜索附近设施或开始导航',
        Config.VOICE.PRIORITY.INFORMATION
      );
      
    } catch (error) {
      console.error('❌ 界面初始化失败:', error);
      Alert.alert('错误', '界面初始化失败: ' + error.message);
    }
  };

  /**
   * 处理位置更新
   */
  const handleLocationUpdate = async (location) => {
    setCurrentLocation(location);
    
    // 如果正在导航，更新导航状态
    if (isNavigating && navigationSession) {
      await updateNavigation(location);
    }
  };

  /**
   * 搜索附近POI
   */
  const searchNearbyPOIs = async (location) => {
    try {
      setIsLoading(true);
      
      const result = await ApiService.searchPOI({
        keywords: '银行',
        location: location.formattedCoords,
        radius: 1000,
        pageSize: 5
      });
      
      if (result.success && result.data.pois) {
        setNearbyPOIs(result.data.pois);
        
        // 播报搜索结果
        const count = result.data.pois.length;
        await VoiceService.speak(
          `找到${count}个附近的银行`,
          Config.VOICE.PRIORITY.INFORMATION
        );
      }
      
    } catch (error) {
      console.error('❌ 搜索POI失败:', error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * 开始导航到POI
   */
  const startNavigationToPOI = async (poi) => {
    try {
      if (!currentLocation) {
        Alert.alert('错误', '无法获取当前位置');
        return;
      }

      setIsLoading(true);
      
      // 播报开始导航
      await VoiceService.speak(
        `开始导航到${poi.name}，距离${Math.round(poi.distance)}米`,
        Config.VOICE.PRIORITY.NAVIGATION
      );
      
      // 震动反馈
      Vibration.vibrate(Config.ACCESSIBILITY.VIBRATION_PATTERNS.SHORT);
      
      // 规划路径
      const routeResult = await ApiService.planRoute({
        origin: currentLocation.formattedCoords,
        destination: poi.location.formattedCoords,
        strategy: 'SAFE_PRIORITY'
      });
      
      if (!routeResult.success) {
        throw new Error('路径规划失败');
      }
      
      // 开始导航
      const navResult = await ApiService.startNavigation({
        route: routeResult.data,
        preferences: {
          voice: true,
          vibration: true,
          safetyFirst: true
        }
      });
      
      if (navResult.success) {
        setNavigationSession(navResult.data);
        setIsNavigating(true);
        
        // 播报路径信息
        const route = routeResult.data;
        await VoiceService.speak(
          `路径规划完成，总距离${Math.round(route.distance)}米，预计${Math.round(route.duration/60)}分钟`,
          Config.VOICE.PRIORITY.NAVIGATION
        );
      }
      
    } catch (error) {
      console.error('❌ 开始导航失败:', error);
      Alert.alert('导航失败', error.message);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * 更新导航状态
   */
  const updateNavigation = async (location) => {
    try {
      if (!navigationSession) return;
      
      const result = await ApiService.updateNavigation({
        sessionId: navigationSession.id,
        currentLocation: [location.longitude, location.latitude]
      });
      
      if (result.success) {
        const data = result.data;
        
        // 处理导航指令
        if (data.current_instruction) {
          const instruction = data.current_instruction;
          await VoiceService.speak(
            instruction.audio_text,
            Config.VOICE.PRIORITY.NAVIGATION
          );
          
          // 震动提醒
          if (instruction.priority <= 2) {
            Vibration.vibrate(Config.ACCESSIBILITY.VIBRATION_PATTERNS.DOUBLE);
          }
        }
        
        // 检查是否需要重新规划
        if (data.status === 'rerouted') {
          await VoiceService.speak(
            '检测到路径偏离，正在重新规划路径',
            Config.VOICE.PRIORITY.NAVIGATION
          );
        }
      }
      
    } catch (error) {
      console.error('❌ 更新导航失败:', error);
    }
  };

  /**
   * 停止导航
   */
  const stopNavigation = async () => {
    try {
      if (navigationSession) {
        await ApiService.stopNavigation({
          sessionId: navigationSession.id
        });
      }
      
      setIsNavigating(false);
      setNavigationSession(null);
      
      await VoiceService.speak(
        '导航已停止',
        Config.VOICE.PRIORITY.INFORMATION
      );
      
      // 震动反馈
      Vibration.vibrate(Config.ACCESSIBILITY.VIBRATION_PATTERNS.LONG);
      
    } catch (error) {
      console.error('❌ 停止导航失败:', error);
    }
  };

  /**
   * 播报当前位置
   */
  const speakCurrentLocation = async () => {
    if (currentLocation) {
      try {
        const result = await ApiService.reverseGeocode({
          location: currentLocation.formattedCoords
        });
        
        if (result.success && result.data.address) {
          await VoiceService.speak(
            `您当前位置：${result.data.address}`,
            Config.VOICE.PRIORITY.INFORMATION
          );
        } else {
          await VoiceService.speak(
            `您当前坐标：经度${currentLocation.longitude.toFixed(6)}，纬度${currentLocation.latitude.toFixed(6)}`,
            Config.VOICE.PRIORITY.INFORMATION
          );
        }
      } catch (error) {
        await VoiceService.speak(
          `您当前坐标：经度${currentLocation.longitude.toFixed(6)}，纬度${currentLocation.latitude.toFixed(6)}`,
          Config.VOICE.PRIORITY.INFORMATION
        );
      }
    } else {
      await VoiceService.speak(
        '正在获取位置信息，请稍候',
        Config.VOICE.PRIORITY.INFORMATION
      );
    }
  };

  /**
   * 清理资源
   */
  const cleanup = () => {
    LocationService.removeLocationCallback(handleLocationUpdate);
    if (isNavigating) {
      stopNavigation();
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* 当前位置信息 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>当前位置</Text>
        <TouchableOpacity
          style={styles.locationButton}
          onPress={speakCurrentLocation}
          accessible={true}
          accessibilityLabel="播报当前位置"
          accessibilityHint="点击播报您的当前位置信息"
        >
          <Text style={styles.locationText}>
            {currentLocation 
              ? `${currentLocation.latitude.toFixed(6)}, ${currentLocation.longitude.toFixed(6)}`
              : '正在获取位置...'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* 导航控制 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>导航控制</Text>
        {isNavigating ? (
          <TouchableOpacity
            style={[styles.button, styles.stopButton]}
            onPress={stopNavigation}
            accessible={true}
            accessibilityLabel="停止导航"
          >
            <Text style={styles.buttonText}>停止导航</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.button, styles.searchButton]}
            onPress={() => currentLocation && searchNearbyPOIs(currentLocation)}
            disabled={isLoading}
            accessible={true}
            accessibilityLabel="搜索附近银行"
          >
            <Text style={styles.buttonText}>
              {isLoading ? '搜索中...' : '搜索附近银行'}
            </Text>
          </TouchableOpacity>
        )}
      </View>

      {/* 附近POI列表 */}
      {nearbyPOIs.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>附近银行</Text>
          {nearbyPOIs.map((poi, index) => (
            <TouchableOpacity
              key={index}
              style={styles.poiItem}
              onPress={() => startNavigationToPOI(poi)}
              disabled={isNavigating || isLoading}
              accessible={true}
              accessibilityLabel={`导航到${poi.name}`}
              accessibilityHint={`距离${Math.round(poi.distance)}米`}
            >
              <Text style={styles.poiName}>{poi.name}</Text>
              <Text style={styles.poiDistance}>
                {Math.round(poi.distance)}米
              </Text>
              <Text style={styles.poiAddress}>{poi.address}</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  content: {
    padding: 20,
  },
  section: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 15,
  },
  locationButton: {
    backgroundColor: '#333333',
    padding: 15,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  locationText: {
    fontSize: 18,
    color: '#FFFFFF',
    textAlign: 'center',
  },
  button: {
    padding: 20,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 2,
  },
  searchButton: {
    backgroundColor: '#0066CC',
    borderColor: '#FFFFFF',
  },
  stopButton: {
    backgroundColor: '#CC0000',
    borderColor: '#FFFFFF',
  },
  buttonText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  poiItem: {
    backgroundColor: '#333333',
    padding: 15,
    marginBottom: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#666666',
  },
  poiName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 5,
  },
  poiDistance: {
    fontSize: 16,
    color: '#FFFF00',
    marginBottom: 5,
  },
  poiAddress: {
    fontSize: 14,
    color: '#CCCCCC',
  },
});

export default NavigationScreen;
