/**
 * BlindStar Mobile App
 * 主应用组件
 */

import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  Alert,
  BackHandler,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

// 导入服务
import LocationService from './services/LocationService';
import VoiceService from './services/VoiceService';
import ApiService from './services/ApiService';

// 导入组件
import NavigationScreen from './components/NavigationScreen';
import Config from './config/Config';

const Stack = createStackNavigator();

const App = () => {
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState(null);

  useEffect(() => {
    initializeApp();
    
    // 处理Android返回键
    const backHandler = BackHandler.addEventListener(
      'hardwareBackPress',
      handleBackPress
    );

    return () => {
      backHandler.remove();
      cleanup();
    };
  }, []);

  /**
   * 初始化应用
   */
  const initializeApp = async () => {
    try {
      console.log('🚀 初始化BlindStar应用...');

      // 1. 检查API连接
      console.log('🌐 检查后端连接...');
      const healthCheck = await ApiService.healthCheck();
      if (!healthCheck.success) {
        throw new Error('无法连接到后端服务');
      }
      console.log('✅ 后端连接正常');

      // 2. 请求位置权限
      console.log('📍 请求位置权限...');
      const hasLocationPermission = await LocationService.requestLocationPermission();
      if (!hasLocationPermission) {
        throw new Error('位置权限被拒绝');
      }
      console.log('✅ 位置权限获取成功');

      // 3. 检查语音功能
      console.log('🎤 检查语音功能...');
      const isVoiceAvailable = await VoiceService.isVoiceAvailable();
      if (isVoiceAvailable) {
        console.log('✅ 语音功能可用');
      } else {
        console.log('⚠️ 语音功能不可用');
      }

      // 4. 获取初始位置
      console.log('📍 获取当前位置...');
      try {
        await LocationService.getCurrentPosition();
        console.log('✅ 位置获取成功');
      } catch (error) {
        console.log('⚠️ 位置获取失败，将使用默认位置');
      }

      // 5. 播报欢迎信息
      await VoiceService.speak(
        '欢迎使用BlindStar盲人导航系统，为您提供安全便捷的出行服务',
        Config.VOICE.PRIORITY.INFORMATION
      );

      setIsInitialized(true);
      console.log('🎉 BlindStar应用初始化完成');

    } catch (error) {
      console.error('❌ 应用初始化失败:', error);
      setInitError(error.message);
      
      // 显示错误对话框
      Alert.alert(
        '初始化失败',
        error.message,
        [
          { text: '重试', onPress: initializeApp },
          { text: '退出', onPress: () => BackHandler.exitApp() }
        ]
      );
    }
  };

  /**
   * 处理返回键
   */
  const handleBackPress = () => {
    Alert.alert(
      '退出应用',
      '确定要退出BlindStar导航吗？',
      [
        { text: '取消', style: 'cancel' },
        { text: '确定', onPress: () => BackHandler.exitApp() }
      ]
    );
    return true;
  };

  /**
   * 清理资源
   */
  const cleanup = () => {
    console.log('🧹 清理应用资源...');
    LocationService.cleanup();
    VoiceService.cleanup();
  };

  /**
   * 加载屏幕
   */
  const LoadingScreen = () => (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      <Text style={styles.title}>BlindStar</Text>
      <Text style={styles.subtitle}>盲人导航系统</Text>
      <Text style={styles.loading}>
        {initError ? `初始化失败: ${initError}` : '正在初始化...'}
      </Text>
    </View>
  );

  /**
   * 主导航栈
   */
  const MainNavigator = () => (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Navigation"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#000000',
          },
          headerTintColor: '#FFFFFF',
          headerTitleStyle: {
            fontWeight: 'bold',
            fontSize: 20,
          },
        }}
      >
        <Stack.Screen
          name="Navigation"
          component={NavigationScreen}
          options={{
            title: 'BlindStar导航',
            headerLeft: null, // 禁用返回按钮
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );

  // 根据初始化状态渲染不同界面
  if (!isInitialized) {
    return <LoadingScreen />;
  }

  return <MainNavigator />;
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 10,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 18,
    color: '#CCCCCC',
    marginBottom: 30,
    textAlign: 'center',
  },
  loading: {
    fontSize: 16,
    color: '#FFFFFF',
    textAlign: 'center',
    marginTop: 20,
  },
});

export default App;
