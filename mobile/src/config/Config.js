/**
 * BlindStar移动端配置文件
 */

const Config = {
  // API配置
  API_BASE_URL: __DEV__ 
    ? 'http://localhost:5000'  // 开发环境
    : 'https://api.blindstar.com',  // 生产环境
  
  API_TIMEOUT: 30000, // 30秒超时
  
  // 定位配置
  LOCATION: {
    // GPS定位选项
    GPS_OPTIONS: {
      enableHighAccuracy: true,
      timeout: 15000,
      maximumAge: 10000,
      distanceFilter: 5, // 5米更新一次
      interval: 1000, // 1秒检查一次
    },
    
    // 定位更新阈值
    UPDATE_THRESHOLD: {
      DISTANCE: 10, // 10米
      TIME: 5000, // 5秒
    },
  },

  // 导航配置
  NAVIGATION: {
    // 语音播报距离阈值
    VOICE_DISTANCE_THRESHOLDS: {
      FAR: 200,     // 200米开始提醒
      MEDIUM: 100,  // 100米重要提醒
      NEAR: 50,     // 50米最后提醒
      ARRIVED: 10,  // 10米内算到达
    },
    
    // 路径偏离检测
    OFF_ROUTE_THRESHOLD: 50, // 50米算偏离路径
    
    // 重新规划延迟
    REROUTE_DELAY: 3000, // 3秒后重新规划
  },

  // 语音配置
  VOICE: {
    // TTS配置
    TTS_OPTIONS: {
      language: 'zh-CN',
      rate: 0.8, // 语速
      pitch: 1.0, // 音调
      volume: 1.0, // 音量
    },
    
    // 语音识别配置
    STT_OPTIONS: {
      language: 'zh-CN',
      timeout: 10000,
      partialResults: true,
    },
    
    // 播报优先级
    PRIORITY: {
      EMERGENCY: 1,    // 紧急情况
      NAVIGATION: 2,   // 导航指令
      INFORMATION: 3,  // 一般信息
      DEBUG: 4,        // 调试信息
    },
  },

  // 无障碍配置
  ACCESSIBILITY: {
    // 震动模式
    VIBRATION_PATTERNS: {
      SHORT: [0, 200],           // 短震动
      LONG: [0, 500],            // 长震动
      DOUBLE: [0, 200, 100, 200], // 双震动
      ALERT: [0, 500, 200, 500, 200, 500], // 警告震动
    },
    
    // 字体大小
    FONT_SIZES: {
      SMALL: 16,
      MEDIUM: 20,
      LARGE: 24,
      EXTRA_LARGE: 28,
    },
    
    // 颜色对比度
    HIGH_CONTRAST: {
      BACKGROUND: '#000000',
      TEXT: '#FFFFFF',
      ACCENT: '#FFFF00',
      WARNING: '#FF0000',
    },
  },

  // 安全配置
  SAFETY: {
    // 紧急联系人
    EMERGENCY_CONTACTS: [
      // 将在运行时从用户设置中加载
    ],
    
    // 危险区域检测
    HAZARD_DETECTION: {
      ENABLED: true,
      SENSITIVITY: 'HIGH', // LOW, MEDIUM, HIGH
    },
    
    // 自动求助
    AUTO_HELP: {
      ENABLED: false,
      TIMEOUT: 300000, // 5分钟无响应自动求助
    },
  },

  // 缓存配置
  CACHE: {
    // 路径缓存
    ROUTE_CACHE_SIZE: 50,
    ROUTE_CACHE_TTL: 3600000, // 1小时
    
    // POI缓存
    POI_CACHE_SIZE: 200,
    POI_CACHE_TTL: 1800000, // 30分钟
    
    // 位置缓存
    LOCATION_CACHE_TTL: 60000, // 1分钟
  },

  // 调试配置
  DEBUG: {
    ENABLED: __DEV__,
    LOG_LEVEL: __DEV__ ? 'DEBUG' : 'ERROR',
    MOCK_GPS: false, // 是否模拟GPS数据
    MOCK_API: false, // 是否模拟API响应
  },

  // 应用信息
  APP: {
    NAME: 'BlindStar',
    VERSION: '1.0.0',
    BUILD: '1',
    AUTHOR: 'BlindStar Team',
    SUPPORT_EMAIL: 'support@blindstar.com',
    WEBSITE: 'https://blindstar.com',
  },

  // 权限配置
  PERMISSIONS: {
    REQUIRED: [
      'android.permission.ACCESS_FINE_LOCATION',
      'android.permission.ACCESS_COARSE_LOCATION',
      'android.permission.RECORD_AUDIO',
      'android.permission.VIBRATE',
    ],
    OPTIONAL: [
      'android.permission.CALL_PHONE',
      'android.permission.SEND_SMS',
      'android.permission.WAKE_LOCK',
    ],
  },

  // 地图配置
  MAP: {
    // 高德地图配置
    AMAP: {
      API_KEY: '请填写你的key',
      // 注意：在生产环境中应该使用环境变量
    },
    
    // 默认地图中心点
    DEFAULT_CENTER: {
      latitude: 39.99923,
      longitude: 116.397428,
    },
    
    // 地图缩放级别
    ZOOM_LEVELS: {
      CITY: 10,
      DISTRICT: 13,
      STREET: 16,
      BUILDING: 18,
    },
  },

  // 网络配置
  NETWORK: {
    // 重试配置
    RETRY: {
      MAX_ATTEMPTS: 3,
      DELAY: 1000, // 1秒
      BACKOFF: 2, // 指数退避
    },
    
    // 超时配置
    TIMEOUTS: {
      LOCATION: 15000,
      NAVIGATION: 30000,
      POI_SEARCH: 20000,
      ROUTE_PLAN: 25000,
    },
  },

  // 存储配置
  STORAGE: {
    // AsyncStorage键名
    KEYS: {
      USER_PREFERENCES: 'user_preferences',
      NAVIGATION_HISTORY: 'navigation_history',
      FAVORITE_PLACES: 'favorite_places',
      EMERGENCY_CONTACTS: 'emergency_contacts',
      CACHE_DATA: 'cache_data',
    },
    
    // 存储限制
    LIMITS: {
      HISTORY_SIZE: 100,
      FAVORITES_SIZE: 50,
      CACHE_SIZE: 10 * 1024 * 1024, // 10MB
    },
  },
};

export default Config;
