/**
 * BlindStar API服务
 * 与Python后端通信的核心服务
 */

import axios from 'axios';
import Config from '../config/Config';

class ApiService {
  constructor() {
    this.baseURL = Config.API_BASE_URL;
    this.timeout = Config.API_TIMEOUT;
    
    // 创建axios实例
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        console.log(`🌐 API请求: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ API请求错误:', error);
        return Promise.reject(error);
      }
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => {
        console.log(`✅ API响应: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('❌ API响应错误:', error.response?.status, error.message);
        return Promise.reject(this._handleError(error));
      }
    );
  }

  /**
   * 错误处理
   */
  _handleError(error) {
    if (error.response) {
      // 服务器响应错误
      return {
        type: 'SERVER_ERROR',
        status: error.response.status,
        message: error.response.data?.message || '服务器错误',
        data: error.response.data
      };
    } else if (error.request) {
      // 网络错误
      return {
        type: 'NETWORK_ERROR',
        message: '网络连接失败，请检查网络设置',
        originalError: error
      };
    } else {
      // 其他错误
      return {
        type: 'UNKNOWN_ERROR',
        message: error.message || '未知错误',
        originalError: error
      };
    }
  }

  /**
   * 获取当前位置
   */
  async getCurrentLocation() {
    try {
      const response = await this.client.get('/api/location/current');
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 搜索POI
   */
  async searchPOI({ keywords, location, radius = 1000, pageSize = 10 }) {
    try {
      const response = await this.client.post('/api/poi/search', {
        keywords,
        location,
        radius,
        page_size: pageSize,
        sort_rule: 'BLIND_FRIENDLY'
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 规划路径
   */
  async planRoute({ origin, destination, strategy = 'SAFE_PRIORITY' }) {
    try {
      const response = await this.client.post('/api/route/plan', {
        origin,
        destination,
        strategy
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 开始导航
   */
  async startNavigation({ route, preferences = {} }) {
    try {
      const response = await this.client.post('/api/navigation/start', {
        route,
        preferences: {
          voice: true,
          vibration: true,
          safetyFirst: true,
          ...preferences
        }
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 更新导航状态
   */
  async updateNavigation({ sessionId, currentLocation }) {
    try {
      const response = await this.client.post('/api/navigation/update', {
        session_id: sessionId,
        location: currentLocation
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 停止导航
   */
  async stopNavigation({ sessionId }) {
    try {
      const response = await this.client.post('/api/navigation/stop', {
        session_id: sessionId
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 获取交通信息
   */
  async getTrafficInfo({ location, radius = 2000 }) {
    try {
      const response = await this.client.post('/api/traffic/status', {
        location,
        radius
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 地理编码
   */
  async geocodeAddress({ address }) {
    try {
      const response = await this.client.post('/api/location/geocode', {
        address
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 逆地理编码
   */
  async reverseGeocode({ location }) {
    try {
      const response = await this.client.post('/api/location/reverse-geocode', {
        location
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 获取输入提示
   */
  async getInputTips({ keywords, city, limit = 10 }) {
    try {
      const response = await this.client.post('/api/poi/tips', {
        keywords,
        city,
        limit
      });
      
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }

  /**
   * 健康检查
   */
  async healthCheck() {
    try {
      const response = await this.client.get('/api/health');
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      return {
        success: false,
        error: error
      };
    }
  }
}

// 导出单例
export default new ApiService();
