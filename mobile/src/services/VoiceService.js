/**
 * BlindStar语音服务
 * 处理语音合成(TTS)和语音识别(STT)功能
 */

import Tts from 'react-native-tts';
import Voice from 'react-native-voice';
import { Platform } from 'react-native';
import Config from '../config/Config';

class VoiceService {
  constructor() {
    this.isSpeaking = false;
    this.isListening = false;
    this.speechQueue = [];
    this.currentPriority = Config.VOICE.PRIORITY.DEBUG;
    
    this._initializeTts();
    this._initializeVoice();
  }

  /**
   * 初始化TTS
   */
  async _initializeTts() {
    try {
      // 设置TTS选项
      const options = Config.VOICE.TTS_OPTIONS;
      
      if (Platform.OS === 'android') {
        await Tts.setDefaultLanguage(options.language);
        await Tts.setDefaultRate(options.rate);
        await Tts.setDefaultPitch(options.pitch);
      } else {
        // iOS配置
        await Tts.setDefaultLanguage(options.language);
        await Tts.setDefaultRate(options.rate);
      }

      // 设置TTS事件监听
      Tts.addEventListener('tts-start', this._onTtsStart.bind(this));
      Tts.addEventListener('tts-finish', this._onTtsFinish.bind(this));
      Tts.addEventListener('tts-cancel', this._onTtsCancel.bind(this));
      Tts.addEventListener('tts-error', this._onTtsError.bind(this));

      console.log('🎤 TTS初始化成功');
    } catch (error) {
      console.error('❌ TTS初始化失败:', error);
    }
  }

  /**
   * 初始化语音识别
   */
  async _initializeVoice() {
    try {
      // 设置语音识别事件监听
      Voice.onSpeechStart = this._onSpeechStart.bind(this);
      Voice.onSpeechEnd = this._onSpeechEnd.bind(this);
      Voice.onSpeechResults = this._onSpeechResults.bind(this);
      Voice.onSpeechPartialResults = this._onSpeechPartialResults.bind(this);
      Voice.onSpeechError = this._onSpeechError.bind(this);

      console.log('🎧 语音识别初始化成功');
    } catch (error) {
      console.error('❌ 语音识别初始化失败:', error);
    }
  }

  /**
   * 语音播报
   */
  async speak(text, priority = Config.VOICE.PRIORITY.INFORMATION, options = {}) {
    try {
      if (!text || typeof text !== 'string') {
        console.warn('⚠️ 无效的语音文本:', text);
        return;
      }

      const speechItem = {
        text: text.trim(),
        priority,
        options: {
          ...Config.VOICE.TTS_OPTIONS,
          ...options
        },
        timestamp: Date.now()
      };

      // 根据优先级处理
      if (priority <= Config.VOICE.PRIORITY.NAVIGATION) {
        // 高优先级：立即播报，清空队列
        this.speechQueue = [];
        await this._speakImmediate(speechItem);
      } else {
        // 低优先级：加入队列
        this.speechQueue.push(speechItem);
        if (!this.isSpeaking) {
          this._processQueue();
        }
      }

    } catch (error) {
      console.error('❌ 语音播报失败:', error);
    }
  }

  /**
   * 立即播报（中断当前播报）
   */
  async speakImmediate(text, options = {}) {
    await this.speak(text, Config.VOICE.PRIORITY.EMERGENCY, options);
  }

  /**
   * 停止当前播报
   */
  async stopSpeaking() {
    try {
      await Tts.stop();
      this.speechQueue = [];
      this.isSpeaking = false;
      console.log('🛑 停止语音播报');
    } catch (error) {
      console.error('❌ 停止语音播报失败:', error);
    }
  }

  /**
   * 开始语音识别
   */
  async startListening(options = {}) {
    try {
      if (this.isListening) {
        console.log('⚠️ 语音识别已在运行');
        return;
      }

      const voiceOptions = {
        ...Config.VOICE.STT_OPTIONS,
        ...options
      };

      await Voice.start(voiceOptions.language, voiceOptions);
      this.isListening = true;
      console.log('🎧 开始语音识别');

    } catch (error) {
      console.error('❌ 开始语音识别失败:', error);
      this.isListening = false;
      throw error;
    }
  }

  /**
   * 停止语音识别
   */
  async stopListening() {
    try {
      await Voice.stop();
      this.isListening = false;
      console.log('🛑 停止语音识别');
    } catch (error) {
      console.error('❌ 停止语音识别失败:', error);
    }
  }

  /**
   * 取消语音识别
   */
  async cancelListening() {
    try {
      await Voice.cancel();
      this.isListening = false;
      console.log('❌ 取消语音识别');
    } catch (error) {
      console.error('❌ 取消语音识别失败:', error);
    }
  }

  /**
   * 检查语音识别是否可用
   */
  async isVoiceAvailable() {
    try {
      return await Voice.isAvailable();
    } catch (error) {
      console.error('❌ 检查语音识别可用性失败:', error);
      return false;
    }
  }

  /**
   * 获取支持的语言
   */
  async getSupportedLanguages() {
    try {
      return await Voice.getSupportedLanguages();
    } catch (error) {
      console.error('❌ 获取支持语言失败:', error);
      return [];
    }
  }

  /**
   * 立即播报
   */
  async _speakImmediate(speechItem) {
    try {
      if (this.isSpeaking) {
        await Tts.stop();
      }
      
      this.isSpeaking = true;
      this.currentPriority = speechItem.priority;
      
      console.log('🔊 播报:', speechItem.text);
      await Tts.speak(speechItem.text, speechItem.options);
      
    } catch (error) {
      console.error('❌ 立即播报失败:', error);
      this.isSpeaking = false;
    }
  }

  /**
   * 处理播报队列
   */
  async _processQueue() {
    if (this.speechQueue.length === 0 || this.isSpeaking) {
      return;
    }

    try {
      const speechItem = this.speechQueue.shift();
      await this._speakImmediate(speechItem);
    } catch (error) {
      console.error('❌ 处理播报队列失败:', error);
      this.isSpeaking = false;
      // 继续处理下一个
      setTimeout(() => this._processQueue(), 1000);
    }
  }

  /**
   * TTS事件处理
   */
  _onTtsStart(event) {
    console.log('🎤 TTS开始播报:', event);
    this.isSpeaking = true;
  }

  _onTtsFinish(event) {
    console.log('✅ TTS播报完成:', event);
    this.isSpeaking = false;
    this.currentPriority = Config.VOICE.PRIORITY.DEBUG;
    
    // 处理队列中的下一个
    setTimeout(() => this._processQueue(), 100);
  }

  _onTtsCancel(event) {
    console.log('❌ TTS播报取消:', event);
    this.isSpeaking = false;
    this.currentPriority = Config.VOICE.PRIORITY.DEBUG;
  }

  _onTtsError(event) {
    console.error('❌ TTS播报错误:', event);
    this.isSpeaking = false;
    this.currentPriority = Config.VOICE.PRIORITY.DEBUG;
    
    // 处理队列中的下一个
    setTimeout(() => this._processQueue(), 1000);
  }

  /**
   * 语音识别事件处理
   */
  _onSpeechStart(event) {
    console.log('🎧 语音识别开始:', event);
    this.isListening = true;
  }

  _onSpeechEnd(event) {
    console.log('🛑 语音识别结束:', event);
    this.isListening = false;
  }

  _onSpeechResults(event) {
    console.log('✅ 语音识别结果:', event.value);
    if (this.onSpeechResults) {
      this.onSpeechResults(event.value);
    }
  }

  _onSpeechPartialResults(event) {
    console.log('📝 语音识别部分结果:', event.value);
    if (this.onSpeechPartialResults) {
      this.onSpeechPartialResults(event.value);
    }
  }

  _onSpeechError(event) {
    console.error('❌ 语音识别错误:', event.error);
    this.isListening = false;
    if (this.onSpeechError) {
      this.onSpeechError(event.error);
    }
  }

  /**
   * 设置语音识别回调
   */
  setSpeechCallbacks({ onResults, onPartialResults, onError }) {
    this.onSpeechResults = onResults;
    this.onSpeechPartialResults = onPartialResults;
    this.onSpeechError = onError;
  }

  /**
   * 清理资源
   */
  cleanup() {
    try {
      // 停止TTS
      Tts.stop();
      
      // 移除TTS事件监听
      Tts.removeAllListeners('tts-start');
      Tts.removeAllListeners('tts-finish');
      Tts.removeAllListeners('tts-cancel');
      Tts.removeAllListeners('tts-error');

      // 停止语音识别
      Voice.cancel();
      Voice.removeAllListeners();

      // 清理状态
      this.isSpeaking = false;
      this.isListening = false;
      this.speechQueue = [];
      
      console.log('🧹 语音服务清理完成');
    } catch (error) {
      console.error('❌ 语音服务清理失败:', error);
    }
  }
}

// 导出单例
export default new VoiceService();
