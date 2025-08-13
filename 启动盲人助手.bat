@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   BlindStar 盲人语音导航助手
echo ========================================
echo.
echo 正在启动专为盲人用户设计的语音导航助手...
echo.
echo 功能特色：
echo ✓ 完全语音化交互 - 所有信息都通过语音播报
echo ✓ 附近地点查询 - 银行、医院、超市、餐厅
echo ✓ 智能语音导航 - 详细路线规划和指引
echo ✓ 位置信息播报 - 详细的环境描述
echo ✓ 安全保障机制 - 优先无障碍路线
echo.
echo 语音命令示例：
echo • "附近银行" - 查找附近银行并详细介绍
echo • "导航到北京大学" - 开始语音导航
echo • "位置" - 播报当前位置和环境
echo • "时间" - 播报当前时间
echo • "帮助" - 获取完整功能说明
echo • "退出" - 退出程序
echo.
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误：未找到Python环境
    echo 请确保已安装Python 3.8或更高版本
    echo.
    pause
    exit /b 1
)

echo ✅ Python环境检查通过
echo.

REM 尝试激活conda环境
call conda activate blind >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 已激活conda blind环境
) else (
    echo ℹ️ 使用默认Python环境
)

echo.
echo 正在启动BlindStar盲人语音助手...
echo 启动后请按照语音提示操作
echo 按 Ctrl+C 可随时退出程序
echo.
echo ========================================
echo.

REM 启动主程序
python blind_voice_assistant.py

echo.
echo ========================================
echo BlindStar 盲人语音助手已退出
echo 感谢使用！
echo ========================================
pause
