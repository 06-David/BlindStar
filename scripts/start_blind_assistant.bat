@echo off
echo ========================================
echo BlindStar 盲人助手启动脚本
echo ========================================

REM 尝试激活conda环境
echo 正在尝试激活conda环境...
call conda activate blind 2>nul
if %errorlevel% neq 0 (
    echo 无法激活blind环境，尝试使用默认Python环境
)

REM 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：Python未安装或不在PATH中
    pause
    exit /b 1
)

echo Python环境检查通过

REM 检查必要的依赖
echo 检查依赖项...
python -c "import pyttsx3; print('TTS: OK')" 2>nul || echo "TTS: 不可用"
python -c "import vosk; print('STT: OK')" 2>nul || echo "STT: 不可用"
python -c "import requests; print('Network: OK')" 2>nul || echo "Network: 不可用"

echo.
echo 启动BlindStar盲人助手...
echo 按Ctrl+C可以随时退出程序
echo ========================================

REM 启动盲人助手
python blind_assistant.py

echo.
echo 程序已退出
pause
