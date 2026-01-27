@echo off
REM 医疗广告合规检测系统 - Windows 快速启动脚本

setlocal enabledelayedexpansion

echo.
echo =====================================
echo 医疗广告合规检测系统
echo =====================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python！
    echo 请先安装 Python 3.8+ 后重试
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [✓] Python 已安装

REM 检查虚拟环境
if not exist "venv" (
    echo [创建] 正在创建虚拟环境...
    python -m venv venv
    echo [✓] 虚拟环境创建完成
)

REM 激活虚拟环境
call venv\Scripts\activate.bat
echo [✓] 虚拟环境已激活

REM 检查依赖
echo [检查] 检查依赖包...
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo [安装] 正在安装依赖包...
    pip install -r requirements.txt -q
    echo [✓] 依赖包安装完成
) else (
    echo [✓] 依赖包已安装
)

echo.
echo =====================================
echo 选择要启动的版本:
echo =====================================
echo 1. 快速版 (app_lite.py) - 推荐，加载 < 2 秒
echo 2. 完整版 (app.py) - 包含 ML 模型，功能完整
echo 3. 退出
echo.

set /p choice="请选择 (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo [启动] 正在启动快速版...
    echo 应用地址: http://localhost:8501
    echo.
    streamlit run app_lite.py
) else if "%choice%"=="2" (
    echo.
    echo [启动] 正在启动完整版（首次可能需要 10-30 秒）...
    echo 应用地址: http://localhost:8501
    echo.
    streamlit run app.py
) else (
    echo 已退出
    exit /b 0
)

endlocal
