#!/bin/bash
# 医疗广告合规检测系统 - Linux/Mac 快速启动脚本

set -e

echo ""
echo "====================================="
echo "医疗广告合规检测系统"
echo "====================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python 3！"
    echo "请先安装 Python 3.8+ 后重试"
    exit 1
fi

echo "[✓] Python 已安装: $(python3 --version)"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "[创建] 正在创建虚拟环境..."
    python3 -m venv venv
    echo "[✓] 虚拟环境创建完成"
fi

# 激活虚拟环境
source venv/bin/activate
echo "[✓] 虚拟环境已激活"

# 安装依赖
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "[安装] 正在安装依赖包..."
    pip install -q -r requirements.txt
    echo "[✓] 依赖包安装完成"
else
    echo "[✓] 依赖包已安装"
fi

echo ""
echo "====================================="
echo "选择要启动的版本:"
echo "====================================="
echo "1. 快速版 (app_lite.py) - 推荐，加载 < 2 秒"
echo "2. 完整版 (app.py) - 包含 ML 模型，功能完整"
echo "3. 退出"
echo ""

read -p "请选择 (1/2/3): " choice

case "$choice" in
    1)
        echo ""
        echo "[启动] 正在启动快速版..."
        echo "应用地址: http://localhost:8501"
        echo ""
        streamlit run app_lite.py
        ;;
    2)
        echo ""
        echo "[启动] 正在启动完整版（首次可能需要 10-30 秒）..."
        echo "应用地址: http://localhost:8501"
        echo ""
        streamlit run app.py
        ;;
    3)
        echo "已退出"
        exit 0
        ;;
    *)
        echo "[错误] 无效选择"
        exit 1
        ;;
esac
