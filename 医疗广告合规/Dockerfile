FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖（其他工具）
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 Streamlit 端口
EXPOSE 8501

# 创建 Streamlit 配置目录和文件
RUN mkdir -p ~/.streamlit && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "port = 8501" >> ~/.streamlit/config.toml && \
    echo "address = \"0.0.0.0\"" >> ~/.streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml && \
    echo "[logger]" >> ~/.streamlit/config.toml && \
    echo "level = \"info\"" >> ~/.streamlit/config.toml

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501').read()" || exit 1

# 启动应用（使用 app_lite.py 快速版本）
CMD ["streamlit", "run", "app_lite.py"]
