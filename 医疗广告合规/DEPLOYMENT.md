# 部署指南 - 如何让他人使用本系统

本指南提供多种方式让他人访问和使用医疗广告合规检测系统。

## 方案一：Streamlit Cloud（推荐 - 最简单）

### 步骤

1. **创建 GitHub 账户**（如果没有）
   - 访问 https://github.com
   - 注册免费账户

2. **上传项目到 GitHub**
   ```powershell
   git init
   git add .
   git commit -m "初始提交：医疗广告合规检测系统"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/medical-ad-compliance.git
   git push -u origin main
   ```

3. **连接 Streamlit Cloud**
   - 访问 https://streamlit.io/cloud
   - 点击 "New app"
   - 选择 GitHub 仓库
   - 选择 `app_lite.py` 作为主文件（推荐）或 `app.py`
   - 点击 Deploy

4. **分享链接**
   - 部署完成后会获得形如 `https://medical-ad-compliance.streamlit.app/` 的公开链接
   - 将链接分享给任何人，他们可以直接在浏览器中使用

### 优点
- ✅ 完全免费（有限免费额度）
- ✅ 无需自己管理服务器
- ✅ 自动更新（GitHub 更新后自动部署）
- ✅ 支持 HTTPS 和自定义域名

### 缺点
- ⚠️ 首次加载可能较慢（冷启动）
- ⚠️ 免费版有资源限制
- ⚠️ 并发用户有限制

---

## 方案二：本地 LAN 共享（局域网内使用）

适合企业内网或同一网络下的多人使用。

### 步骤

1. **获取本机 IP 地址**
   ```powershell
   ipconfig
   # 查看 IPv4 地址，如 192.168.x.x
   ```

2. **修改 Streamlit 配置文件**
   
   创建 `~/.streamlit/config.toml`（或 `%USERPROFILE%\.streamlit\config.toml` on Windows）：
   ```toml
   [server]
   headless = true
   address = "0.0.0.0"
   port = 8501
   enableXsrfProtection = true
   ```

3. **启动应用**
   ```powershell
   streamlit run app_lite.py --server.address=0.0.0.0
   ```

4. **他人访问**
   - 在同一网络的其他电脑上打开浏览器
   - 访问 `http://YOUR_IP:8501`（如 `http://192.168.1.100:8501`）

### 优点
- ✅ 完全免费
- ✅ 无需互联网
- ✅ 响应快速
- ✅ 数据本地存储

### 缺点
- ⚠️ 仅限同一网络
- ⚠️ 需要保持服务器运行
- ⚠️ 无 HTTPS（不建议涉及敏感数据）

---

## 方案三：Docker 容器化部署

使用 Docker 让任何人都能快速部署。

### 步骤

1. **创建 Dockerfile**
   
   在项目根目录创建 `Dockerfile`：
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

      # 安装系统依赖（无需 Tesseract，项目已移除 OCR 功能）
      RUN apt-get update && apt-get install -y \
         git \
         && rm -rf /var/lib/apt/lists/*

   # 复制项目文件
   COPY . .

   # 安装 Python 依赖
   RUN pip install --no-cache-dir -r requirements.txt

   # 暴露端口
   EXPOSE 8501

   # 创建 Streamlit 配置
   RUN mkdir -p ~/.streamlit && \
       echo "[server]" > ~/.streamlit/config.toml && \
       echo "headless = true" >> ~/.streamlit/config.toml && \
       echo "port = 8501" >> ~/.streamlit/config.toml && \
       echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml

   # 启动应用
   CMD ["streamlit", "run", "app_lite.py"]
   ```

2. **创建 .dockerignore**
   ```
   .git
   __pycache__
   *.pyc
   .streamlit
   reports.db
   models/
   ```

3. **构建 Docker 镜像**
   ```powershell
   docker build -t medical-ad-compliance:latest .
   ```

4. **运行容器**
   ```powershell
   docker run -p 8501:8501 medical-ad-compliance:latest
   ```

5. **他人使用**
   - 只需安装 Docker
   - 运行同样的命令即可

### 优点
- ✅ 环境隔离，避免依赖冲突
- ✅ 跨平台兼容（Windows/Mac/Linux）
- ✅ 易于扩展和部署

### 缺点
- ⚠️ 需要学习 Docker
- ⚠️ 镜像较大（包含所有依赖）

---

## 方案四：云服务部署（付费但功能完整）

### A. Heroku（已停止免费服务）
已不再推荐，Heroku 取消了免费层。

### B. AWS / Azure / GCP
适合企业级部署。

### C. Hugging Face Spaces（推荐，免费）
1. 访问 https://huggingface.co/spaces
2. 创建新 Space，选择 Streamlit 模板
3. 上传项目文件
4. 自动获得公开链接

### D. Railway / Render / Fly.io
- Railway：https://railway.app/
- Render：https://render.com/
- Fly.io：https://fly.io/

都提供免费试用和按需付费方案。

---

## 方案五：生成可执行文件（Windows 应用）

使用 PyInstaller 将项目打包为 .exe 文件。

### 步骤

1. **安装 PyInstaller**
   ```powershell
   pip install pyinstaller
   ```

2. **创建启动脚本** `launcher.py`
   ```python
   import subprocess
   import sys
   
   subprocess.run([sys.executable, "-m", "streamlit", "run", "app_lite.py"])
   ```

3. **打包**
   ```powershell
   pyinstaller --onefile --windowed launcher.py
   ```

4. **分发**
   - `dist/launcher.exe` 即可单独运行
   - 用户只需双击即可启动应用

### 优点
- ✅ 用户友好，无需 Python 环境
- ✅ 一键启动

### 缺点
- ⚠️ 文件较大（100MB+）
- ⚠️ Windows 可能被误报为病毒

---

## 推荐方案总结

| 使用场景 | 推荐方案 | 难度 | 成本 |
|---------|---------|------|------|
| **快速分享给他人** | Streamlit Cloud | ⭐ 简单 | 免费 |
| **企业内网使用** | LAN 共享 | ⭐ 简单 | 免费 |
| **长期运营/商业** | Docker + 云服务 | ⭐⭐⭐ 中等 | 按需付费 |
| **个人电脑应用** | .exe 可执行文件 | ⭐⭐ 中等 | 免费 |
| **完整企业方案** | AWS/Azure + 域名 | ⭐⭐⭐ 复杂 | 每月几十到几百元 |

---

## 快速开始：Streamlit Cloud 详细步骤

### 1. 初始化 Git 仓库
```powershell
cd c:\Users\mesleo\Desktop\医疗广告合规
git init
git config user.name "Your Name"
git config user.email "your@email.com"
git add .
git commit -m "医疗广告合规检测系统初始版本"
```

### 2. 创建 GitHub 仓库
- 访问 https://github.com/new
- 仓库名：`medical-ad-compliance`
- 描述：Medical Advertisement Compliance Detection System
- 创建后按照指引 push 代码

### 3. 在 Streamlit Cloud 部署
```
https://streamlit.io/cloud → Sign in with GitHub
→ New app → medical-ad-compliance repo → app_lite.py
```

### 4. 分享链接
部署完成后获得 URL，如：
```
https://medical-ad-compliance.streamlit.app/
```

立即将链接分享给同事、客户、朋友！

---

## 常见问题

### Q: 如何更新已部署的应用？
A: 只需在 GitHub 推送更新，Streamlit Cloud 会自动重新部署。

### Q: 如何保护数据隐私？
A: 
- 不要在 Streamlit Cloud 存储敏感数据
- 使用 LAN 共享方案在内网部署
- 启用 Streamlit 的身份验证功能

### Q: 如何处理大量并发用户？
A: 
- 小规模：Streamlit Cloud 足够
- 中等规模：使用 Docker + Kubernetes
- 大规模：搭建专门的 API 服务（FastAPI）

### Q: 成本多少？
A:
- Streamlit Cloud 免费版：$0/月
- LAN 共享：$0/月
- Docker 自建：$5-50/月（取决于云服务商）
- 企业级方案：$100+/月

---

## 建议的部署方案

**对于您的项目，推荐步骤：**

1. **现在**：使用 Streamlit Cloud 快速分享
   ```
   5 分钟内让所有人都能使用
   ```

2. **后期**：如需更多功能
   ```
   Docker + 自建服务器或云服务
   ```

3. **规模化**：如成为商业产品
   ```
   FastAPI 后端 + React 前端 + 专业托管
   ```

---

## 需要帮助？

如果您想使用某个方案，告诉我具体需求，我可以帮您：
- 配置 Docker 文件
- 设置 GitHub 仓库
- 优化部署配置
- 创建部署文档

