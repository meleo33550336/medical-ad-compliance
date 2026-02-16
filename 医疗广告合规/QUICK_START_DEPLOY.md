# 🚀 5 分钟快速部署指南

## 目标：让所有人都能使用您的医疗广告合规检测系统

选择下方任意一种方案，5 分钟内让他人访问！

---

## 方案 A：Streamlit Cloud（最简单，完全免费）⭐ 推荐

**适合场景**：想快速分享给所有人，无需管理服务器

### 步骤（5 分钟）

#### 1️⃣ 上传到 GitHub（3 分钟）

```bash
# 在项目根目录运行
git init
git add .
git commit -m "Medical advertisement compliance detection system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/medical-ad-compliance.git
git push -u origin main
```

**没有 Git？**[安装 Git](https://git-scm.com/download)

#### 2️⃣ 在 Streamlit Cloud 部署（2 分钟）

1. 访问 [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. 用 GitHub 账户登录
3. 点击 **New app**
4. 选择刚上传的仓库：`medical-ad-compliance`
5. 指定文件：`app_lite.py`（推荐快速版）
6. 点击 **Deploy**

#### 3️⃣ 完成！

部署完成后，您会获得一个公开 URL，如：
```
https://medical-ad-compliance.streamlit.app/
```

### 分享给他人

- 直接分享这个链接
- 他们可以立即在任何设备上使用，无需安装任何东西

### 成本

- **完全免费**（Streamlit Cloud 提供免费额度）

---

## 方案 B：Docker 部署（企业级，推荐自建服务）

**适合场景**：想在自己的服务器上部署，或分发给内部使用

### 前置条件

- [安装 Docker](https://www.docker.com/products/docker-desktop)

### 步骤（10 分钟）

#### 1️⃣ 构建 Docker 镜像

```bash
cd c:\Users\mesleo\Desktop\医疗广告合规
docker build -t medical-ad-compliance:v1 .
```

#### 2️⃣ 运行容器

```bash
docker run -d -p 8501:8501 \
  --name medical-ad-compliance \
  medical-ad-compliance:v1
```

#### 3️⃣ 访问应用

打开浏览器：`http://localhost:8501`

#### 4️⃣ 分享给他人（局域网内）

他人在同一网络上访问：
```
http://YOUR_IP:8501
```

查询您的 IP：
```bash
ipconfig  # Windows
ifconfig  # Mac/Linux
```

### 分发给他人

对方只需安装 Docker，然后运行：
```bash
docker run -d -p 8501:8501 \
  medical-ad-compliance:v1
```

### 成本

- **完全免费**（如果在自己的服务器上）
- 或按月付费部署到云服务（AWS、Azure 等）

---

## 方案 C：本地网络共享（LAN 部署）

**适合场景**：只想在公司内网或家庭网络中使用

### 前置条件

- 仅需 Python 3.8+

### 步骤（5 分钟）

#### 1️⃣ 启动应用

```bash
# Windows
.\start.bat
# 选择选项 1（快速版）

# Linux/Mac
chmod +x start.sh
./start.sh
```

#### 2️⃣ 获取您的 IP 地址

Windows：
```powershell
ipconfig
# 查找 IPv4 Address，如 192.168.1.100
```

Mac/Linux：
```bash
ifconfig
# 查找 inet address
```

#### 3️⃣ 分享给他人

同一网络内的其他电脑访问：
```
http://YOUR_IP:8501
```

例如：`http://192.168.1.100:8501`

### 成本

- **完全免费**

---

## 方案 D：Hugging Face Spaces（另一个免费云部署）

**适合场景**：想要完全免费的云部署，集成 Hugging Face 生态

### 步骤（10 分钟）

1. 访问 [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. 点击 **Create new Space**
3. 选择 **Streamlit** 模板
4. 上传项目文件（或连接 GitHub）
5. 指定启动文件：`app_lite.py`
6. 发布

### 优点

- 完全免费
- 自动 HTTPS
- 支持 GPU（需付费）

---

## 方案 E：自建 FastAPI 后端（高级生产方案）

**适合场景**：需要高并发、想要 REST API、或集成到其他系统

[见 DEPLOYMENT.md 中的详细说明](./DEPLOYMENT.md)

---

## 📊 方案对比

| 方案 | 难度 | 成本 | 速度 | 用户 | 推荐度 |
|------|------|------|------|------|--------|
| **Streamlit Cloud** | ⭐ | 免费 | 中 | 小-中 | ⭐⭐⭐⭐⭐ |
| **Docker** | ⭐⭐ | 免费-少量 | 快 | 中 | ⭐⭐⭐⭐ |
| **LAN 共享** | ⭐ | 免费 | 快 | 小 | ⭐⭐⭐ |
| **Hugging Face** | ⭐ | 免费 | 中 | 小-中 | ⭐⭐⭐⭐ |
| **FastAPI** | ⭐⭐⭐ | 按量付费 | 快 | 大 | ⭐⭐⭐ |

---

## 🎯 推荐流程

### 第 1 步：立即分享（推荐）
使用 **方案 A：Streamlit Cloud**
- 3 分钟内完成
- 免费使用
- 所有人都能访问
- ✅ **立即做这个**

### 第 2 步：生产部署（可选）
当用户增多时，迁移到：
- **方案 B：Docker**（成本低）
- **方案 E：FastAPI**（功能最强）

---

## ❓ 常见问题

### Q: Streamlit Cloud 需要付费吗？
**A:** 不需要，完全免费。有免费额度限制，但对个人使用足够。

### Q: Docker 镜像有多大？
**A:** 约 1.5GB（不包含 OCR 相关依赖）

### Q: 其他人上传的文件会存储吗？
**A:** 在 Streamlit Cloud 上不会持久化。本地部署会存储在 SQLite 数据库中。

### Q: 如何更新部署的应用？
**A:** 
- **Streamlit Cloud**：在 GitHub 推送更新，自动重新部署
- **Docker**：重建镜像，重启容器
- **LAN/本地**：重启应用即可

### Q: 支持 HTTPS 吗？
**A:** 
- **Streamlit Cloud**：自动支持
- **Docker/LAN**：需要额外配置（Nginx）

### Q: 如何处理大量并发用户？
**A:** 
- < 100 用户：任何方案都可以
- 100-1000 用户：Docker 或 Hugging Face Spaces
- > 1000 用户：FastAPI + 专业云服务

---

## 🆘 遇到问题？

### 问题 1：Streamlit Cloud 部署失败

**解决**：
1. 检查 requirements.txt 中的所有依赖
2. 确保仓库中有 `.streamlit/config.toml`
3. 查看部署日志了解详细错误

### 问题 2：Docker 构建失败

**解决**：
1. 确保安装了最新的 Docker
2. 检查 Dockerfile 中的系统依赖
3. 运行 `docker build --no-cache` 重新构建

### 问题 3：LAN 访问不了

**解决**：
1. 确保防火墙允许 8501 端口
2. 使用正确的 IP 地址（运行 `ipconfig`）
3. 确认两台电脑在同一网络

---

## 📞 更多帮助

- 查看 [DEPLOYMENT.md](./DEPLOYMENT.md) 了解详细部署指南
- 查看 [README.md](./README.md) 了解项目说明
- 查看 [README_DEPLOY.md](./README_DEPLOY.md) 了解配置指南

---

## ✅ 立即开始

**最简单的方式（推荐）：**

```bash
# 1. 上传到 GitHub
git add .
git commit -m "medical ad compliance"
git push

# 2. 访问 Streamlit Cloud
# https://streamlit.io/cloud

# 3. 完成！5 分钟内全球都能访问您的应用
```

**现在就开始吧！** 🚀

