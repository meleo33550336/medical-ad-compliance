# 医疗广告合规检测系统 - 部署说明

## 快速启动

### 本地运行（开发）
```bash
# 快速版本（推荐）
streamlit run app_lite.py

# 完整版本（包含 ML 模型）
streamlit run app.py
```

### Docker 部署（推荐生产环境）
```bash
# 构建镜像
docker build -t medical-ad-compliance .

# 运行容器
docker run -p 8501:8501 medical-ad-compliance

# 访问应用
# http://localhost:8501
```

### 云端部署（Streamlit Cloud - 最简单）

1. **上传到 GitHub**
   ```bash
   git init
   git add .
   git commit -m "medical ad compliance system"
   git remote add origin https://github.com/YOUR_USERNAME/medical-ad-compliance.git
   git push -u origin main
   ```

2. **在 Streamlit Cloud 部署**
   - 访问 https://streamlit.io/cloud
   - 使用 GitHub 账户登录
   - 点击 "New app"
   - 选择此仓库，指定 `app_lite.py`
   - Deploy

3. **分享链接**
   - 获得公开 URL，分享给任何人

## 版本选择

- **app_lite.py** ✅ 推荐
  - 加载时间：< 2 秒
  - 功能：规则匹配、敏感词检测、分词
  - 适用：快速检测、生产环境

- **app.py** 完整版
  - 加载时间：10-30 秒（首次）
  - 功能：包含语义相似度、BERT 分类
  - 适用：深度分析、研究

## 系统要求

### 最低配置
- Python 3.8+
- 2GB RAM
- 200MB 磁盘空间

### 推荐配置
- Python 3.9+
- 4GB RAM
- 500MB 磁盘空间

### 依赖项
- Python 包：见 requirements.txt

## 配置

### 环境变量
```bash
# 可选配置
SEMANTIC_THRESHOLD=0.65            # 语义相似度阈值
```

### Streamlit 配置
已在 Dockerfile 中配置，本地运行时可修改 `~/.streamlit/config.toml`

## 访问方式

| 部署方式 | URL 示例 | 访问范围 |
|---------|---------|--------|
| 本地 | `http://localhost:8501` | 仅本机 |
| LAN | `http://192.168.1.100:8501` | 同一网络 |
| Docker | `http://localhost:8501` | 本机/网络 |
| Streamlit Cloud | `https://xxx.streamlit.app` | 全网公开 |
| AWS/Azure | 需配置域名和 HTTPS | 全网公开 |

## 常见问题

### 如何让不同网络的人访问？
使用 Streamlit Cloud 或云服务（AWS、Azure 等）

### 如何加速首次加载？
使用 `app_lite.py` 版本（无 ML 模型）

### 如何保护用户数据？
- 使用 LAN 部署（数据本地存储）
- 或使用自托管云服务
- 避免在公开云服务上存储敏感信息

### 如何处理大量用户？
- 小规模：Streamlit Cloud 足够
- 中等规模：Docker + 自建服务器
- 大规模：FastAPI 后端 + 专业托管

## 支持

详见 DEPLOYMENT.md 了解更多部署方案和技术细节

---

**推荐步骤**：
1. 本地测试：`streamlit run app_lite.py`
2. 快速分享：使用 Streamlit Cloud（5分钟）
3. 长期部署：Docker + 云服务

