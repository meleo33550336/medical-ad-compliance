# 医疗广告合规检测系统

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-green.svg)

一个高效的医疗广告文本合规性检测系统，采用多层检测策略（规则匹配、语义分析、BERT 分类）确保检测精度。

## ✨ 核心特性

- ⚡ **快速检测**：快速版本 2 秒内加载
- 🔍 **多层检测**：规则 + 语义相似度 + BERT 分类
<!-- OCR 已移除：当前仅支持文本输入 -->
- 🎨 **可视化报告**：违规词红色标记、详细数据统计
- 📁 **批量处理**：支持批量文件检测
- 💾 **数据存储**：SQLite 数据库自动保存历史
- ⚙️ **规则管理**：Web 界面编辑检测规则
- 🌐 **云端部署**：支持多种部署方案

# 命令行使用（文本）

```powershell
# 文本检测
python main.py --input "本产品保证治愈，百分之百安全"
```

2. **启动应用**
```powershell
streamlit run app_lite.py      # 快速版（推荐）
# streamlit run app.py         # 完整版
```

3. **访问应用**
- 打开浏览器访问：http://localhost:8501

### 命令行使用

```powershell
# 文本检测（文件或直接文本）
python main.py --input "本产品保证治愈，百分之百安全"
```

输出：程序会打印 JSON 报告，并在 `reports/` 目录下保存一个报告文件。

## 📦 部署方案

### 方案 1：Streamlit Cloud（推荐 - 3 分钟快速分享）

**最简单！一行命令让所有人都能使用：**

1. 上传到 GitHub
   ```bash
   git add .
   git commit -m "initial"
   git push origin main
   ```

2. 在 [Streamlit Cloud](https://streamlit.io/cloud) 部署
   - 选择仓库
   - 指定 `app_lite.py`
   - Deploy

3. 获得公开链接：`https://xxx.streamlit.app/`

### 方案 2：Docker 部署（推荐企业部署）

```bash
docker build -t medical-compliance .
docker run -p 8501:8501 medical-compliance
# 访问 http://localhost:8501
```

### 方案 3：本地 LAN 共享（局域网内使用）

```powershell
streamlit run app_lite.py --server.address=0.0.0.0
# 其他电脑访问：http://YOUR_IP:8501
```

[📖 查看完整部署指南 →](./DEPLOYMENT.md)

## 📊 系统架构

```
输入文本
    ↓
[1] 规则匹配 (正则表达式)
    ↓
[3] 敏感词检测 (字典匹配)
    ↓
[4] 语义相似度 (Sentence-Transformers BERT)
    ↓
[5] 分类器 (微调 BERT)
    ↓
合规性报告 (JSON)
```

## 📈 性能指标

| 指标 | 快速版 | 完整版 |
|-----|-------|--------|
| **加载时间** | < 2 秒 | 10-30 秒 |
| **单次检测** | 100ms | 500ms |
| **批量处理** | 1000 条/分钟 | 200 条/分钟 |
| **内存占用** | 100MB | 4GB |

### 版本选择

**快速版** (`app_lite.py`) - 推荐首先使用：
- 仅使用违规规则 + 敏感词正则检测
- 启动时间 < 2 秒，检测 < 1 秒
- 无需加载任何机器学习模型
- 适合快速批量检测或网络条件较弱的环境

**完整版** (`app.py`) - 功能最全：
- 包含规则检测 + 语义相似度 + 分类器
- 首次启动 10-30 秒（加载模型），之后 < 5 秒
- 自动缓存模型，后续调用高速
- 侧边栏可选择启用/禁用各检测模块

### 若遇到超时

1. **立即尝试**：
   ```powershell
   streamlit run app_lite.py
   ```
   这个版本应该能秒速加载。

2. **若需完整功能**，使用 `app.py` 但**禁用语义检测**：
   - 启动 `streamlit run app.py`
   - 在侧边栏**取消勾选**"启用语义相似度检测"
   - 只启用规则匹配和分类器，快速完成

3. **环境要求检查**：
   - 确保网络连接良好（首次会下载模型）
   - 建议至少 2GB 可用内存
   - GPU 可选但不必需（CPU 也能运行）

说明：
- 项目已移除图片 OCR 支持；Web 应用仅支持文本输入。
- 语义模型默认使用 `paraphrase-multilingual-MiniLM-L12-v2`，可以在 `utils/semantic.py` 中替换为其他 Hugging Face 模型。

Web 应用特性：
- 支持文本直接输入（图片/OCR 功能已移除）。
- 多层级检测：违规规则匹配、敏感词正则匹配、语义相似度检测、微调分类器预测。
- 实时显示详细检测结果（标签页分类）。
- 支持下载完整检测报告（JSON 格式）。
- 侧边栏配置项：调整语义阈值、启用/禁用各检测模块。
- **新增功能**：
  - **数据库历史存储**：自动保存所有检测报告到 `reports.db` SQLite 数据库。
  - **历史报告查询**：查看历史检测数据、统计信息、下载单条报告或批量删除。
  - **规则管理**：在 Web 界面直接编辑、上传违规规则和敏感词规则文件。
  - **批量检测**：支持上传文本文件（TXT/CSV，每行一条）进行批量检测，生成统计报告并下载。

训练 classifier（示例）

1. 准备数据：在 `data/train.csv` 和 `data/val.csv` 中准备两列 `text,label`（label: 0 合规，1 违规）。示例数据已包含在仓库中。

2. 创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. 运行训练（使用 `bert-base-chinese` 或 `uer/bert-base-chinese-cluecorpussmall`）：

```powershell
python train_classifier.py --model_name uer/bert-base-chinese-cluecorpussmall --epochs 3 --batch_size 8
```

训练完成后模型会保存在 `models/violation_classifier`，可以通过 `utils/classifier.py` 中的 `predict_text()` 进行推理。

示例推理：

```python
from utils.classifier import predict_text
print(predict_text("本品可以根治所有疾病，百分百有效"))
```

注意：训练需要 GPU 才能快速完成，CPU 下训练会较慢。根据数据量与需求调整 `epochs`、`batch_size` 与 `max_length`。

下一步建议：
- 将敏感规则抽象为可编辑的 JSON/数据库配置并提供管理界面。
- 为语义检测准备更多的敏感示例并可选地在本地微调分类器以提高准确性。
- 在生产环境部署 API 服务（FastAPI）或扩展 Web 应用功能。

#### 新增功能详解

##### 1. 数据库历史存储（`utils/database.py`）

-- 自动将所有检测报告保存到 `reports.db`（SQLite）。
-- 字段包括：检测文本、判定结果、违规数量、完整报告 JSON、创建时间。（注：`ocr_text` 为历史字段，当前不使用）
- 提供统计接口：查询总检测数、合规数、违规数等。

##### 2. 历史报告查询（`历史报告` 页面）

- 查看最近 100 条检测历史。
- 快速预览：文本摘要（前 60 字）、判定结果、违规数量。
- 点击"👁️ 查看"展开完整报告详情（JSON 格式）。
- 下载单条报告或删除历史记录。

##### 3. 规则管理（`规则管理` 页面）

- **编辑规则**：直接在文本框中编辑 `violation_rules.txt` 或 `sensitive_words.txt`。
- **上传规则**：通过文件上传替换规则文件。
- **实时预览**：显示当前规则总数与前 20 条规则内容。
- 修改即时保存，无需重启应用。

##### 4. 批量检测（`批量检测` 页面）

- 上传 TXT 或 CSV 文件（每行一条待检测文本）。
- 实时显示处理进度条。
- 逐条检测并汇总统计：总数、合规数、违规数。
- 下载批量结果（JSON 格式）。

#### 文件结构

```
医疗广告合规/
├── app.py                              # Streamlit 主应用（新增页面：批量检测、历史报告、规则管理）
├── main.py                             # 命令行工具
├── train_classifier.py                 # 分类器微调脚本
├── config.py                           # 配置文件
├── requirements.txt                    # 依赖（新增 streamlit）
├── violation_rules.txt                 # 违规规则库
├── sensitive_words.txt                 # 敏感词规则库
├── reports.db                          # SQLite 数据库（自动生成）
├── data/
│   ├── train.csv                       # 训练数据集
│   └── val.csv                         # 验证数据集
├── models/
│   └── violation_classifier/           # 微调后的模型（训练后生成）
├── utils/
│   ├── ocr_removed.py                  # OCR 模块（已归档，项目不再支持 OCR）
│   ├── text_processing.py              # 文本处理（分词、正则匹配）
│   ├── semantic.py                     # 语义相似度（Sentence-Transformers）
│   ├── classifier.py                   # 分类器推理
│   └── database.py                     # 数据库操作（新增）
└── reports/                            # 检测报告输出目录
```
