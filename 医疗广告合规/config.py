# 配置项
import os

SENSITIVE_WORDS_FILE = "sensitive_words.txt"
# 正则匹配是否使用忽略大小写（默认不忽略），可以设置为 re.IGNORECASE
REGEX_FLAGS = 0
# 语义相似度阈值（0-1），超过则标记为疑似敏感
SEMANTIC_THRESHOLD = 0.65
# 报告输出目录
REPORT_DIR = "reports"

#（已移除）OCR/Tesseract 相关配置
