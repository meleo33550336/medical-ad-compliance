import jieba
import re
from pathlib import Path


def load_sensitive_words(path):
    p = Path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]


def tokenize(text):
    return list(jieba.cut_for_search(text))


def regex_matches(text, sensitive_words, flags=0):
    matches = []
    for w in sensitive_words:
        try:
            # 使用简单的词边界匹配，如果需要更复杂可在配置中扩展
            pattern = re.compile(re.escape(w), flags)
            for m in pattern.finditer(text):
                matches.append({
                    'word': w,
                    'span': [m.start(), m.end()],
                    'matched_text': m.group(0)
                })
        except re.error:
            continue
    return matches


def load_violation_rules(path):
    return load_sensitive_words(path)


def violation_matches(text, rules=None, rules_file=None, flags=re.IGNORECASE):
    """匹配违规词规则并返回所有命中项与位置。

    返回格式：[{ 'rule': <规则文本>, 'span': [start, end], 'matched_text': <实际匹配> }, ...]
    优先使用传入的 `rules` 列表；否则从 `rules_file` 加载。
    """
    if rules is None:
        if rules_file:
            rules = load_violation_rules(rules_file)
        else:
            rules = []

    results = []
    for r in rules:
        if not r:
            continue
        try:
            # 对中文/英文均采用直接匹配，忽略大小写
            pattern = re.compile(re.escape(r), flags)
            for m in pattern.finditer(text):
                results.append({
                    'rule': r,
                    'span': [m.start(), m.end()],
                    'matched_text': m.group(0)
                })
        except re.error:
            continue

    # 按匹配起始位置排序
    results.sort(key=lambda x: x['span'][0])
    return results
