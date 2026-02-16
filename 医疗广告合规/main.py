import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from utils.text_processing import load_sensitive_words, tokenize, regex_matches, violation_matches
from utils.semantic import load_model, semantic_matches
from utils.classifier import predict_text
import config


def build_report(input_text, ocr_text, tokens, regex_res, semantic_res, output_path=None):
    verdict = '合规'
    # 如果任一检测模块给出疑似违规判定，则整体判为疑似违规
    if (regex_res and len(regex_res) > 0) or (semantic_res and len(semantic_res) > 0):
        verdict = '疑似违规'
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'input_text': input_text,
        'ocr_text': ocr_text,
        'tokens': tokens,
        'regex_matches': regex_res,
        'violation_rule_matches': [],
        'semantic_matches': semantic_res,
        'classifier': None,
        'verdict': verdict
    }
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='医疗广告合规检测 简易版')
    parser.add_argument('--input', '-i', required=True, help='输入文本或文本文件路径')
    parser.add_argument('--report', '-r', default=None, help='报告输出路径（默认 reports/report_<ts>.json）')
    parser.add_argument('--semantic-threshold', type=float, default=config.SEMANTIC_THRESHOLD)
    
    args = parser.parse_args()

    input_path = args.input
    ocr_text = ''
    input_text = ''
    # 直接读取文本或把参数当作文本内容
    if os.path.exists(input_path):
        input_text = Path(input_path).read_text(encoding='utf-8')
    else:
        input_text = input_path

    # 文本处理
    sensitive_list = load_sensitive_words(config.SENSITIVE_WORDS_FILE)
    tokens = tokenize(input_text)
    regex_res = regex_matches(input_text, sensitive_list, flags=config.REGEX_FLAGS)

    # 基于用户提供的违规规则文件进行严格规则匹配
    violation_rules_file = 'violation_rules.txt'
    violation_res = violation_matches(input_text, rules_file=violation_rules_file)

    # 语义检测：使用敏感词作为示例进行相似度比对
    model = load_model()
    semantic_res = semantic_matches(input_text, sensitive_list, model=model, threshold=args.semantic_threshold)

    # 另外使用已微调的分类器（若存在）进行预测
    classifier_res = None
    try:
        classifier_res = predict_text(input_text)
    except Exception:
        classifier_res = None

    # 输出报告
    if args.report:
        out_path = args.report
    else:
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out_dir = Path(config.REPORT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'report_{ts}.json'

    report = build_report(input_text, ocr_text, tokens, regex_res, semantic_res, output_path=out_path)
    # 在报告中填入违规规则匹配和分类器结果
    report['violation_rule_matches'] = violation_res
    report['classifier'] = classifier_res
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
