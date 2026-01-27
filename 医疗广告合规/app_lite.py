"""
轻量级版本 - 仅使用快速规则检测，不加载语义模型和分类器
这个版本应该能秒速加载和检测
"""
import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import tempfile

from utils.ocr import ocr_from_image
from utils.text_processing import tokenize, regex_matches, violation_matches, load_sensitive_words, load_violation_rules
from utils.database import init_db, save_report, get_reports, get_report_by_id, delete_report, get_statistics
import config

# 缓存规则加载
@st.cache_data
def load_rules():
    return {
        'sensitive': load_sensitive_words('sensitive_words.txt'),
        'violation': load_violation_rules('violation_rules.txt')
    }

@st.cache_resource
def init_db_once():
    init_db()


def highlight_text(text, matches):
    """将匹配的位置标红。
    
    匹配格式：[{'span': [start, end], 'matched_text': '...', ...}, ...]
    """
    if not matches:
        return text
    
    # 按起始位置倒序排列，从后往前替换以避免位置偏移
    sorted_matches = sorted(matches, key=lambda x: x['span'][0], reverse=True)
    
    result = text
    for match in sorted_matches:
        start, end = match['span']
        matched_text = match['matched_text']
        # 用红色高亮替换
        highlighted = f'<span style="background-color: #ffcccc; color: red; font-weight: bold;">{matched_text}</span>'
        result = result[:start] + highlighted + result[end:]
    
    return result


st.set_page_config(page_title='医疗广告合规检测系统', layout='wide')
st.title('🏥 医疗广告合规检测系统 (轻量版)')
st.markdown('---')

init_db_once()

page = st.sidebar.radio('选择功能', ('快速检测', '批量检测', '历史报告', '规则管理'))

with st.sidebar:
    st.header('⚙️ 配置')
    st.caption('本版本仅使用规则检测（快速），不加载机器学习模型。')

if page == '快速检测':
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📝 输入')
        input_text = st.text_area('输入文本', height=300, placeholder='输入医疗广告文案...')
    
    with col2:
        st.subheader('📊 结果')
        
        if st.button('🔍 检测', key='detect'):
            if input_text:
                with st.spinner('检测中...'):
                    rules = load_rules()
                    tokens = tokenize(input_text)
                    regex_res = regex_matches(input_text, rules['sensitive'])
                    violation_res = violation_matches(input_text, rules=rules['violation'])
                    
                    has_violations = len(violation_res) > 0 or len(regex_res) > 0
                    verdict = '合规' if not has_violations else '疑似违规'
                    verdict_display = '🟢 合规' if not has_violations else '🔴 疑似违规'
                    
                    st.markdown(f"### {verdict_display}")
                    
                    tab1, tab2, tab3 = st.tabs(['违规规则', '敏感词', '分词'])
                    
                    with tab1:
                        if violation_res:
                            st.markdown(f"**发现 {len(violation_res)} 个违规规则匹配**")
                            highlighted_v = highlight_text(input_text, violation_res)
                            st.markdown(highlighted_v, unsafe_allow_html=True)
                            st.divider()
                            st.caption('**匹配详情：**')
                            for idx, m in enumerate(violation_res, 1):
                                st.caption(f"{idx}. 规则：`{m['rule']}`")
                        else:
                            st.info('✓ 无')
                    
                    with tab2:
                        if regex_res:
                            st.markdown(f"**发现 {len(regex_res)} 个敏感词匹配**")
                            highlighted_r = highlight_text(input_text, regex_res)
                            st.markdown(highlighted_r, unsafe_allow_html=True)
                            st.divider()
                            st.caption('**匹配详情：**')
                            for idx, m in enumerate(regex_res, 1):
                                st.caption(f"{idx}. 敏感词：`{m['word']}`")
                        else:
                            st.info('✓ 无')
                    
                    with tab3:
                        st.caption(' / '.join(tokens))
                    
                    st.markdown('---')
                    
                    report = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'input_text': input_text,
                        'tokens': tokens,
                        'regex_matches': regex_res,
                        'violation_rule_matches': violation_res,
                        'verdict': verdict
                    }
                    
                    col_save, col_download = st.columns(2)
                    with col_save:
                        if st.button('💾 保存'):
                            save_report(report)
                            st.success('✅ 已保存')
                    with col_download:
                        st.download_button(
                            '📥 下载',
                            json.dumps(report, ensure_ascii=False, indent=2),
                            f'report_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json',
                            'application/json'
                        )

elif page == '批量检测':
    st.subheader('📦 批量检测')
    uploaded = st.file_uploader('上传文件（每行一条）', type=['txt', 'csv'])
    
    if uploaded and st.button('🔍 检测'):
        lines = [l.strip() for l in uploaded.read().decode('utf-8').split('\n') if l.strip()]
        st.info(f'检测 {len(lines)} 条')
        
        results = []
        rules = load_rules()
        progress = st.progress(0)
        
        for idx, text in enumerate(lines):
            regex_res = regex_matches(text, rules['sensitive'])
            violation_res = violation_matches(text, rules=rules['violation'])
            has_v = len(violation_res) > 0 or len(regex_res) > 0
            results.append({
                'text': text[:100],
                'verdict': '违规' if has_v else '合规',
                'count': len(violation_res) + len(regex_res)
            })
            progress.progress((idx + 1) / len(lines))
        
        st.dataframe(results)
        st.download_button(
            '📥 下载',
            json.dumps(results, ensure_ascii=False, indent=2),
            f'batch_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json',
            'application/json'
        )

elif page == '历史报告':
    st.subheader('📜 历史')
    stats = get_statistics()
    col1, col2, col3 = st.columns(3)
    col1.metric('总数', stats['total'])
    col2.metric('违规', stats['violations'])
    col3.metric('合规', stats['compliant'])
    
    reports = get_reports(limit=20)
    if reports:
        for report_id, ts, text, verdict, count, _ in reports:
            icon = '🟢' if verdict == '合规' else '🔴'
            st.write(f"{icon} {text[:50]}... **{verdict}**")
            if st.button('删除', key=f'd_{report_id}'):
                delete_report(report_id)
                st.rerun()

elif page == '规则管理':
    st.subheader('⚙️ 管理规则')
    rule_type = st.radio('类型', ('违规规则', '敏感词'))
    file_path = Path('violation_rules.txt' if rule_type == '违规规则' else 'sensitive_words.txt')
    
    content = file_path.read_text(encoding='utf-8') if file_path.exists() else ''
    new_content = st.text_area('规则内容', content, height=300)
    
    if st.button('💾 保存'):
        file_path.write_text(new_content, encoding='utf-8')
        st.cache_data.clear()
        st.success('✅ 已保存')

st.markdown('---')
st.caption('轻量版 - 仅规则检测，秒速加载')
