"""
轻量级版本 - 仅使用快速规则检测，不加载语义模型和分类器
这个版本应该能秒速加载和检测
"""
import streamlit as st
import json
import time
from pathlib import Path
from datetime import datetime, timezone
# 图片/OCR 功能已移除；保留为文本检测应用

from utils.text_processing import tokenize, regex_matches, violation_matches, load_sensitive_words, load_violation_rules
from utils.database import init_db, save_report, get_reports, get_report_by_id, delete_report, get_statistics
import config

# 国际化（简易）
_I18N = {
    '中文': {'detect':'快速检测', 'batch':'批量检测', 'history':'历史报告', 'rules':'规则管理', 'title':'🏥 医疗广告合规检测系统 — 轻量版', 'input_placeholder':'输入医疗广告文案...', 'run_detect':'🔍 检测', 'save':'💾 保存', 'download':'📥 下载', 'demo':'🎛️ 演示动效', 'settings':'⚙️ 配置'},
    'English': {'detect':'Quick Detect', 'batch':'Batch', 'history':'History', 'rules':'Rules', 'title':'Medical Ad Compliance — Lite', 'input_placeholder':'Enter advertisement text...', 'run_detect':'🔍 Detect', 'save':'💾 Save', 'download':'📥 Download', 'demo':'🎛️ Demo', 'settings':'⚙️ Settings'}
}

def t(lang, key):
    return _I18N.get(lang, _I18N['中文']).get(key, key)


# 缓存规则加载（按语言）
@st.cache_data
def load_rules(lang='中文'):
    if lang == 'English':
        s_file = 'sensitive_words_en.txt'
        v_file = 'violation_rules_en.txt'
    else:
        s_file = 'sensitive_words.txt'
        v_file = 'violation_rules.txt'

    if not Path(s_file).exists():
        s_file = 'sensitive_words.txt'
    if not Path(v_file).exists():
        v_file = 'violation_rules.txt'

    return {
        'sensitive': load_sensitive_words(s_file),
        'violation': load_violation_rules(v_file)
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
        highlighted = f'<span class="neon-match">{matched_text}</span>'
        result = result[:start] + highlighted + result[end:]
    
    return result


st.set_page_config(page_title='医疗广告合规检测系统', layout='wide', page_icon='🛰️')

# 视觉样式注入（与完整版保持一致）
st.markdown(
        """
        <style>
        :root{--bg1:#e8f7ff;--bg2:#f4fbff;--accent:#0077ff;--accent2:#00c2ff;--muted:#4b6b80}
        body, [data-testid='stAppViewContainer']{background: linear-gradient(135deg,var(--bg1),var(--bg2)) !important; color: #08233b}
        .app-header{padding:14px;border-radius:10px;margin-bottom:10px; background: linear-gradient(90deg, rgba(255,255,255,0.95), rgba(255,255,255,0.98)); box-shadow: 0 6px 18px rgba(10,30,60,0.06); backdrop-filter: blur(6px); border:1px solid rgba(10,30,60,0.04)}
        .app-title{font-size:22px; color: var(--accent); font-weight:700; letter-spacing:1px}
        .card{background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,255,255,0.98)); padding:12px; border-radius:10px; border:1px solid rgba(10,30,60,0.04)}
        .stButton>button {background: linear-gradient(90deg,var(--accent),var(--accent2)) !important; color: #ffffff !important; font-weight:700; border-radius:8px !important}
        textarea, input, .stTextInput>div>input {background: #ffffff !important; color: #08233b !important; border-radius:8px !important}
        .neon-match{background: rgba(255,230,240,0.6); color:#b30052; font-weight:700; padding:2px 4px; border-radius:4px}
        .neon-sensitive{background: rgba(220,255,250,0.6); color:#0077ff; font-weight:700; padding:2px 4px; border-radius:4px}
        [data-testid='stSidebar']{background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(250,255,255,0.98)) !important; border-right:1px solid rgba(10,30,60,0.04)}
        </style>

        <div class="app-header">
            <div class="app-title">🏥 医疗广告合规检测系统 — 轻量版</div>
        </div>
        """,
        unsafe_allow_html=True,
)

st.markdown('---')

init_db_once()

with st.sidebar:
    lang = st.selectbox('语言 / Language', ('中文', 'English'))
    st.header(t(lang, 'settings'))
    st.caption('本版本仅使用规则检测（快速），不加载机器学习模型。')

    pages = {'detect': t(lang, 'detect'), 'batch': t(lang, 'batch'), 'history': t(lang, 'history'), 'rules': t(lang, 'rules')}
    page = st.radio('选择功能' if lang == '中文' else 'Select', options=list(pages.keys()), format_func=lambda k: pages[k])

if page == 'detect':
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📝 输入')
        input_text = st.text_area(t(lang, 'input_placeholder'), height=300, placeholder=t(lang, 'input_placeholder'))
    
    with col2:
        st.subheader('📊 结果')
        
        if st.button(t(lang, 'run_detect'), key='detect'):
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
                    
                    tab1, tab2, tab3 = st.tabs([t(lang, 'rules'), '敏感词' if lang == '中文' else 'Sensitive', '分词' if lang == '中文' else 'Tokens'])
                    
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
                        if st.button(t(lang, 'save')):
                            save_report(report)
                            st.success('✅ 已保存')
                    with col_download:
                        st.download_button(
                            t(lang, 'download'),
                            json.dumps(report, ensure_ascii=False, indent=2),
                            f'report_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json',
                            'application/json'
                        )

elif page == 'batch':
    st.subheader('📦 批量检测')
    uploaded = st.file_uploader('上传文件（每行一条）', type=['txt', 'csv'])
    
    if uploaded and st.button(t(lang, 'run_detect')):
        lines = [l.strip() for l in uploaded.read().decode('utf-8').split('\n') if l.strip()]
        st.info(f'检测 {len(lines)} 条')
        
        results = []
        rules = load_rules(lang=lang)
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

elif page == 'history':
    st.subheader('📜 历史')
    stats = get_statistics()
    c1, c2, c3 = st.columns(3)
    c1.metric('总数', stats['total'])
    c2.metric('违规', stats['violations'])
    c3.metric('合规', stats['compliant'])

    st.markdown('---')
    # 动效演示
    if st.button(t(lang, 'demo')):
        def run_animation_lite():
            c1, c2, c3 = st.columns(3)
            p1 = c1.empty(); p2 = c2.empty(); p3 = c3.empty()
            for v in range(0, 101, 5):
                p1.metric('示例总数', v)
                p2.metric('示例违规', int(v*0.4))
                p3.metric('示例合规', int(v*0.6))
                time.sleep(0.04)
            s = st.empty()
            s.markdown("<div class='scanner' style='height:6px;border-radius:6px;margin-top:8px;'></div>", unsafe_allow_html=True)
            time.sleep(1.2)
            s.empty()
        run_animation_lite()
    reports = get_reports(limit=50)
    # 小型统计图
    try:
        import plotly.express as px
        verdicts = ['合规', '疑似违规']
        counts = [stats.get('compliant', 0), stats.get('violations', 0)]
        fig = px.pie(names=verdicts, values=counts, color=verdicts, color_discrete_map={'合规':'#00ffd5','疑似违规':'#ff4d9e'}, title='判定占比')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#cfe9f8')
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    if reports:
        for i in range(0, len(reports), 2):
            row = reports[i:i+2]
            cols = st.columns(len(row))
            for col, (report_id, ts, text, verdict, count, _) in zip(cols, row):
                with col:
                    icon = '🟢' if verdict == '合规' else '🔴'
                    html = f"""
                    <div class='fancy-card'>
                      <div class='title'>{icon} {text[:100]}</div>
                      <div class='meta'>{ts} · 违规数: <strong style='color:#ff4d9e'>{count}</strong></div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
                    full = get_report_by_id(report_id)
                    data = json.dumps(full, ensure_ascii=False, indent=2) if full else '{}'
                    cdl, cdel = st.columns([1,1])
                    with cdl:
                        st.download_button('📥 下载', data, f'report_{report_id}.json', 'application/json', key=f'dl_{report_id}')
                    with cdel:
                        if st.button('删除', key=f'del_{report_id}'):
                            delete_report(report_id)
                            st.experimental_rerun()

elif page == 'rules':
    st.subheader('⚙️ 管理规则')
    rule_type = st.radio('类型', ('违规规则', '敏感词'))
    file_path = Path('violation_rules_en.txt' if (rule_type == '违规规则' and lang == 'English' and Path('violation_rules_en.txt').exists()) else ('violation_rules.txt' if rule_type == '违规规则' else ('sensitive_words_en.txt' if lang == 'English' and Path('sensitive_words_en.txt').exists() else 'sensitive_words.txt')))
    
    content = file_path.read_text(encoding='utf-8') if file_path.exists() else ''
    new_content = st.text_area('规则内容', content, height=300)
    
    if st.button('💾 保存'):
        file_path.write_text(new_content, encoding='utf-8')
        st.cache_data.clear()
        st.success('✅ 已保存')

st.markdown('---')
st.caption('轻量版 - 仅规则检测，秒速加载')
