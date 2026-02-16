import streamlit as st
import json
import time
import time
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import plotly.express as px
# 在最开始使用缓存装饰器
@st.cache_resource
def get_semantic_model_cached():
    """缓存语义模型，仅加载一次。"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


@st.cache_resource
def get_classifier_cached():
    """缓存分类器模型，仅加载一次。"""
    try:
        from transformers import BertTokenizerFast, BertForSequenceClassification
        import tor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizerFast.from_pretrained('models/violation_classifier')
        model = BertForSequenceClassification.from_pretrained('models/violation_classifier')
        model.to(device)
        model.eval()
        return {'model': model, 'tokenizer': tokenizer, 'device': device}
    except Exception:
        return None


@st.cache_data
def load_rules_cached(lang='中文'):
    """按语言缓存规则文件加载。支持 '中文' 与 'English'，若英文文件不存在则回退到默认文件。"""
    from utils.text_processing import load_sensitive_words, load_violation_rules
    if lang == 'English':
        s_file = 'sensitive_words_en.txt'
        v_file = 'violation_rules_en.txt'
    else:
        s_file = 'sensitive_words.txt'
        v_file = 'violation_rules.txt'

    # 回退逻辑
    if not Path(s_file).exists():
        s_file = 'sensitive_words.txt'
    if not Path(v_file).exists():
        v_file = 'violation_rules.txt'

    return {
        'sensitive': load_sensitive_words(s_file),
        'violation': load_violation_rules(v_file)
    }


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
        # 使用霓虹样式高亮（违规使用粉色，敏感词使用青绿色）
        # 默认使用粉色风格；调用方可以替换为其他类名
        highlighted = f'<span class="neon-match">{matched_text}</span>'
        result = result[:start] + highlighted + result[end:]
    
    return result


# 导入其他模块
from utils.text_processing import tokenize, regex_matches, violation_matches
from utils.database import init_db, save_report, get_reports, get_report_by_id, delete_report, get_statistics
import config

# ============ 初始化 ============
st.set_page_config(page_title='医疗广告合规检测系统', layout='wide', initial_sidebar_state='expanded', page_icon='🛰️')

# --- 视觉样式注入（科技感 / 霓虹 + 玻璃材质）
st.markdown(
        """
        <style>
        :root{--bg1:#071021;--bg2:#081126;--accent:#00ffd5;--accent2:#7c3aed;--muted:#9fb3c8}
        body, [data-testid='stAppViewContainer']{background: linear-gradient(135deg,var(--bg1),var(--bg2)) !important; color: #cfe9f8}
        .app-header{padding:18px;border-radius:12px;margin-bottom:12px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 8px 30px rgba(2,6,23,0.7); backdrop-filter: blur(6px); border:1px solid rgba(255,255,255,0.03)}
        .app-title{font-size:28px; color: var(--accent); font-weight:700; letter-spacing:1px; text-shadow:0 0 18px rgba(0,255,213,0.08)}
        .app-sub{color:var(--muted); margin-top:4px}
        .card{background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:14px; border-radius:10px; border:1px solid rgba(255,255,255,0.03)}
        .stButton>button {background: linear-gradient(90deg,var(--accent),var(--accent2)) !important; color: #04111a !important; font-weight:700; border-radius:8px !important; padding:8px 12px; box-shadow:0 6px 18px rgba(124,58,237,0.12)}
        textarea, input, .stTextInput>div>input {background: rgba(255,255,255,0.02) !important; color: #e6f7ff !important; border-radius:8px !important}
        .neon-match{background: rgba(255,20,147,0.10); color:#ff4d9e; font-weight:700; padding:2px 4px; border-radius:4px}
        .neon-sensitive{background: rgba(0,255,213,0.06); color:var(--accent); font-weight:700; padding:2px 4px; border-radius:4px}
        /* 卡片炫彩边框 */
        .fancy-card{padding:14px; border-radius:12px; background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.008)); border:1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 30px rgba(2,6,23,0.6); transition: transform .18s ease, box-shadow .18s ease}
        .fancy-card:hover{transform: translateY(-6px); box-shadow: 0 18px 40px rgba(124,58,237,0.16);}
        .fancy-card .title{font-weight:700; color:var(--accent);}
        .fancy-card .meta{color:var(--muted); font-size:12px}
        /* 减少侧边栏亮度 */
        [data-testid='stSidebar']{background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.008)) !important; border-right:1px solid rgba(255,255,255,0.02)}
        </style>

        <div class="app-header">
            <div class="app-title">🏥 医疗广告合规检测系统</div>
            <div class="app-sub">创新 · 快速 · 科技感视觉</div>
        </div>
        """,
        unsafe_allow_html=True,
)

# 语言支持文本
_I18N = {
    '中文': {
        'detect':'检测', 'batch':'批量检测', 'history':'历史报告', 'rules':'规则管理',
        'title':'🏥 医疗广告合规检测系统', 'input_placeholder':'输入医疗广告文案...',
        'run_detect':'🔍 运行检测', 'save':'💾 保存', 'download':'📥 下载', 'no_history':'📭 暂无检测历史',
        'settings':'⚙️ 检测配置', 'demo':'🎛️ 演示动效'
    },
    'English': {
        'detect':'Detect', 'batch':'Batch', 'history':'History', 'rules':'Rules',
        'title':'Medical Ad Compliance Checker', 'input_placeholder':'Enter advertisement text...',
        'run_detect':'🔍 Run Detection', 'save':'💾 Save', 'download':'📥 Download', 'no_history':'No history yet',
        'settings':'⚙️ Settings', 'demo':'🎛️ Demo Anim'
    }
}

def t(lang, key):
    return _I18N.get(lang, _I18N['中文']).get(key, key)

@st.cache_resource
def init_database():
    """初始化数据库（仅一次）。"""
    init_db()
    return True

init_database()

st.title('🏥 医疗广告合规检测系统')
st.markdown('---')

# ============ 侧边栏 ============
with st.sidebar:
    # 语言选择器（中文 / English）
    lang = st.selectbox('语言 / Language', ('中文', 'English'))

    # 页面选项：使用内部 keys 并显示本地化标签
    pages = {
        'detect': t(lang, 'detect'),
        'batch': t(lang, 'batch'),
        'history': t(lang, 'history'),
        'rules': t(lang, 'rules')
    }

    page = st.radio('选择功能' if lang == '中文' else 'Select', options=list(pages.keys()), format_func=lambda k: pages[k])

    if page != 'rules':
        st.header(t(lang, 'settings'))
        semantic_threshold = st.slider('语义相似度阈值', 0.0, 1.0, config.SEMANTIC_THRESHOLD, 0.01)
        enable_classifier = st.checkbox('启用微调分类器', value=False)  # 默认关闭以加速
        enable_violation_rules = st.checkbox('启用违规规则匹配', value=True)
        enable_semantic = st.checkbox('启用语义相似度检测', value=False)  # 默认关闭以加速
    # 动效演示按钮（本地化）
    if st.button(t(lang, 'demo')):
        def run_animation():
            demo_cols = st.columns(3)
            t1 = demo_cols[0].empty()
            t2 = demo_cols[1].empty()
            t3 = demo_cols[2].empty()
            # 演示计数动画
            for i in range(0, 101, 5):
                t1.metric('示例检测数', f'{i}')
                t2.metric('示例违规数', f'{int(i*0.35)}')
                t3.metric('示例合规数', f'{int(i*0.65)}')
                time.sleep(0.04)
            # 显示短暂扫描条
            scan = st.empty()
            scan.markdown("<div class='scanner' style='height:6px;border-radius:6px;margin-top:8px;'></div>", unsafe_allow_html=True)
            time.sleep(1.2)
            scan.empty()
        run_animation()

    st.markdown('---')
    st.markdown('⏱️ **性能优化**')
    st.caption('为了加速首次加载，已默认关闭'
               '语义检测和分类器。请在需要时启用。')


# ============ 检测页面 ============
if page == 'detect':
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('📝 输入')
        input_text = st.text_area('请输入要检测的文本', height=300, placeholder='输入医疗广告文案...')

    with col2:
        st.subheader('📊 检测结果')
        perform_detection = False
        if input_text:
            perform_detection = st.button('🔍 运行检测', key='text_detect_btn')
        
        if perform_detection and input_text:
            with st.spinner('正在检测...'):
                # 1. 分词
                st.info('📊 执行分词...')
                tokens = tokenize(input_text)
                
                    # 2. 加载规则（按语言）
                    rules_data = load_rules_cached(lang=lang)
                # 如果两类规则均为空，提示可能的文件路径问题
                if not rules_data.get('sensitive') and not rules_data.get('violation'):
                    proj_root = Path(__file__).resolve().parent
                    st.warning(f'⚠️ 未加载到规则文件（或文件为空）。请确认以下文件存在并包含规则：\n- {proj_root / "violation_rules.txt"}\n- {proj_root / "sensitive_words.txt"}')
                
                # 3. 敏感词正则匹配
                st.info('🔍 执行敏感词匹配...')
                regex_res = regex_matches(input_text, rules_data['sensitive'], flags=config.REGEX_FLAGS)
                
                # 4. 违规规则匹配
                violation_res = []
                if enable_violation_rules:
                    st.info('📋 执行违规规则匹配...')
                    violation_res = violation_matches(input_text, rules=rules_data['violation'])
                
                # 5. 语义相似度检测
                semantic_res = []
                if enable_semantic:
                    st.info('⏳ 加载语义模型中（首次较慢）...')
                    try:
                        model = get_semantic_model_cached()
                        st.info('🔄 执行语义相似度检测...')
                        from numpy import argmax
                        from sklearn.metrics.pairwise import cosine_similarity
                        texts = [input_text] + rules_data['sensitive']
                        embs = model.encode(texts, convert_to_numpy=True)
                        sims = cosine_similarity(embs[0:1], embs[1:])[0]
                        for i, s in enumerate(sims):
                            if s >= semantic_threshold:
                                semantic_res.append({'example': rules_data['sensitive'][i], 'score': float(s)})
                        semantic_res = sorted(semantic_res, key=lambda x: x['score'], reverse=True)[:3]
                    except Exception as e:
                        st.warning(f'⚠️ 语义检测失败: {e}')
                
                # 6. 分类器预测
                classifier_res = None
                if enable_classifier:
                    st.info('⏳ 加载分类器中（首次较慢）...')
                    try:
                        import torch
                        import numpy as np
                        clf = get_classifier_cached()
                        if clf:
                            tokenizer, model, device = clf['tokenizer'], clf['model'], clf['device']
                            inputs = tokenizer(input_text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            with torch.no_grad():
                                outputs = model(**inputs)
                                logits = outputs.logits.cpu().numpy()[0]
                                e_x = np.exp(logits - np.max(logits))
                                probs = e_x / e_x.sum()
                                label = int(argmax(probs))
                                classifier_res = {'label': label, 'score': float(probs[label]), 'probs': [float(p) for p in probs]}
                    except Exception as e:
                        st.warning(f'⚠️ 分类器加载失败: {e}')
                
                # 7. 综合判定
                has_violations = len(violation_res) > 0 or len(regex_res) > 0 or len(semantic_res) > 0
                if classifier_res and classifier_res.get('label') == 1:
                    has_violations = True
                
                verdict = '合规' if not has_violations else '疑似违规'
                verdict_display = '🟢 合规' if not has_violations else '🔴 疑似违规'
                
                st.markdown(f"### 综合判定: {verdict_display}")
                
                # 结果标签页
                tab1, tab2, tab3, tab4, tab5 = st.tabs(['违规规则匹配', '敏感词匹配', '语义相似度', '分类器', '分词结果'])
                
                with tab1:
                    if enable_violation_rules and violation_res:
                        st.markdown(f"**发现 {len(violation_res)} 个违规规则匹配**")
                        st.markdown('---')
                        st.markdown('**检测文本（红色标记为问题位置）：**')
                        highlighted = highlight_text(input_text, violation_res)
                        st.markdown(highlighted, unsafe_allow_html=True)
                        st.markdown('---')
                        st.markdown('**详细匹配：**')
                        for idx, match in enumerate(violation_res, 1):
                            st.write(f"**{idx}. 违规词**: `{match['rule']}`")
                            st.divider()
                    else:
                        st.info('✅ 未检测到违规规则匹配')
                
                        :root{--bg1:#e8f7ff;--bg2:#f4fbff;--accent:#0077ff;--accent2:#00c2ff;--muted:#4b6b80}
                        body, [data-testid='stAppViewContainer']{background: linear-gradient(135deg,var(--bg1),var(--bg2)) !important; color: #08233b}
                        .app-header{padding:18px;border-radius:12px;margin-bottom:12px; background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(255,255,255,0.95)); box-shadow: 0 6px 18px rgba(10,30,60,0.08); backdrop-filter: blur(6px); border:1px solid rgba(10,30,60,0.04)}
                        .app-title{font-size:28px; color: var(--accent); font-weight:700; letter-spacing:1px}
                        .app-sub{color:var(--muted); margin-top:4px}
                        .card{background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,255,255,0.98)); padding:14px; border-radius:10px; border:1px solid rgba(10,30,60,0.04)}
                        .stButton>button {background: linear-gradient(90deg,var(--accent),var(--accent2)) !important; color: #ffffff !important; font-weight:700; border-radius:8px !important; padding:8px 12px; box-shadow:0 6px 18px rgba(0,124,255,0.12)}
                        textarea, input, .stTextInput>div>input {background: #ffffff !important; color: #08233b !important; border-radius:8px !important; border:1px solid rgba(10,30,60,0.06)}
                        .neon-match{background: rgba(255,230,240,0.6); color:#b30052; font-weight:700; padding:2px 4px; border-radius:4px}
                        .neon-sensitive{background: rgba(220,255,250,0.6); color:#0077ff; font-weight:700; padding:2px 4px; border-radius:4px}
                        /* 动画效果 */
                        @keyframes pulse {0% {filter: drop-shadow(0 0 0 rgba(0,124,255,0.0)); transform: translateY(0)} 50% {filter: drop-shadow(0 0 18px rgba(0,124,255,0.12)); transform: translateY(-4px)} 100% {filter: drop-shadow(0 0 0 rgba(0,124,255,0.0)); transform: translateY(0)}}
                        .glow {animation: pulse 1.6s infinite ease-in-out}
                        @keyframes scanner {0% {background-position: -200px 0;} 100% {background-position: 200px 0;}}
                        .scanner {background: linear-gradient(90deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.12) 50%, rgba(255,255,255,0.02) 100%); background-size: 200% 100%; animation: scanner 1.6s linear infinite}
                        /* 侧边栏更轻的背景 */
                        [data-testid='stSidebar']{background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(250,255,255,0.98)) !important; border-right:1px solid rgba(10,30,60,0.04)}
                    else:
                        st.info('✅ 未检测到敏感词')
                
                with tab3:
                    if enable_semantic and semantic_res:
                        st.markdown(f"**发现 {len(semantic_res)} 个相似度高的示例**")
                        for idx, match in enumerate(semantic_res, 1):
                            st.write(f"**{idx}. 示例**: `{match['example']}`")
                            st.write(f"   **相似度分数**: {match['score']:.4f}")
                            st.divider()
                    elif enable_semantic:
                        st.info('✅ 未检测到高相似度示例')
                    else:
                        st.info('⏭️ 语义检测已禁用')
                
                with tab4:
                    if classifier_res:
                        label_text = '疑似违规 (label=1)' if classifier_res['label'] == 1 else '合规 (label=0)'
                        st.write(f"**分类结果**: {label_text}")
                        st.write(f"**置信度**: {classifier_res['score']:.4f}")
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            st.metric('合规概率', f"{classifier_res['probs'][0]:.4f}")
                        with col_prob2:
                            st.metric('违规概率', f"{classifier_res['probs'][1]:.4f}")
                    else:
                        st.info('⏭️ 分类器未启用或加载失败')
                
                with tab5:
                    st.write('**分词结果**')
                    tokens_str = ' / '.join(tokens)
                    st.text(tokens_str)
                    st.write(f'**总词数**: {len(tokens)}')
                
                # 生成报告
                st.markdown('---')
                report_data = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'input_text': input_text,
                    'ocr_text': '',
                    'tokens': tokens,
                    'regex_matches': regex_res,
                    'violation_rule_matches': violation_res,
                    'semantic_matches': semantic_res,
                    'classifier': classifier_res,
                    'verdict': verdict
                }
                
                col_save, col_download = st.columns(2)
                with col_save:
                    if st.button('💾 保存到历史'):
                        save_report(report_data)
                        st.success('✅ 报告已保存到数据库')
                
                with col_download:
                    report_json = json.dumps(report_data, ensure_ascii=False, indent=2)
                    st.download_button(
                        label='📥 下载报告 (JSON)',
                        data=report_json,
                        file_name=f'report_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json',
                        mime='application/json'
                    )

# ============ 批量检测页面 ============
elif page == 'batch':
    st.subheader('📦 批量检测')
    st.markdown('**上传包含文本的文件进行批量检测**')
    
    uploaded_file = st.file_uploader('选择文本文件（每行一条）', type=['txt', 'csv'])
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            st.info(f'📊 检测到 {len(lines)} 条文本')
            
            if st.button('🔍 开始批量检测'):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                rules_data = load_rules_cached(lang=lang)
                
                for idx, text in enumerate(lines):
                    status_text.text(f'处理中 {idx + 1}/{len(lines)}...')
                    
                    try:
                        regex_res = regex_matches(text, rules_data['sensitive'])
                        violation_res = []
                        if enable_violation_rules:
                            violation_res = violation_matches(text, rules=rules_data['violation'])
                        
                        has_violations = len(violation_res) > 0 or len(regex_res) > 0
                        verdict = '疑似违规' if has_violations else '合规'
                        
                        results.append({
                            'text': text[:100],
                            'verdict': verdict,
                            'violation_count': len(violation_res) + len(regex_res)
                        })
                    except Exception:
                        results.append({
                            'text': text[:100],
                            'verdict': '错误',
                            'violation_count': 0
                        })
                    
                    progress_bar.progress((idx + 1) / len(lines))
                
                status_text.empty()
                st.success(f'✅ 批量检测完成')
                
                st.subheader('检测结果')
                st.dataframe(results)
                
                compliant = sum(1 for r in results if r['verdict'] == '合规')
                violations = sum(1 for r in results if r['verdict'] == '疑似违规')
                
                col1, col2, col3 = st.columns(3)
                col1.metric('总数', len(results))
                col2.metric('合规', compliant)
                col3.metric('违规', violations)
                
                result_json = json.dumps(results, ensure_ascii=False, indent=2)
                st.download_button(
                    label='📥 下载批量结果 (JSON)',
                    data=result_json,
                    file_name=f'batch_results_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json',
                    mime='application/json'
                )
        except Exception as e:
            st.error(f'❌ 文件读取失败: {e}')

# ============ 历史报告页面 ============
elif page == 'history':
    st.subheader('📜 检测历史')
    
    stats = get_statistics()
    c1, c2, c3 = st.columns(3)
    c1.metric('总检测数', stats['total'])
    c2.metric('违规数', stats['violations'])
    c3.metric('合规数', stats['compliant'])

    # 可视化：按判定绘制柱状图
    st.markdown('---')
    reports = get_reports(limit=100)
    verdicts = ['合规', '疑似违规']
    counts = [stats.get('compliant', 0), stats.get('violations', 0)]
    fig = px.bar(x=verdicts, y=counts, color=verdicts, color_discrete_map={
        '合规':'#00ffd5', '疑似违规':'#ff4d9e'
    }, labels={'x':'判定','y':'数量'}, title='检测判定分布')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#cfe9f8')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    if reports:
        st.markdown(f'**显示最近 {len(reports)} 条记录（卡片视图）**')
        # 以两列排版卡片
        for i in range(0, len(reports), 2):
            row = reports[i:i+2]
            cols = st.columns(len(row))
            for col, (report_id, timestamp, input_text, verdict, violation_count, created_at) in zip(cols, row):
                with col:
                    icon = '🟢' if verdict == '合规' else '🔴'
                    html = f"""
                    <div class='fancy-card'>
                      <div class='title'>{icon} {input_text[:120]}</div>
                      <div class='meta'>{timestamp} · 违规数: <strong style='color:#ff4d9e'>{violation_count}</strong></div>
                      <div style='margin-top:8px;color:var(--muted)'>判定：<strong>{verdict}</strong></div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
                    # 操作按钮
                    full_report = get_report_by_id(report_id)
                    if full_report:
                        report_json = json.dumps(full_report, ensure_ascii=False, indent=2)
                    else:
                        report_json = '{}'
                    btn_col1, btn_col2 = st.columns([1,1])
                    with btn_col1:
                        st.download_button('📥 下载', report_json, f'report_{report_id}.json', 'application/json', key=f'dl_{report_id}')
                    with btn_col2:
                        if st.button('🗑️ 删除', key=f'del_{report_id}'):
                            delete_report(report_id)
                            st.experimental_rerun()
    else:
        st.info('📭 暂无检测历史')

# ============ 规则管理页面 ============
elif page == 'rules':
    st.subheader('⚙️ 规则管理')
    
    rule_type = st.radio('选择规则类型', ('违规规则', '敏感词规则'))
    
    if rule_type == '违规规则':
        file_path = Path('violation_rules.txt')
    else:
        file_path = Path('sensitive_words.txt')
    
    st.markdown(f'**编辑: {file_path.name}**')
    
    if file_path.exists():
        current_content = file_path.read_text(encoding='utf-8')
    else:
        current_content = ''
    
    # 诊断与恢复工具
    st.markdown('**诊断与恢复**')
    diag_col1, diag_col2 = st.columns([1, 1])
    with diag_col1:
        if st.button('🔧 运行诊断'):
            msgs = []
            for fname in ['violation_rules.txt', 'sensitive_words.txt']:
                p = Path(fname)
                if p.exists():
                    size = p.stat().st_size
                    if size > 0:
                        msgs.append(f'{fname}: 存在 ({size} bytes)')
                    else:
                        msgs.append(f'{fname}: 存在，但文件为空')
                else:
                    msgs.append(f'{fname}: 不存在')

            st.info('\n'.join(msgs))

    with diag_col2:
        if st.button('🔄 恢复默认规则'):
            # 默认规则样本（简短版）
            default_violation = '''国家级
世界级
首选
包治百病
无副作用
百分之百
保证治愈
特效
速效
永久''' 
            default_sensitive = '''百分之百
保证治愈
无副作用
快速治愈
立竿见影
国家认可
权威证实
零风险
唯一疗法
长期安全'''
            try:
                Path('violation_rules.txt').write_text(default_violation, encoding='utf-8')
                Path('sensitive_words.txt').write_text(default_sensitive, encoding='utf-8')
                st.cache_data.clear()
                st.success('✅ 已恢复默认规则到 violation_rules.txt 与 sensitive_words.txt')
                # 更新 current_content 以在编辑器中显示
                current_content = Path(file_path).read_text(encoding='utf-8')
            except Exception as e:
                st.error(f'恢复失败: {e}')

    new_content = st.text_area('规则内容（每行一条）', value=current_content, height=400)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('💾 保存修改'):
            file_path.write_text(new_content, encoding='utf-8')
            st.cache_data.clear()  # 清除缓存
            st.success(f'✅ {file_path.name} 已保存')
    
    with col2:
        if st.button('🔄 重新加载'):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        uploaded_rule_file = st.file_uploader('或上传规则文件', type=['txt'])
        if uploaded_rule_file is not None:
            content = uploaded_rule_file.read().decode('utf-8')
            file_path.write_text(content, encoding='utf-8')
            st.cache_data.clear()
            st.success(f'✅ {file_path.name} 已更新')
    
    rules = [line.strip() for line in new_content.split('\n') if line.strip()]
    st.info(f'📊 当前共有 {len(rules)} 条规则')
    
    with st.expander('📋 规则预览'):
        for idx, rule in enumerate(rules[:20], 1):
            st.write(f"{idx}. {rule}")
        if len(rules) > 20:
            st.write(f"... 还有 {len(rules) - 20} 条规则")

# 页脚
st.markdown('---')
st.markdown('**医疗广告合规检测系统** | 功能：检测 | 批量检测 | 历史报告 | 规则管理')
st.markdown('💡 提示: 为加速首次加载，已默认关闭语义检测和分类器。请在需要时启用。')
