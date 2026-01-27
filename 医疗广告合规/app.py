import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import tempfile

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
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizerFast.from_pretrained('models/violation_classifier')
        model = BertForSequenceClassification.from_pretrained('models/violation_classifier')
        model.to(device)
        model.eval()
        return {'model': model, 'tokenizer': tokenizer, 'device': device}
    except Exception:
        return None


@st.cache_data
def load_rules_cached():
    """缓存规则文件加载。"""
    from utils.text_processing import load_sensitive_words, load_violation_rules
    return {
        'sensitive': load_sensitive_words('sensitive_words.txt'),
        'violation': load_violation_rules('violation_rules.txt')
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
        # 用红色高亮替换
        highlighted = f'<span style="background-color: #ffcccc; color: red; font-weight: bold;">{matched_text}</span>'
        result = result[:start] + highlighted + result[end:]
    
    return result


# 导入其他模块
from utils.ocr import ocr_from_image
from utils.text_processing import tokenize, regex_matches, violation_matches
from utils.database import init_db, save_report, get_reports, get_report_by_id, delete_report, get_statistics
import config

# ============ 初始化 ============
st.set_page_config(page_title='医疗广告合规检测系统', layout='wide', initial_sidebar_state='expanded')

@st.cache_resource
def init_database():
    """初始化数据库（仅一次）。"""
    init_db()
    return True

init_database()

st.title('🏥 医疗广告合规检测系统')
st.markdown('---')

# ============ 侧边栏 ============
page = st.sidebar.radio('选择功能', ('检测', '批量检测', '历史报告', '规则管理'))

with st.sidebar:
    if page != '规则管理':
        st.header('⚙️ 检测配置')
        semantic_threshold = st.slider('语义相似度阈值', 0.0, 1.0, config.SEMANTIC_THRESHOLD, 0.01)
        enable_classifier = st.checkbox('启用微调分类器', value=False)  # 默认关闭以加速
        enable_violation_rules = st.checkbox('启用违规规则匹配', value=True)
        enable_semantic = st.checkbox('启用语义相似度检测', value=False)  # 默认关闭以加速
    
    st.markdown('---')
    st.markdown('⏱️ **性能优化**')
    st.caption('为了加速首次加载，已默认关闭'
               '语义检测和分类器。请在需要时启用。')


# ============ 检测页面 ============
if page == '检测':
    detection_mode = st.radio('选择检测类型', ('文本检测', '图片OCR检测'))
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('📝 输入')
        
        if detection_mode == '文本检测':
            input_text = st.text_area('请输入要检测的文本', height=300, placeholder='输入医疗广告文案...')
            ocr_text = ''
        else:
            st.markdown('**上传图片进行 OCR 识别**')
            uploaded_file = st.file_uploader('选择图片文件', type=['jpg', 'jpeg', 'png', 'bmp'])
            input_text = ''
            ocr_text = ''
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='上传的图片', use_column_width=True)
                
                if st.button('🔍 运行 OCR 识别', key='ocr_btn'):
                    with st.spinner('正在进行 OCR 识别...'):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                                tmp.write(uploaded_file.getbuffer())
                                tmp_path = tmp.name
                            ocr_text = ocr_from_image(tmp_path)
                            input_text = ocr_text
                            st.success('✅ OCR 识别完成')
                            st.text_area('OCR 识别结果', value=ocr_text, height=200, disabled=True)
                        except Exception as e:
                            st.error(f'❌ OCR 识别失败: {e}')

    with col2:
        st.subheader('📊 检测结果')
        
        if detection_mode == '文本检测' and input_text:
            perform_detection = st.button('🔍 运行检测', key='text_detect_btn')
        elif detection_mode == '图片OCR检测' and input_text:
            perform_detection = st.button('🔍 运行检测', key='image_detect_btn')
        else:
            perform_detection = False
        
        if perform_detection and input_text:
            with st.spinner('正在检测...'):
                # 1. 分词
                st.info('📊 执行分词...')
                tokens = tokenize(input_text)
                
                # 2. 加载规则
                rules_data = load_rules_cached()
                
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
                
                with tab2:
                    if regex_res:
                        st.markdown(f"**发现 {len(regex_res)} 个敏感词匹配**")
                        st.markdown('---')
                        st.markdown('**检测文本（红色标记为问题位置）：**')
                        highlighted = highlight_text(input_text, regex_res)
                        st.markdown(highlighted, unsafe_allow_html=True)
                        st.markdown('---')
                        st.markdown('**详细匹配：**')
                        for idx, match in enumerate(regex_res, 1):
                            st.write(f"**{idx}. 敏感词**: `{match['word']}`")
                            st.divider()
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
                    'ocr_text': ocr_text,
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
elif page == '批量检测':
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
                rules_data = load_rules_cached()
                
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
elif page == '历史报告':
    st.subheader('📜 检测历史')
    
    stats = get_statistics()
    col1, col2, col3 = st.columns(3)
    col1.metric('总检测数', stats['total'])
    col2.metric('违规数', stats['violations'])
    col3.metric('合规数', stats['compliant'])
    
    st.markdown('---')
    
    reports = get_reports(limit=50)
    if reports:
        st.markdown(f'**显示最近 {len(reports)} 条记录**')
        
        for report_id, timestamp, input_text, verdict, violation_count, created_at in reports:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                verdict_icon = '🟢' if verdict == '合规' else '🔴'
                st.write(f"{verdict_icon} {input_text[:60]}...")
            
            with col2:
                st.write(f"**{verdict}**")
            
            with col3:
                st.write(f"**违规数**: {violation_count}")
            
            with col4:
                if st.button('👁️ 查看', key=f'view_{report_id}'):
                    st.session_state[f'view_report_{report_id}'] = True
        
        for report_id, _, _, _, _, _ in reports:
            if st.session_state.get(f'view_report_{report_id}', False):
                st.markdown('---')
                full_report = get_report_by_id(report_id)
                if full_report:
                    st.json(full_report)
                    col1, col2 = st.columns(2)
                    with col1:
                        report_json = json.dumps(full_report, ensure_ascii=False, indent=2)
                        st.download_button(
                            label='📥 下载此报告',
                            data=report_json,
                            file_name=f'report_{report_id}.json',
                            mime='application/json',
                            key=f'download_{report_id}'
                        )
                    with col2:
                        if st.button('🗑️ 删除', key=f'delete_{report_id}'):
                            delete_report(report_id)
                            st.success('✅ 报告已删除')
                            st.rerun()
                
                if st.button('关闭详情', key=f'close_{report_id}'):
                    st.session_state[f'view_report_{report_id}'] = False
                    st.rerun()
    else:
        st.info('📭 暂无检测历史')

# ============ 规则管理页面 ============
elif page == '规则管理':
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
