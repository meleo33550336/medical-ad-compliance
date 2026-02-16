from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import numpy as np

_model = None
_tokenizer = None
_device = None


def load_classifier(model_dir='models/violation_classifier'):
    global _model, _tokenizer, _device
    if _model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        _model = BertForSequenceClassification.from_pretrained(model_dir)
        _model.to(_device)
        _model.eval()
    return _model, _tokenizer, _device


def predict_text(text, model_dir='models/violation_classifier', max_length=256):
    """返回 dict: { 'label': 0 or 1, 'score': float, 'probs': [p0, p1] }"""
    model, tokenizer, device = load_classifier(model_dir)
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        probs = _softmax(logits)
        label = int(np.argmax(probs))
        score = float(probs[label])
    return {'label': label, 'score': score, 'probs': [float(p) for p in probs]}


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
