from PIL import Image, ImageOps, ImageFilter
import pytesseract
import io
import base64
import os


def set_tesseract_cmd(cmd=None):
    """设置 pytesseract 的 tesseract 可执行文件路径（可选）。"""
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def _preprocess_image(img, resize_max=1600, to_grayscale=True, do_threshold=True):
    """简单的图像预处理：灰度、缩放、增强对比、二值化（可选）。"""
    # resize if too large
    w, h = img.size
    max_size = max(w, h)
    if max_size > resize_max:
        scale = resize_max / float(max_size)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    if to_grayscale:
        img = ImageOps.grayscale(img)

    # 轻微锐化
    img = img.filter(ImageFilter.SHARPEN)

    if do_threshold:
        # 简单二值化（可根据场景调参）
        img = img.point(lambda p: 0 if p < 128 else 255)

    return img


def ocr_from_image(image_path, lang='chi_sim+eng', provider='tesseract', preprocess=True, tesseract_cmd=None, api_params=None):
    """从图片提取文字。

    provider: 'tesseract'（本地 Tesseract）或 'baidu'/'aliyun'（占位模板，需要填写 api_params）。
    tesseract_cmd: 可选，传入 tesseract 可执行路径以覆盖环境配置。
    api_params: 如果使用第三方 OCR，可传入 dict（如 api_key、secret 等）。
    """
    if provider == 'tesseract':
        # 可选地指定可执行文件路径
        if tesseract_cmd:
            set_tesseract_cmd(tesseract_cmd)
        elif os.getenv('TESSERACT_CMD'):
            set_tesseract_cmd(os.getenv('TESSERACT_CMD'))

        try:
            img = Image.open(image_path)
        except Exception as e:
            raise RuntimeError(f"打开图片失败: {e}")

        if preprocess:
            try:
                img = _preprocess_image(img)
            except Exception:
                # 如果预处理失败，回退到原图
                pass

        try:
            text = pytesseract.image_to_string(img, lang=lang)
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR 识别失败: {e}")
        return text

    # 以下为第三方 OCR 的占位实现，包含调用示例提示
    if provider == 'baidu':
        return ocr_from_baidu(image_path, **(api_params or {}))
    if provider == 'aliyun' or provider == 'oss':
        return ocr_from_aliyun(image_path, **(api_params or {}))

    raise ValueError(f"未知 OCR provider: {provider}")


def ocr_from_baidu(image_path, app_id=None, api_key=None, secret_key=None):
    """占位：如何使用百度 OCR 的示例（需自行填写 credentials 并安装 requests）。

    参考流程：读取图片 base64，调用百度 OCR 接口，解析返回的 JSON。
    此处不直接实现网络调用，避免硬编码凭据。
    """
    raise NotImplementedError("请使用真实的百度 OCR SDK 或 HTTP 接口并在此处实现调用（示例已注释）")


def ocr_from_aliyun(image_path, access_key_id=None, access_key_secret=None):
    """占位：阿里云 OCR 接入点（自行实现）。"""
    raise NotImplementedError("请使用阿里云 OCR SDK/接口并在此处实现调用")

