"""
核心模块初始化
"""
from .captcha_processor import CaptchaProcessor
from .paddle_recognizer import PaddleRecognizer, cosine_similarity
__all__ = [
    'CaptchaProcessor',
    'PaddleRecognizer',
    'cosine_similarity',
]
