# -*- coding: utf-8 -*-
import os
from core.match_ques_features import ImageMatcher
from config.settings import settings
from .generate_char_dictionary import generate_dictionary

def run_test_matching():
    """
    运行匹配测试，演示 ImageMatcher 的功能。
    """
    dict_full_path = os.path.join(settings.paths.base_dir, settings.paths.dict_image_dir)

    # 确保字典图片已经生成
    if not os.path.exists(dict_full_path) or not os.listdir(dict_full_path):
        print(f"字典目录 '{dict_full_path}' 不存在或为空。正在尝试生成字典图片...")
        generate_dictionary()
        if not os.path.exists(dict_full_path) or not os.listdir(dict_full_path):
            print("字典图片生成失败，请检查 training/generate_char_dictionary.py。")
            return

    # 初始化 ImageMatcher (它会自动从settings加载路径)
    matcher = ImageMatcher()
    if not matcher.dictionary: # 如果字典为空，则初始化失败
        return

    # --- 设置权重 ---
    weights = {
        'ssim': 0.1,
        'hog': 0.5,
        'proj': 0.4
    }
    print(f"\n当前匹配权重: {weights}")

    # --- 选择一个示例图片进行匹配 ---
    test_char = '定'
    test_image_path = f"./data/dataset/paddle/images/val/geetest_1769936341131_172_0.png"

    if not os.path.exists(test_image_path):
        print(f"测试图片 '{test_image_path}' 不存在。")
        print(f"请确保 '{test_char}' 在 '{settings.paths.char_dict_path}' 文件中。")
        return

    print(f"\n正在识别测试图片: {test_image_path} (期望结果: '{test_char}')")
    best_match_char, score = matcher.find_best_match(test_image_path, weights)

    print(f"\n------------------------------------")
    print(f"识别结果:")
    print(f"  最匹配的字符: '{best_match_char}'")
    print(f"  匹配得分: {score:.4f}")
    print(f"------------------------------------")

    if best_match_char == test_char:
        print("测试成功：识别结果与期望结果一致！")
    else:
        print("测试失败：识别结果与期望结果不一致。")
if __name__ == '__main__':
    run_test_matching()
