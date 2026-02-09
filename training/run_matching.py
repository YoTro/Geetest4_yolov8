# -*- coding: utf-8 -*-
import os
import argparse
from PIL import Image
from core.matcher import ImageMatcher
from config.settings import settings
from training.generate_char_dictionary import generate_dictionary

def run_test_matching():
    """
    运行基础匹配测试，演示 find_best_match 的功能。
    """
    dict_full_path = os.path.join(settings.paths.base_dir, settings.paths.dict_image_dir)

    # 确保字典图片已经生成
    if not os.path.exists(dict_full_path) or not os.listdir(dict_full_path):
        print(f"字典目录 '{dict_full_path}' 不存在或为空。正在尝试生成字典图片...")
        generate_dictionary()
        if not os.path.exists(dict_full_path) or not os.listdir(dict_full_path):
            print("字典图片生成失败，请检查 training/generate_char_dictionary.py。")
            return

    # 初始化 ImageMatcher
    matcher = ImageMatcher()
    if not matcher.dictionary:
        return

    weights = {'ssim': 0.1, 'hog': 0.6, 'proj': 0.3}
    print(f"\n当前匹配权重: {weights}")

    test_char = '柳'
    test_image_path = f"./data/dataset/paddle/images/val/geetest_1770134346618_494_2.png"
    if not os.path.exists(test_image_path):
        print(f"测试图片 '{test_image_path}' 不存在。请检查路径。")
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

def run_advanced_matching_test():
    """
    运行高级匹配测试，演示 find_best_matches_for_main_image 的功能。
    该方法使用 GrabCut、minAreaRect 和匈牙利算法。
    """
    print("\n--- 开始高级匹配测试 ---")
    matcher = ImageMatcher(use_cache=False) # 建议在高级匹配时不使用旧缓存

    # !!! 重要 !!!
    # 请在此处填入您的测试图片路径。
    # main_image_paths 应该是从主图中分割出的、带复杂背景的文字图片。
    # ques_image_paths 应该是对应的、背景清晰的“问题”文字图片。
    # 确保两个列表的长度一致。
    main_image_paths = [
        settings.paths.base_dir+"/data/dataset/paddle/main_images/208 (40)_char_2.png",
        settings.paths.base_dir+"/data/dataset/paddle/main_images/4_char_2.png",
        settings.paths.base_dir+"/data/dataset/paddle/main_images/102_char_2.png",
    ]
    ques_image_paths = [
        settings.paths.base_dir+"/data/dataset/paddle/images/val/geetest_1770134346618_494_2.png",
        settings.paths.base_dir+"/data/dataset/paddle/images/val/geetest_1770134345843_393_2.png",
        settings.paths.base_dir+"/data/dataset/paddle/images/val/geetest_1770134345843_393_0.png",
    ]
    
    # 检查路径是否为占位符
    if settings.paths.base_dir+"data/dataset/trocr/images" in main_image_paths[0]:
        print("\n[警告] 请打开 training/run_matching.py 并修改 'run_advanced_matching_test' 函数中的图片路径。")
        print("您需要提供待匹配的 'main' 图片和 'ques' 图片的真实路径。")
        return

    try:
        main_char_images = [Image.open(p) for p in main_image_paths]
        ques_char_images = [Image.open(p) for p in ques_image_paths]
    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e}")
        print("请确保您提供的图片路径正确。")
        return

    print(f"已加载 {len(main_char_images)} 张主图文字和 {len(ques_char_images)} 张问题文字用于匹配。")

    # 调用高级匹配方法
    matches = matcher.find_best_matches_for_main_image(main_char_images, ques_char_images)

    print("\n--- 高级匹配结果 (匈牙利算法) ---")
    if not matches:
        print("没有找到任何匹配。")
        return
        
    print("最佳匹配组合:")
    for main_idx, ques_idx, score in matches:
        main_img_name = os.path.basename(main_image_paths[main_idx])
        ques_img_name = os.path.basename(ques_image_paths[ques_idx])
        print(f"  主图图片 '{main_img_name}' (索引 {main_idx}) "
              f"-> 问题图片 '{ques_img_name}' (索引 {ques_idx}) "
              f"| 相似度得分: {score:.4f}")
    print("------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行 ImageMatcher 的匹配测试。")
    parser.add_argument(
        "mode",
        nargs='?',
        default="simple",
        choices=["simple", "advanced"],
        help="选择测试模式: 'simple' (基础匹配) 或 'advanced' (高级匹配)。默认为 'simple'。"
    )
    args = parser.parse_args()

    if args.mode == "simple":
        run_test_matching()
    elif args.mode == "advanced":
        run_advanced_matching_test()
