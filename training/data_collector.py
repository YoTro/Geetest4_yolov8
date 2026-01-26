"""
数据收集器
使用代理池和多线程并发收集Geetest验证码图片，或通过延迟降频进行单线程收集。
"""
import logging
import random
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

from config import settings
from core.gt4 import GeetestV4
from utils import image_processor

def _fetch_proxies_from_url(proxy_url: str) -> List[str]:
    """从URL获取代理列表。"""
    logger = logging.getLogger(__name__)
    try:
        response = requests.get(proxy_url, timeout=10)
        response.raise_for_status()
        proxies = response.text.strip().splitlines()
        logger.info(f"成功从URL加载 {len(proxies)} 个代理。")
        return [p for p in proxies if p]
    except requests.RequestException as e:
        logger.error(f"从URL加载代理失败: {e}")
        return []

def _fetch_proxies_from_file(proxy_file: Path) -> List[str]:
    """从本地文件加载代理列表。"""
    logger = logging.getLogger(__name__)
    try:
        with open(proxy_file, 'r', encoding='utf-8') as f:
            proxies = f.read().strip().splitlines()
            logger.info(f"成功从文件 {proxy_file.name} 加载 {len(proxies)} 个代理。")
            return [p for p in proxies if p]
    except IOError as e:
        logger.error(f"从文件加载代理失败: {e}")
        return []

def _collect_single_sample(proxy: Optional[str], output_dir: list, captcha_id: str) -> bool:
    """使用单个代理或无代理收集一个验证码样本。"""
    logger = logging.getLogger(__name__)
    session = None
    try:
        session = requests.Session()
        # 如果提供了代理，则为会话设置代理
        if proxy:
            proxies = {"http": proxy, "https": proxy}
            session.proxies = proxies
        
        geetest = GeetestV4(captcha_id, geetest_config=settings.geetest, session=session)
        
        load_data = geetest.load()
        if load_data.get("status") != "success":
            logger.warning(f"加载验证码失败 (代理: {proxy or '无'}): {load_data.get('msg', '未知错误')}")
            return False

        image_urls = geetest.extract_image_urls(load_data)
        if not image_urls.get("main_img"):
            logger.warning(f"未能提取到图片URL (代理: {proxy or '无'})。")
            return False
        
        main_image = image_processor.download_image(session, image_urls["main_img"])
        if main_image is None:
            logger.warning(f"下载主图片失败 (代理: {proxy or '无'})。")
            return False
            
        timestamp = int(time.time() * 1000)
        random_part = random.randint(100, 999)
        filename = f"geetest_{timestamp}_{random_part}.png"
        image_path = output_dir[0] / filename
        main_image.save(image_path)

        # Handle ques_imgs
        if image_urls.get("ques_imgs"):
            for i, ques_img_url in enumerate(image_urls["ques_imgs"]):
                ques_image = image_processor.download_image(session, ques_img_url)
                if ques_image:
                    ques_filename = f"geetest_{timestamp}_{random_part}_{i}.png"
                    ques_image_path = output_dir[1] / ques_filename
                    ques_image.save(ques_image_path)
                else:
                    logger.warning(f"下载问题图片失败: {ques_img_url} (代理: {proxy or '无'})。")

        return True

    except Exception as e:
        logger.warning(f"收集样本时发生错误 (代理: {proxy or '无'}): {e}")
        return False

def run_collection_pipeline(
    num_samples: int,
    output_dir: str,
    captcha_id: str,
    proxy_source: Optional[str],
    max_workers: int = 10,
    delay: float = 3.0
):
    """
    运行数据收集流水线，支持代理模式和降频模式。
    """
    logger = logging.getLogger(__name__)
    main_image_output_dir = Path(output_dir) / "images"
    ques_image_output_dir = Path(output_dir) /"ques_imgs"
    main_image_output_dir.mkdir(parents=True, exist_ok=True)
    ques_image_output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = [main_image_output_dir, ques_image_output_dir]
    logger.info(f"数据将保存到: {image_output_dir}")

    # ---- 代理模式 ----
    if proxy_source:
        logger.info(f"检测到代理源，将使用 {max_workers} 个工作线程的代理模式。")
        
        proxies = []
        # 检查是URL、文件还是单个代理字符串
        if proxy_source.startswith("http://") or proxy_source.startswith("https://"):
            # 可能是代理URL列表或单个代理
            if "\n" in requests.get(proxy_source).text: # 假设URL返回的是文本列表
                 proxies = _fetch_proxies_from_url(proxy_source)
            else: # 单个代理字符串
                proxies = [proxy_source]
                logger.info("检测到单个代理字符串。")
        elif Path(proxy_source).is_file():
            proxies = _fetch_proxies_from_file(Path(proxy_source))
        else:
            # 假设为单个代理字符串
            proxies = [proxy_source]
            logger.info("检测到单个代理字符串。")

        if not proxies:
            logger.error("没有可用的代理，收集任务终止。")
            return

        collected_count = 0
        proxy_idx = 0
        
        with tqdm(total=num_samples, desc="收集中 (代理模式)") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = set()
                proxy_idx = 0
                collected_count += 1
                while len(futures) > 0 or pbar.n < num_samples:
                    # 动态补充任务，直到达到总数或工作线程满载
                    while len(futures) < max_workers and pbar.n + len(futures) < num_samples:
                        next_proxy = proxies[proxy_idx % len(proxies)]
                        proxy_idx += 1
                        futures.add(executor.submit(_collect_single_sample, next_proxy, image_output_dir, captcha_id))

                    if not futures:
                        break

                    # 处理已完成的任务
                    done, futures = as_completed(futures), set()
                    
                    for future in done:
                        if future.result():
                            pbar.update(1)
                        
                        # 检查是否需要启动新任务
                        if pbar.n + len(futures) < num_samples:
                            next_proxy = proxies[proxy_idx % len(proxies)]
                            proxy_idx += 1
                            futures.add(executor.submit(_collect_single_sample, next_proxy, image_output_dir, captcha_id))
                    
                    if pbar.n >= num_samples:
                        # 取消所有剩余任务
                        for f in futures:
                            f.cancel()
                        break
    
    # ---- 降频模式 (无代理) ----
    else:
        logger.info(f"未提供代理源，将使用单线程降频模式 (每次请求间隔约 {delay} 秒)。")
        collected_count = 0
        with tqdm(total=num_samples, desc="收集中 (降频模式)") as pbar:
            for i in range(num_samples):
                success = _collect_single_sample(proxy=None, output_dir=image_output_dir, captcha_id=captcha_id)
                if success:
                    collected_count += 1
                    pbar.update(1)
                
                # 在两次请求之间等待，避免请求过快
                if i < num_samples - 1:
                    sleep_time = delay + random.uniform(-1.0, 1.0)
                    time.sleep(max(0.5, sleep_time)) # 保证至少等待0.5秒
    
    logger.info(f"收集任务完成！成功收集 {collected_count}/{num_samples} 个样本。")
