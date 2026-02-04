"""数据收集器
使用代理池和多线程并发收集Geetest验证码图片，或通过延迟降频进行单线程收集。
"""
import logging
import random
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from curl_cffi.requests import Session, get, RequestsError
from tqdm import tqdm

from config import settings
from core.gt4 import GeetestV4
from utils import image_processor

def _fetch_proxies_from_url(proxy_url: str) -> List[str]:
    """从URL获取代理列表。"""
    logger = logging.getLogger(__name__)
    try:
        response = get(proxy_url, impersonate="chrome110", timeout=10)
        response.raise_for_status()
        proxies = response.text.strip().splitlines()
        logger.info(f"成功从URL加载 {len(proxies)} 个代理。")
        return [p for p in proxies if p]
    except RequestsError as e:
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

def _get_train_val_split_folder() -> str:
    """Randomly determines if an item belongs to 'train' or 'val' split (9:1 ratio)."""
    return "train" if random.random() < 0.9 else "val"

def _collect_single_sample(proxy: Optional[str], output_dir: list, captcha_id: str) -> bool:
    """使用单个代理或无代理收集一个验证码样本。"""
    logger = logging.getLogger(__name__)
    session = None
    try:
        session = Session(
            impersonate=settings.geetest.impersonate_browser,
            timeout=settings.geetest.request_timeout
        )
        if proxy:
            session.proxies = {"http": proxy, "https": proxy}
        
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

        # Decide whether to save in train or val
        split_folder = _get_train_val_split_folder()
            
        timestamp = int(time.time() * 1000)
        random_part = random.randint(100, 999)
        filename = f"geetest_{timestamp}_{random_part}.png"
        image_path = output_dir[0] / split_folder / filename
        main_image.save(image_path)

        # Handle ques_imgs
        if image_urls.get("ques_imgs"):
            ques_dir = output_dir[1] / split_folder
            for i, ques_img_url in enumerate(image_urls["ques_imgs"]):
                ques_image = image_processor.download_image(session, ques_img_url)
                if ques_image:
                    ques_filename = f"geetest_{timestamp}_{random_part}_{i}.png"
                    ques_image_path = ques_dir / ques_filename
                    ques_image.save(ques_image_path)
                else:
                    logger.warning(f"下载问题图片失败: {ques_img_url} (代理: {proxy or '无'})。")

        return True

    except Exception as e:
        # Check if the exception text contains content from our specific JSON decode log
        if "JSON decode failed" in str(e):
             # The detailed error is already logged in gt4.py, so we just log a simpler warning here
            logger.warning(f"收集样本时发生JSON解析错误 (代理: {proxy or '无'}), 已被Cloudflare拦截。")
        else:
            logger.warning(f"收集样本时发生未知错误 (代理: {proxy or '无'}): {e}")
        return False
    finally:
        if session:
            session.close()

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

    # Create train and val directories
    for folder in [main_image_output_dir, ques_image_output_dir]:
        (folder / "train").mkdir(parents=True, exist_ok=True)
        (folder / "val").mkdir(parents=True, exist_ok=True)

    image_output_dir = [main_image_output_dir, ques_image_output_dir]
    logger.info(f"数据将保存到: {output_dir}, 并按9:1比例分配到train/val文件夹。")

    if proxy_source:
        logger.info(f"检测到代理源，将使用 {max_workers} 个工作线程的代理模式。")
        
        proxies = []
        if proxy_source.startswith("http://") or proxy_source.startswith("https://"):
            try:
                # Test if the URL returns a list or is a single proxy
                response = get(proxy_source, impersonate="chrome110", timeout=10)
                response.raise_for_status()
                if "\n" in response.text or " " in response.text:
                    proxies = _fetch_proxies_from_url(proxy_source)
                else:
                    proxies = [proxy_source.strip()]
            except RequestsError:
                 proxies = [proxy_source.strip()] # Assume it's a single proxy if URL fetch fails
        elif Path(proxy_source).is_file():
            proxies = _fetch_proxies_from_file(Path(proxy_source))
        else:
            proxies = [proxy_source.strip()]

        if not proxies:
            logger.error("没有可用的代理，收集任务终止。")
            return
        
        logger.info(f"加载了 {len(proxies)} 个代理。")

        with tqdm(total=num_samples, desc="收集中 (代理模式)") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_collect_single_sample, random.choice(proxies), image_output_dir, captcha_id) for _ in range(min(num_samples, max_workers))}
                
                while futures:
                    done = set()
                    for future in as_completed(futures):
                        if future.result():
                            pbar.update(1)
                        
                        done.add(future)
                        
                        if pbar.n < num_samples:
                            # Submit a new task to replace the one that just finished
                            new_task = executor.submit(_collect_single_sample, random.choice(proxies), image_output_dir, captcha_id)
                            futures.add(new_task)
                    
                    futures -= done
    
    else: # Delay mode
        logger.info(f"未提供代理源，将使用单线程降频模式 (每次请求间隔约 {delay} 秒)。")
        with tqdm(total=num_samples, desc="收集中 (降频模式)") as pbar:
            for _ in range(num_samples):
                if _collect_single_sample(proxy=None, output_dir=image_output_dir, captcha_id=captcha_id):
                    pbar.update(1)
                
                if pbar.n < num_samples:
                    time.sleep(max(0.5, delay + random.uniform(-1.0, 1.0)))
    
    logger.info(f"收集任务完成！")