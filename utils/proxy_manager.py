# utils/proxy_manager.py
import logging
from pathlib import Path
from typing import List, Optional
from curl_cffi.requests import get, RequestsError

from config import settings

logger = logging.getLogger(__name__)

def _fetch_proxies_from_url_internal(proxy_url: str) -> List[str]:
    """从URL获取代理列表。"""
    try:
        # Use impersonate_browser from settings
        response = get(proxy_url, impersonate=settings.geetest.impersonate_browser, timeout=settings.geetest.request_timeout)
        response.raise_for_status()
        proxies = response.text.strip().splitlines()
        logger.info(f"成功从URL加载 {len(proxies)} 个代理。")
        return [p for p in proxies if p]
    except RequestsError as e:
        logger.error(f"从URL加载代理失败: {e}")
        return []

def _fetch_proxies_from_file_internal(proxy_file: Path) -> List[str]:
    """从本地文件加载代理列表。"""
    try:
        with open(proxy_file, 'r', encoding='utf-8') as f:
            proxies = f.read().strip().splitlines()
            logger.info(f"成功从文件 {proxy_file.name} 加载 {len(proxies)} 个代理。")
            return [p for p in proxies if p]
    except IOError as e:
        logger.error(f"从文件加载代理失败: {e}")
        return []

def get_proxies(proxy_source: str) -> List[str]:
    """
    根据 proxy_source 的类型（URL、文件路径或单代理字符串）获取代理列表。
    """
    if proxy_source.startswith("http://") or proxy_source.startswith("https://"):
        try:
            # Test if the URL returns a list or is a single proxy
            response = get(proxy_source, impersonate=settings.geetest.impersonate_browser, timeout=settings.geetest.request_timeout)
            response.raise_for_status()
            if "\n" in response.text or " " in response.text: # Heuristic to check if it's a list
                return _fetch_proxies_from_url_internal(proxy_source)
            else:
                return [proxy_source.strip()] # It's a single proxy URL
        except RequestsError:
             logger.warning(f"从URL '{proxy_source}' 获取代理列表失败，尝试将其视为单个代理字符串。")
             return [proxy_source.strip()] # Fallback: Assume it's a single proxy if URL fetch fails

    elif Path(proxy_source).is_file():
        return _fetch_proxies_from_file_internal(Path(proxy_source))
    else: # Treat as single proxy string
        return [proxy_source.strip()]

