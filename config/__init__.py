"""
配置模块
=========
该模块提供了一个全局的、统一的配置实例。

可以直接从任何地方导入并使用:
`from config import settings`

`settings` 对象包含了项目中所有可配置的参数。
"""

from .settings import Settings, settings

__all__ = [
    'Settings',  # The main config class for type hinting and instantiation
    'settings',  # The global singleton instance
]
