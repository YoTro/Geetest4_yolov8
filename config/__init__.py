import sys
import os

# Add the project root and 'libs' directory to sys.path
# This ensures that internal modules like 'ppocr' can be found.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

libs_path = os.path.join(project_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

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
