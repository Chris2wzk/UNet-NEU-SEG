import os
import importlib

# 这里的逻辑会动态导入 blocks 文件夹下所有的 .py
# 并将其全局符号注入到 blocks 包的命名空间里，从而方便后续 `from blocks import *`
# 使用时注意：若有不同文件内同名类/函数，后导入的会覆盖先导入的。

__all__ = []

this_dir = os.path.dirname(__file__)
for f in os.listdir(this_dir):
    if f.endswith('.py') and not f.startswith('__'):
        module_name = f[:-3]  # 去掉 .py 后缀
        m = importlib.import_module(f'.{module_name}', package=__name__)
        for attr in dir(m):
            if not attr.startswith('_'):
                globals()[attr] = getattr(m, attr)
                __all__.append(attr)

