import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Type, Union
import ipdb

def load_plugins_from_dirs(
    dirs: List[Union[str, Path]],
    base_class: Type,
    pkg_prefix: str,
) -> Dict[str, Type]:
    plugins: Dict[str, Type] = {}
    for directory in dirs:
        p = Path(directory)
        if not p.exists() or not p.is_dir():
            continue
        for file in p.glob("*.py"):
            if file.name.startswith("_"):
                continue
            module_name = f"{pkg_prefix}.{file.stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                raise ImportError(f"Failed to import plugin {module_name}: {e}")

            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if (
                    isinstance(attribute, type)
                    and issubclass(attribute, base_class)
                    and attribute is not base_class
                ):
                    plugin_name = getattr(attribute, "name", attribute.__name__)
                    plugins[plugin_name] = attribute
    return plugins
