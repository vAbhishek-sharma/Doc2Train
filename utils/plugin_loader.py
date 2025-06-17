import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Type, Union


def load_plugins_from_dirs(
    dirs: List[Union[str, Path]],
    base_class: Type,
    pkg_prefix: str,
) -> Dict[str, Type]:
    """
    Load all plugins (subclasses of `base_class`) from the given directories,
    using `pkg_prefix` as the dotted package name to anchor relative imports.

    Args:
        dirs: List of directory paths (str or Path) to search for plugin modules.
        base_class: The base class that plugins must extend.
        pkg_prefix: Dotted path for the package context (e.g. 'plugins.llm_plugins').

    Returns:
        A dict mapping plugin names to their classes.
    """
    plugins: Dict[str, Type] = {}
    for directory in dirs:
        p = Path(directory)
        if not p.exists() or not p.is_dir():
            continue
        for file in p.glob("*.py"):
            if file.name.startswith("_"):
                # skip private or special modules
                continue
            module_name = f"{pkg_prefix}.{file.stem}"
            # Load as a normal module so relative imports work
            spec = importlib.util.spec_from_file_location(module_name, str(file))
            module = importlib.util.module_from_spec(spec)
            # Ensure relative imports inside the plugin resolve to pkg_prefix
            module.__package__ = pkg_prefix
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)  # type: ignore
            except Exception as e:
                raise ImportError(f"Failed to load plugin {module_name}: {e}")
            # Discover subclasses of base_class
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
