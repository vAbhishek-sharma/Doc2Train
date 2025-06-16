# utils/plugin_loader.py

import importlib.util
from importlib.metadata import entry_points
from pathlib import Path
from typing import Type
from typing import Union, List, Dict

def load_plugins_from_dirs(
    dirs: Union[List[Path], List[str]],
    base_class: Type,
    entry_point_group: str
) -> Dict[str, Type]:
    """
    Scan filesystem dirs *and* setuptools entry-points for subclasses of `base_class`.
    Returns a dict mapping plugin-name → plugin-class.
    """
    plugins: dict[str, Type] = {}

    # 1) Filesystem scan
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for file in p.glob("*_plugin.py"):
            spec = importlib.util.spec_from_file_location(file.stem, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            for attr in dir(module):
                cls = getattr(module, attr)
                if (
                    isinstance(cls, type)
                    and issubclass(cls, base_class)
                    and cls is not base_class
                ):
                    name = attr.lower().replace("plugin", "")
                    plugins[name] = cls

    # 2) setuptools‐style entry points via importlib.metadata
    eps = entry_points()
    # Python 3.10+ API:
    matched = eps.select(group=entry_point_group) if hasattr(eps, "select") \
              else eps.get(entry_point_group, [])

    for ep in matched:
        try:
            cls = ep.load()
        except Exception:
            continue
        if (
            isinstance(cls, type)
            and issubclass(cls, base_class)
            and cls is not base_class
        ):
            plugins[ep.name] = cls

    return plugins
