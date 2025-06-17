"""
core/plugin_setup.py

Central orchestration for discovering and registering all plugin types via their managers.
"""

from core.llm_plugin_manager import LLMPluginManager
from processors.processor_plugin_manager import ProcessorPluginManager
from outputs.writer_plugin_manager import WriterPluginManager
from outputs.formatter_plugin_manager import FormatterPluginManager

from plugins.llm_plugins.base_llm_plugin import register_llm_plugin
from processors.base_processor import register_processor
from outputs.base_writer import register_writer
from outputs.base_formatters import register_formatter


def set_plugins(config: dict):
    """
    Discover & register LLM, Processor, Writer, and Formatter plugins using
    their respective PluginManager classes.
    """
    # — LLM plugins —
    llm_mgr = LLMPluginManager(config)
    for name, cls in llm_mgr.plugins.items():
        register_llm_plugin(name, cls)

    # — Processor plugins —
    proc_mgr = ProcessorPluginManager(config)
    for name, cls in proc_mgr.plugins.items():
        register_processor(name, cls)

    # — Writer plugins —
    writer_mgr = WriterPluginManager(config)
    for name, cls in writer_mgr.plugins.items():
        register_writer(name, cls)

    # — Formatter plugins —
    fmt_mgr = FormatterPluginManager(config)
    for name, cls in fmt_mgr.plugins.items():
        register_formatter(name, cls)
