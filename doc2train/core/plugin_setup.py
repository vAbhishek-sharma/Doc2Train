"""
core/plugin_setup.py

Central orchestration for discovering and registering all plugin types via their managers.
"""

from doc2train.core.llm_plugin_manager import LLMPluginManager
from doc2train.processors.processor_plugin_manager import ProcessorPluginManager
from doc2train.outputs.writer_plugin_manager import WriterPluginManager
from doc2train.outputs.formatter_plugin_manager import FormatterPluginManager

from doc2train.plugins.llm_plugins.base_llm_plugin import register_llm_plugin
from doc2train.processors.base_processor import register_processor
from doc2train.outputs.base_writer import register_writer
from doc2train.outputs.base_formatters import register_formatter
import ipdb
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
    for info in proc_mgr.plugins.values():
        name = info.get('name')
        extensions = info.get('extensions', [])
        proc_cls = info.get('class')
        register_processor(name, extensions, proc_cls)

    # — Writer plugins —
    writer_mgr = WriterPluginManager(config)
    for name, writer_cls in writer_mgr.plugins.items():
        register_writer(name, writer_cls)

    # — Formatter plugins —
    fmt_mgr = FormatterPluginManager(config)
    for name, fmt_cls in fmt_mgr.plugins.items():
        register_formatter(name, fmt_cls)
