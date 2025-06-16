# core/plugin_setup.py

from pathlib import Path
import sys
import os
from utils.plugin_loader import load_plugins_from_dirs

# Import base classes directly
from plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin, register_llm_plugin
from processors.base_processor import BaseProcessor, register_processor
from outputs.base_writer import BaseWriter, register_writer
from outputs.base_formatters import BaseFormatter, register_formatter

import ipdb

def set_plugins(config):
    ipdb.set_trace()
    """
    Discover & register LLM, Processor, Writer, and Formatter plugins in one place.
    """
    # 1) LLM plugins
    llm_dirs = [
        Path(__file__).parent.parent / 'plugins' / 'llm_plugins',
        *config.get('llm_plugin_dirs', [])
    ]
    llm_eps = "doc2train.llm_plugins"
    for name, cls in load_plugins_from_dirs(llm_dirs, BaseLLMPlugin, llm_eps).items():
        register_llm_plugin(name, cls)

    # 2) Processor plugins
    proc_dirs = [
        Path(__file__).parent.parent / 'processors_plugin',
        *config.get('processor_plugin_dirs', [])
    ]
    proc_eps = "doc2train.processor_plugins"
    for name, cls in load_plugins_from_dirs(proc_dirs, BaseProcessor, proc_eps).items():
        register_processor(name, cls)

    # 3) Writer plugins
    writer_dirs = [
        Path(__file__).parent.parent / 'outputs' / 'writer_plugins',
        *config.get('writer_plugin_dirs', [])
    ]
    writer_eps = "doc2train.writer_plugins"
    for name, cls in load_plugins_from_dirs(writer_dirs, BaseWriter, writer_eps).items():
        register_writer(name, cls)

    # 4) Formatter plugins
    fmt_dirs = [
        Path(__file__).parent.parent / 'outputs' / 'formatter_plugins',
        *config.get('formatter_plugin_dirs', [])
    ]
    fmt_eps = "doc2train.formatter_plugins"
    for name, cls in load_plugins_from_dirs(fmt_dirs, BaseFormatter, fmt_eps).items():
        register_formatter(name, cls)
