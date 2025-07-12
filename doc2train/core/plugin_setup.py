"""
core/plugin_setup.py

Central orchestration for discovering and registering all plugin types via their managers.
"""

from doc2train.core.plugin_managers.llm_plugin_manager import LLMPluginManager
from doc2train.core.plugin_managers.processor_plugin_manager import ProcessorPluginManager
from doc2train.core.plugin_managers.writer_plugin_manager import WriterPluginManager
from doc2train.core.plugin_managers.formatter_plugin_manager import FormatterPluginManager
from doc2train.core.plugin_managers.generator_plugin_manager import GeneratorPluginManager

from doc2train.core.registries.llm_registry import register_llm_plugin
from doc2train.core.registries.processor_registry import register_processor
from doc2train.core.registries.writer_registry import register_writer
from doc2train.core.registries.formatter_registry import register_formatter
from doc2train.core.registries.generator_registry import register_generator

from doc2train.core.registries.llm_registry import _LLM_REGISTRY
from doc2train.core.registries.processor_registry import _PROCESSOR_REGISTRY
from doc2train.core.registries.writer_registry import _WRITER_REGISTRY
from doc2train.core.registries.formatter_registry import _FORMATTER_REGISTRY
from doc2train.core.registries.generator_registry import _GENERATOR_REGISTRY



def set_plugins(config: dict):
    # --- LLM plugins ---
    llm_mgr = LLMPluginManager(config)
    llm_configs = config.get("llm_providers", {})  # Assumes structure: {name: {provider config}}
    for name, cls in llm_mgr.plugins.items():

        provider_config = llm_configs.get(name, {})
        register_llm_plugin(name, cls, config=provider_config)

    # --- Processor plugins ---
    proc_mgr = ProcessorPluginManager(config)

    for info in proc_mgr.plugins.values():
        name = info.get('name')
        extensions = info.get('extensions', [])
        proc_cls = info.get('class')
        register_processor(name, extensions, proc_cls)

    # --- Writer plugins ---
    writer_mgr = WriterPluginManager(config)
    for name, writer_cls in writer_mgr.plugins.items():
        register_writer(name, writer_cls)

    # --- Formatter plugins ---
    fmt_mgr = FormatterPluginManager(config)
    for name, fmt_cls in fmt_mgr.plugins.items():
        register_formatter(name, fmt_cls)

    # --- Generator plugins ---
    gen_mgr = GeneratorPluginManager(config)
    for name, gen_cls in gen_mgr.plugins.items():
        register_generator(name, gen_cls)

