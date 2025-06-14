# NEW: core/llm_plugin_manager.py
"""
LLM Plugin Manager for loading and managing LLM provider plugins
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin

class LLMPluginManager:
    """
    Manager for LLM provider plugins
    """

    def __init__(self):
        self.plugins = {}
        self.builtin_providers = ['openai', 'deepseek', 'local']  # Keep existing providers
        self._load_builtin_plugins()

    def _load_builtin_plugins(self):
        """Load built-in LLM plugins"""
        plugins_dir = Path(__file__).parent.parent / 'plugins' / 'llm_plugins'

        if plugins_dir.exists():
            # Try to load built-in plugins
            for plugin_file in plugins_dir.glob('*_plugin.py'):
                try:
                    self.load_plugin(str(plugin_file))
                except Exception as e:
                    print(f"⚠️ Failed to load built-in plugin {plugin_file.name}: {e}")

    def load_plugin(self, plugin_path: str) -> bool:
        try:
            # Convert file path to module path
            plugin_rel_path = os.path.relpath(plugin_path, start=os.getcwd()).replace(os.sep, '.')
            module_name = os.path.splitext(plugin_rel_path)[0]  # remove .py

            plugin_module = importlib.import_module(module_name)

            for attr_name in dir(plugin_module):
                attr = getattr(plugin_module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseLLMPlugin) and attr != BaseLLMPlugin:
                    plugin_name = attr_name.lower().replace('plugin', '')
                    config = self._get_plugin_config(plugin_name)
                    plugin_instance = attr(config)
                    self.plugins[plugin_name] = plugin_instance
                    print(f"✅ Loaded LLM plugin: {plugin_name}")
                    return True

            return False

        except Exception as e:
            print(f"❌ Failed to load plugin {plugin_path}: {e}")
            return False

    def _get_plugin_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a plugin from environment variables

        Args:
            provider_name: Name of the provider

        Returns:
            Configuration dictionary
        """
        config = {}

        # Standard config mapping
        env_mappings = {
            'anthropic': {
                'api_key': 'ANTHROPIC_API_KEY',
                'base_url': 'ANTHROPIC_BASE_URL'
            },
            'google': {
                'api_key': 'GOOGLE_API_KEY',
                'base_url': 'GOOGLE_BASE_URL'
            },
            'cohere': {
                'api_key': 'COHERE_API_KEY',
                'base_url': 'COHERE_BASE_URL'
            },
            'mistral': {
                'api_key': 'MISTRAL_API_KEY',
                'base_url': 'MISTRAL_BASE_URL'
            }
        }

        if provider_name in env_mappings:
            for key, env_var in env_mappings[provider_name].items():
                value = os.getenv(env_var)
                if value:
                    config[key] = value

        # Generic fallback - look for PROVIDERNAME_API_KEY pattern
        if not config.get('api_key'):
            generic_key = f"{provider_name.upper()}_API_KEY"
            api_key = os.getenv(generic_key)
            if api_key:
                config['api_key'] = api_key

        return config

    def discover_plugins(self, plugin_dirs: List[str]):
        """
        Discover and load all plugins from multiple directories
        Note: Built-in plugins are already loaded in __init__
        """
        total_loaded = 0

        for plugin_dir in plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                print(f"⚠️ Plugin directory does not exist: {plugin_dir}")
                continue

            loaded_count = 0
            for plugin_file in plugin_path.glob("*_plugin.py"):
                if self.load_plugin(str(plugin_file)):
                    loaded_count += 1

            if loaded_count > 0:
                print(f"📦 Discovered {loaded_count} plugins in {plugin_dir}")

            total_loaded += loaded_count

        print(f"✅ Total additional plugins loaded: {total_loaded}")

    def get_plugin(self, provider_name: str) -> Optional[BaseLLMPlugin]:
        """
        Get plugin by provider name

        Args:
            provider_name: Name of the provider

        Returns:
            Plugin instance or None
        """
        return self.plugins.get(provider_name)

    def get_available_providers(self) -> List[str]:
        """
        Get list of all available providers (builtin + plugins)

        Returns:
            List of provider names
        """
        return self.builtin_providers + list(self.plugins.keys())

    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a provider

        Args:
            provider_name: Name of the provider

        Returns:
            Provider information or None
        """
        plugin = self.get_plugin(provider_name)
        if plugin:
            return plugin.get_provider_info()
        return None

    def list_plugins(self):
        """Print information about all loaded plugins"""
        if not self.plugins:
            print("No LLM plugins loaded")
            return

        print("🔌 Loaded LLM Plugins:")
        for name, plugin in self.plugins.items():
            info = plugin.get_provider_info()
            capabilities = []
            if info['capabilities']['text']:
                capabilities.append('text')
            if info['capabilities']['vision']:
                capabilities.append('vision')
            if info['capabilities']['streaming']:
                capabilities.append('streaming')

            status = "✅ Ready" if info['config_valid'] else "❌ Config missing"
            print(f"   {name}: {', '.join(capabilities)} - {status}")



# Global plugin manager instance
_plugin_manager = LLMPluginManager()

def get_plugin_manager() -> LLMPluginManager:
    """Get the global plugin manager"""
    return _plugin_manager


"""
Enhanced LLM client with plugin support and direct media processing
"""

from core.llm_plugin_manager import get_plugin_manager
from pathlib import Path


def call_llm_plugin(provider: str, prompt: str, task: str = 'general',
                   images: List[Union[str, bytes]] = None, **kwargs) -> str:
    """
    NEW: Call LLM using plugin system

    Args:
        provider: Provider name (from plugins)
        prompt: Text prompt
        task: Task type for routing
        images: Optional images for vision models
        **kwargs: Additional parameters

    Returns:
        Generated response
    """
    plugin_manager = get_plugin_manager()
    plugin = plugin_manager.get_plugin(provider)

    if not plugin:
        raise ValueError(f"Plugin not found for provider: {provider}")

    if not plugin.validate_config():
        raise Exception(f"Invalid configuration for provider: {provider}")

    # Use vision model if images are provided and supported
    if images and plugin.capabilities['vision']:
        return plugin.call_vision_model(prompt, images, **kwargs)
    else:
        return plugin.call_text_model(prompt, **kwargs)

def process_media_directly(media_path: str, provider: str = None,
                          prompt: str = None, **kwargs) -> str:
    """
    NEW: Process images/videos directly with LLM, skipping traditional processors

    Args:
        media_path: Path to image or video file
        provider: LLM provider to use (auto-detect if None)
        prompt: Optional prompt to guide analysis
        **kwargs: Additional parameters

    Returns:
        LLM analysis of the media
    """
    if not Path(media_path).exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    plugin_manager = get_plugin_manager()

    # Auto-detect provider if not specified
    if not provider:
        # Find first available vision-capable provider
        for provider_name in plugin_manager.get_available_providers():
            plugin = plugin_manager.get_plugin(provider_name)
            if plugin and plugin.capabilities['vision'] and plugin.validate_config():
                provider = provider_name
                break

        if not provider:
            # Fallback to built-in providers
            provider_info = _get_vision_provider()
            if provider_info:
                # Use existing vision processing
                with open(media_path, 'rb') as f:
                    image_data = f.read()

                image_data_dict = {
                    'path': media_path,
                    'data': image_data,
                    'ocr_text': ''
                }

                default_prompt = prompt or "Analyze this image thoroughly and extract all relevant information including text, objects, scenes, and context."
                return call_llm_vision(default_prompt, image_data_dict)
            else:
                raise Exception("No vision-capable providers available")

    # Use plugin for direct processing
    plugin = plugin_manager.get_plugin(provider)
    if not plugin:
        raise ValueError(f"Plugin not found for provider: {provider}")

    return plugin.process_direct_media(media_path, prompt, **kwargs)

def get_available_providers() -> List[str]:
    """
    NEW: Get all available providers (enhanced to include plugins)

    Returns:
        List of available provider names
    """
    plugin_manager = get_plugin_manager()
    return plugin_manager.get_available_providers()

def get_provider_capabilities(provider: str) -> Dict[str, Any]:
    """
    NEW: Get capabilities of a specific provider

    Args:
        provider: Provider name

    Returns:
        Capabilities dictionary
    """
    plugin_manager = get_plugin_manager()

    # Check plugins first
    info = plugin_manager.get_provider_info(provider)
    if info:
        return info['capabilities']

    # Fallback to built-in providers
    builtin_capabilities = {
        'openai': {'text': True, 'vision': True, 'streaming': False},
        'deepseek': {'text': True, 'vision': False, 'streaming': False},
        'local': {'text': True, 'vision': False, 'streaming': False}
    }

    return builtin_capabilities.get(provider, {})

def discover_llm_plugins(plugin_dir: str):
    """
    NEW: Discover and load LLM plugins from directory

    Args:
        plugin_dir: Directory containing plugin files
    """
    plugin_manager = get_plugin_manager()
    plugin_manager.discover_plugins(plugin_dir)

def list_llm_plugins():
    """
    NEW: List all loaded LLM plugins
    """
    plugin_manager = get_plugin_manager()
    plugin_manager.list_plugins()


# NEW: Enhanced settings.py additions
"""
Add these configurations to your existing settings.py
"""

# NEW: Plugin settings
PLUGIN_DIRS = {
    'processors': os.getenv('PROCESSOR_PLUGIN_DIR', 'plugins/processors'),
    'llm_providers': os.getenv('LLM_PLUGIN_DIR', 'plugins/llm_plugins')
}

# NEW: Direct media processing settings
DIRECT_MEDIA_PROCESSING = {
    'enabled': os.getenv('ENABLE_DIRECT_MEDIA', 'true').lower() == 'true',
    'default_provider': os.getenv('DIRECT_MEDIA_PROVIDER', 'auto'),  # auto-detect best provider
    'default_prompt': os.getenv('DIRECT_MEDIA_PROMPT',
        "Analyze this image/video comprehensively. Extract all text, identify objects, "
        "describe scenes, and provide relevant context and insights."),
    'supported_formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp', '.mp4', '.avi', '.mov']
}

# NEW: Enhanced LLM routing to include plugins
def get_enhanced_llm_routing():
    """Get LLM routing including plugin providers"""
    routing = LLM_ROUTING.copy()

    # Add plugin-specific routing
    plugin_manager = get_plugin_manager()
    for provider in plugin_manager.get_available_providers():
        if provider not in ['openai', 'deepseek', 'local']:  # Skip built-ins
            capabilities = get_provider_capabilities(provider)
            if capabilities.get('vision'):
                routing[f'{provider}_vision'] = f"{provider}/vision"
            if capabilities.get('text'):
                routing[f'{provider}_text'] = f"{provider}/text"

    return routing

# NEW: Enhanced validation function
def validate_enhanced_config():
    """Enhanced configuration validation including plugins"""
    issues = validate_config()  # Call existing validation

    # Validate plugin directories
    for plugin_type, plugin_dir in PLUGIN_DIRS.items():
        if not Path(plugin_dir).exists():
            issues.append(f"{plugin_type} plugin directory not found: {plugin_dir}")

    # Check plugin configurations
    plugin_manager = get_plugin_manager()
    for provider in plugin_manager.get_available_providers():
        if provider not in ['openai', 'deepseek', 'local']:
            info = plugin_manager.get_provider_info(provider)
            if info and not info['config_valid']:
                issues.append(f"Plugin {provider} configuration incomplete")

    return issues
