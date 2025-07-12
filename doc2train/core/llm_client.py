# core/llm_client.py
"""
Simple LLM client - handles calls to different AI providers
Supports OpenAI, DeepSeek, and local models
"""

from pathlib import Path
import time
import openai
import requests
import json
import base64

from typing import Dict, Any, Optional, List  # Added this line
from doc2train.config.settings import *
from doc2train.core.registries.llm_registry import (
    get_llm_plugin,
    get_available_providers,
    get_vision_provider,
    list_llm_plugins,
    get_llm_plugin_class
)
from doc2train.core.registries.llm_registry import get_available_providers as _registry_get_providers


def call_llm(prompt: str, task: str = 'general', max_retries: int = 3) -> str:
    """
    Call LLM via registry plugin with automatic provider selection and fallback.
    """
    provider_info = _get_provider_for_task(task)

    for attempt in range(max_retries):
        try:
            plugin_cls = provider_info['plugin_cls']
            if hasattr(plugin_cls, "call"):
                return plugin_cls.call(prompt, task=task)
            else:
                raise Exception(f"Provider {provider_info['provider']} missing .call() method.")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Attempt {attempt + 1} failed: {error_msg}")

            if attempt < max_retries - 1:
                provider_info = _get_fallback_provider()
            else:
                print(f"❌ All LLM attempts failed after {max_retries} retries")
                return '{"error": "LLM processing failed - continuing with extraction only"}'
    raise Exception("All retry attempts failed")



def call_vision_llm(prompt: str, images: List[bytes], config: dict) -> str:
    """
    Send `prompt` and `image_data` to a vision-capable LLM.
    Retries up to llm.max_retries, then returns any ocr_text fallback.
    """


    llm_conf    = config.get("llm", {})
    max_retries = llm_conf.get("max_retries", 1)
    backoff     = llm_conf.get("retry_backoff", 1)

    # Resolve vision-capable provider + model
    info = get_vision_provider(
        provider_name=llm_conf.get("provider"),
        model_name=   llm_conf.get("model"),
    )
    plugin_cls   = info["plugin_cls"]
    model_to_use = info["model"]
    plugin = plugin_cls(config)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            if not hasattr(plugin_cls, "call_vision"):
                raise AttributeError(f"{info['provider']} lacks call_vision()")
            return plugin.call_vision(prompt, images, model=model_to_use)
        except Exception as e:
            last_exc = e
            print(f"⚠️ Vision attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(backoff)

    # Retries exhausted → fallback to OCR if available
    ocr_text = getattr(image_data, "ocr_text", "") or ""
    if ocr_text:
        return f"Image contains text: {ocr_text}"

    raise RuntimeError(f"All vision attempts failed: {last_exc}")





def _resolve_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Pick an LLM provider + model, with fallbacks:
    1. If provider_name is given and registered, use it.
        – If model_name is supported there, use it; otherwise use its default model.
    2. If provider_name is missing or unregistered but model_name is given,
        pick the first provider that supports that model.
    3. Otherwise, fall back to the first available provider and its default model.
    Returns a dict: {"provider": str, "plugin_cls": class, "model": str}
    """
    available = get_available_providers()

    # 1) Try the requested provider
    if provider_name in available:
        cls = get_llm_plugin_class(provider_name)
        models = cls.supported_models()
        if model_name and model_name in models:
            return {"provider": provider_name, "plugin_cls": cls, "model": model_name}
        # fall back to that provider’s default
        return {
            "provider": provider_name,
            "plugin_cls": cls,
            "model": cls.get_default_model()
        }

    # 2) No valid provider requested → look for any provider that supports the requested model
    if model_name:
        for name in available:
            cls = get_llm_plugin_class(name)
            if model_name in cls.supported_models():
                return {"provider": name, "plugin_cls": cls, "model": model_name}

    # 3) Ultimate fallback: first provider + its default
    first = available[0]
    cls = get_llm_plugin_class(first)
    return {"provider": first, "plugin_cls": cls, "model": cls.get_default_model()}



def _get_provider_for_task(task: str) -> dict:
    """
    Select best provider/plugin for a task.
    This version prefers any registered and configured plugin that claims to support the task.
    """
    # Example logic: first provider that is configured and claims to support this task
    for name in get_available_providers():
        plugin_cls = get_llm_plugin(name)
        if hasattr(plugin_cls, "supports_task"):
            if plugin_cls.supports_task(task):
                return {"provider": name, "plugin_cls": plugin_cls}
    # Fallback: just return the first available provider
    for name in get_available_providers():
        plugin_cls = get_llm_plugin(name)
        return {"provider": name, "plugin_cls": plugin_cls}
    raise Exception("No available LLM providers for task: " + task)

def _get_fallback_provider() -> dict:
    # Fallback to the first available provider
    for name in get_available_providers():
        plugin_cls = get_llm_plugin(name)
        return {"provider": name, "plugin_cls": plugin_cls}
    raise Exception("No fallback LLM providers available.")

def _get_vision_provider() -> dict:
    # Select first provider that supports vision
    for name in get_available_providers():
        plugin_cls = get_llm_plugin_class(name)
        if  plugin_cls.supports_vision:
            return {"provider": name, "plugin_cls": plugin_cls}
    return None

def test_provider(provider: str, model: Optional[str] = None) -> bool:
    """
    Attempt a minimal call to the named LLM provider to verify that
    it’s correctly configured and reachable.
    Returns True if the provider responds, False on any exception.
    """
    plugin_cls = get_llm_plugin(provider)
    if not plugin_cls:
        return False

    try:
        # Instantiate the plugin (this should pick up API keys, base_url, etc.)
        plugin = plugin_cls()

        # If the plugin offers a lightweight health-check, use it:
        if hasattr(plugin, "test_connection"):
            plugin.test_connection()
        else:
            # Fallback: try listing available models or a zero-length completion
            if hasattr(plugin, "list_models"):
                plugin.list_models()
            else:
                # Last-ditch: tokenize an empty string
                plugin.estimate_tokens("")
        return True

    except Exception:
        return False


def get_available_providers() -> List[str]:
    """Return list of currently configured LLM plugin providers."""
    return _registry_get_providers()

def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Return an estimate of how many tokens `text` will encode to in the given model.
    - If tiktoken is installed, uses actual encoding for OpenAI models.
    - Otherwise falls back to a simple word-based heuristic.
    """
    try:
        import tiktoken
        # pick a default OpenAI model if none provided
        model_name = model or "gpt-3.5-turbo-0613"
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except (ImportError, KeyError):
        # fallback: split on whitespace/punctuation
        tokens = re.findall(r"\S+", text)
        return len(tokens)



def estimate_cost(text: str,
                  provider: str,
                  model: Optional[str] = None,
                  output_tokens: Optional[int] = None,
                  num_images: int = 0) -> float:

    # 1. Resolve plugin & model
    plugin_cls = get_llm_plugin(provider)
    model = model or plugin_cls.get_default_model()
    model_info = plugin_cls.supported_models().get(model, {})

    # 2. Estimate tokens
    input_tokens  = estimate_tokens(text)
    output_tokens = output_tokens or input_tokens

    # 3. Pull cost metadata
    cost_meta = model_info.get("cost", {})
    cost = (input_tokens/1_000) * cost_meta.get("input_per_1k", 0.0)
    cost += (output_tokens/1_000) * cost_meta.get("output_per_1k", 0.0)

    # 4. If this is a vision model & charges per image
    img_cost = cost_meta.get("image")
    if img_cost and num_images:
        if isinstance(img_cost, dict):
            # tiered (low, medium, high)
            cost += sum(img_cost.values()) * num_images
        else:
            cost += img_cost * num_images

    return cost


def process_media_directly(media_path: str, provider: str = None,
                          prompt: str = None, **kwargs) -> str:
    from pathlib import Path
    if not Path(media_path).exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    if not provider:
        for provider_name in get_available_providers():
            plugin_cls = get_llm_plugin(provider_name)
            if getattr(plugin_cls, "supports_vision", False) and plugin_cls.configured():
                provider = provider_name
                break
        if not provider:
            raise Exception("No vision-capable providers available")

    plugin_cls = get_llm_plugin(provider)
    if not plugin_cls:
        raise ValueError(f"Plugin not found for provider: {provider}")

    if hasattr(plugin_cls, "process_direct_media"):
        return plugin_cls.process_direct_media(media_path, prompt, **kwargs)
    else:
        raise NotImplementedError(f"Provider {provider} does not support direct media processing.")




if __name__ == "__main__":
    # Test available providers
    print("Testing available providers...")

    providers = get_available_providers()
    print(f"Available providers: {providers}")

    for provider in providers:
        test_provider(provider)

    # Test basic functionality if providers available
    if providers:
        try:
            test_prompt = "What is machine learning in one sentence?"
            response = call_llm(test_prompt, 'general')
            print(f"\nTest response: {response}")

            cost = estimate_cost(test_prompt, 'general')
            print(f"Estimated cost: ${cost:.6f}")

        except Exception as e:
            print(f"Test failed: {e}")
