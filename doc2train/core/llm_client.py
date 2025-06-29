# core/llm_client.py
"""
Simple LLM client - handles calls to different AI providers
Supports OpenAI, DeepSeek, and local models
"""

from pathlib import Path
import openai
import requests
import json
import base64
import ipdb
from typing import Dict, Any, Optional, List  # Added this line
from doc2train.config.settings import *
from doc2train.core.registries.llm_registry import (
    get_llm_plugin,
    get_available_providers,
    list_llm_plugins
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



def call_vision_llm(prompt: str, image_data: dict, max_retries: int = 3) -> str:
    provider_info = _get_vision_provider()
    if not provider_info:
        raise Exception("No vision-capable provider available")
    plugin_cls = provider_info['plugin_cls']
    for attempt in range(max_retries):
        try:
            if hasattr(plugin_cls, "call_vision"):
                return plugin_cls.call_vision(prompt, image_data)
            else:
                raise Exception(f"Provider {provider_info['provider']} does not support vision API")
        except Exception as e:
            print(f"⚠️  Vision attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                ocr_text = image_data.get('ocr_text', '')
                if ocr_text:
                    return f"Image contains text: {ocr_text}"
                else:
                    return "Unable to process image"
    return "Image processing failed"

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
        plugin_cls = get_llm_plugin(name)
        if getattr(plugin_cls, "supports_vision", False):
            return {"provider": name, "plugin_cls": plugin_cls}
    return None

def _call_openai(prompt: str, provider_info: Dict) -> str:
    """Call OpenAI API"""
    config = provider_info['config']

    if not config.get('api_key'):
        raise Exception("OpenAI API key not configured")

    client = openai.OpenAI(
        api_key=config['api_key'],
        base_url=config.get('base_url', 'https://api.openai.com/v1')
    )

    response = client.chat.completions.create(
        model=provider_info['model'],
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.7
    )

    return response.choices[0].message.content

def _call_deepseek(prompt: str, provider_info: Dict) -> str:
    """Call DeepSeek API"""
    config = provider_info['config']

    if not config.get('api_key'):
        raise Exception("DeepSeek API key not configured")

    client = openai.OpenAI(
        api_key=config['api_key'],
        base_url=config.get('base_url', 'https://api.deepseek.com')
    )

    response = client.chat.completions.create(
        model=provider_info['model'],
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=8000,
        temperature=0.7
    )

    return response.choices[0].message.content

def _call_local(prompt: str, provider_info: Dict) -> str:
    """Call local LLM (Ollama, LM Studio, etc.)"""
    config = provider_info['config']

    # Try Ollama first
    ollama_url = config.get('ollama_url', 'http://localhost:11434')

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": provider_info['model'],
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Ollama API error: {response.status_code}")

    except requests.exceptions.RequestException:
        # Fallback to LM Studio format
        try:
            lm_studio_url = "http://localhost:1234/v1/chat/completions"

            response = requests.post(
                lm_studio_url,
                json={
                    "model": provider_info['model'],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"LM Studio API error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Local LLM not available: {e}")

def _call_openai_vision(prompt: str, image_data: Dict, provider_info: Dict) -> str:
    """Call OpenAI vision API"""
    config = provider_info['config']

    if not config.get('api_key'):
        raise Exception("OpenAI API key not configured")

    client = openai.OpenAI(
        api_key=config['api_key'],
        base_url=config.get('base_url', 'https://api.openai.com/v1')
    )

    # Prepare image data
    image_base64 = None

    if 'data' in image_data:
        # Raw image bytes
        image_base64 = base64.b64encode(image_data['data']).decode('utf-8')
    elif 'path' in image_data:
        # Image file path
        with open(image_data['path'], 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
    else:
        raise Exception("No image data provided")

    # Create vision message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=provider_info['model'],
        messages=messages,
        max_tokens=1000
    )

    return response.choices[0].message.content

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
