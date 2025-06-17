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
from typing import Dict, Any, Optional, List  # Added this line
from config.settings import *

def call_llm(prompt: str, task: str = 'general', max_retries: int = 3) -> str:
    """
    Call LLM with automatic provider selection and fallback
    """
    # Get the best provider/model for this task
    provider_info = _get_provider_for_task(task)

    for attempt in range(max_retries):
        try:
            if provider_info['provider'] == 'openai':
                return _call_openai(prompt, provider_info)
            elif provider_info['provider'] == 'deepseek':
                return _call_deepseek(prompt, provider_info)
            elif provider_info['provider'] == 'local':
                return _call_local(prompt, provider_info)
            else:
                raise ValueError(f"Unknown provider: {provider_info['provider']}")

        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Attempt {attempt + 1} failed: {error_msg}")

            # NEW: Handle specific API errors more gracefully
            if "insufficient_quota" in error_msg or "429" in error_msg:
                print(f"💰 OpenAI quota exceeded. Switching to fallback provider...")
                provider_info = _get_fallback_provider()
                continue
            elif "authentication" in error_msg.lower() or "401" in error_msg:
                print(f"🔑 Authentication failed for {provider_info['provider']}. Switching to fallback...")
                provider_info = _get_fallback_provider()
                continue

            if attempt < max_retries - 1:
                # Try fallback provider on retry
                provider_info = _get_fallback_provider()
            else:
                # NEW: On final failure, return a placeholder instead of crashing
                print(f"❌ All LLM attempts failed after {max_retries} retries")
                return '{"error": "LLM processing failed - continuing with extraction only"}'

    raise Exception("All retry attempts failed")
def call_vision_llm(prompt: str, image_data: Dict, max_retries: int = 3) -> str:
    """
    Call vision LLM to describe/analyze images

    Args:
        prompt: The prompt for image analysis
        image_data: Dictionary containing image data and metadata
        max_retries: Number of retry attempts

    Returns:
        Generated image description
    """
    # Get vision-capable provider
    provider_info = _get_vision_provider()

    if not provider_info:
        raise Exception("No vision-capable provider available")

    for attempt in range(max_retries):
        try:
            if provider_info['provider'] == 'openai':
                return _call_openai_vision(prompt, image_data, provider_info)
            else:
                raise ValueError(f"Vision not supported for: {provider_info['provider']}")

        except Exception as e:
            print(f"⚠️  Vision attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                continue
            else:
                # Fallback to OCR text if vision fails
                ocr_text = image_data.get('ocr_text', '')
                if ocr_text:
                    return f"Image contains text: {ocr_text}"
                else:
                    return "Unable to process image"

    return "Image processing failed"

def _get_provider_for_task(task: str) -> Dict[str, str]:
    """Get the best provider configuration for a specific task"""
    route = LLM_ROUTING.get(task, LLM_ROUTING['fallback'])
    provider_name, model_type = route.split('/')

    provider_config = LLM_PROVIDERS.get(provider_name, {})
    model_name = provider_config.get('models', {}).get(model_type)

    return {
        'provider': provider_name,
        'model': model_name,
        'config': provider_config
    }

def _get_fallback_provider() -> Dict[str, str]:
    """Get fallback provider configuration"""
    return _get_provider_for_task('fallback')

def _get_vision_provider() -> Optional[Dict[str, str]]:
    """Get a vision-capable provider"""
    for provider_name, config in LLM_PROVIDERS.items():
        if config.get('models', {}).get('vision'):
            return {
                'provider': provider_name,
                'model': config['models']['vision'],
                'config': config
            }
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

def test_provider(provider_name: str) -> bool:
    """
    Test if a provider is working correctly

    Args:
        provider_name: Name of provider to test

    Returns:
        True if provider is working
    """
    try:
        test_prompt = "Hello, please respond with 'OK' if you can understand this message."

        provider_info = {
            'provider': provider_name,
            'model': LLM_PROVIDERS[provider_name]['models']['text'],
            'config': LLM_PROVIDERS[provider_name]
        }

        response = call_llm(test_prompt, 'general')

        if response and len(response.strip()) > 0:
            print(f"✅ {provider_name} is working")
            return True
        else:
            print(f"❌ {provider_name} returned empty response")
            return False

    except Exception as e:
        print(f"❌ {provider_name} test failed: {e}")
        return False

def get_available_providers() -> List[str]:
    """
    Get list of available and working providers

    Returns:
        List of working provider names
    """
    available = []

    for provider_name in LLM_PROVIDERS.keys():
        if provider_name == 'local':
            # Skip local providers in quick check
            continue

        try:
            # Quick check for API key
            config = LLM_PROVIDERS[provider_name]
            if config.get('api_key'):
                available.append(provider_name)
        except:
            continue

    return available

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for cost calculation

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 4 characters for English
    return len(text) // 4

def estimate_cost(text: str, task: str, provider: str = None) -> float:
    """
    Estimate cost for processing text

    Args:
        text: Text to process
        task: Task type
        provider: Specific provider (optional)

    Returns:
        Estimated cost in USD
    """
    if not provider:
        provider_info = _get_provider_for_task(task)
        provider = provider_info['provider']

    tokens = estimate_tokens(text)

    # Rough cost estimates (as of 2024)
    cost_per_1k_tokens = {
        'openai': 0.00015,  # GPT-4o-mini
        'deepseek': 0.00007,  # DeepSeek
        'local': 0.0  # Free
    }

    rate = cost_per_1k_tokens.get(provider, 0.0001)
    return (tokens / 1000) * rate


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
