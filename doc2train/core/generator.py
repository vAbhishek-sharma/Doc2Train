# outputs/generator.py

from doc2train.core.registries.generator_registry import get_generator
from typing import Dict, List, Any, Optional
import random
from doc2train.core.extractor import chunk_text
import ipdb
from typing import Any, Dict, List, Optional

def generate_data(
    text: str,
    images: Optional[List[Any]],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Route to plugins for processing based on mode: 'extract', 'generate', or 'full'.
    """
    mode = config.get('mode')
    if mode == 'full':
        return _full_data_generation_mode(text, images, config)
    elif mode == 'generate':
        return _generate_data_generation_mode(text, images, config)
    else:
        return _media_only_data_generation_mode(text, images, config)


# -------------------
# Helper functions
# -------------------

def filter_by_quality(data: List[Dict], quality_threshold: float = 0.7) -> List[Dict]:
    """
    Filter generated data based on a quality score field (if present).
    Only keeps items with score >= threshold.
    """
    filtered = []
    for item in data:
        score = item.get("quality_score", 1.0)  # Default to 1.0 if not present
        if score >= quality_threshold:
            filtered.append(item)
    return filtered

def merge_generated_data(results: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Merge all generated outputs into a single flat list.
    Useful for certain output formats.
    """
    merged = []
    for data_list in results.values():
        merged.extend(data_list)
    return merged

def sample_data(data: List[Dict], n: int = 5, randomize: bool = True) -> List[Dict]:
    """
    Sample n items from the data for preview or debugging.
    """
    if not data:
        return []
    if randomize:
        return random.sample(data, min(len(data), n))
    else:
        return data[:n]

def get_data_stats(data: List[Dict]) -> Dict[str, Any]:
    """
    Return simple stats about generated data (count, avg length, etc.)
    """
    count = len(data)
    avg_len = sum(len(str(d)) for d in data) / count if count else 0
    return {"count": count, "avg_length": avg_len}



#  Generators mode handlers

# -------------------
# Mode handlers
# -------------------

def _full_data_generation_mode(
    text: str,
    images: Optional[List[Any]],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    # Generate both text and media data
    text_results = _text_data_generation(text, images, config)
    media_results = _media_data_generation(text, images, config)
    # Merge dictionaries
    results = {**text_results, **media_results}
    return results


def _media_only_data_generation_mode(
    text: str,
    images: Optional[List[Any]],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    # Generate only media data
    return _media_data_generation(text, images, config)


def _generate_data_generation_mode(
    text: str,
    images: Optional[List[Any]],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    # Generate only text data
    return _text_data_generation(text, images, config)


# -------------------
# Internal generators
# -------------------

def _text_data_generation(
    text: str,
    images: Optional[List[Any]],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    gen_types = config.get('text_generators', [])
    custom_prompts = config.get('custom_prompts', {})
    results: Dict[str, List[Dict[str, Any]]] = {t: [] for t in gen_types}

    for gen_type in gen_types:
        gen_cls = get_generator(gen_type)
        if not gen_cls:
            print(f"⚠️ No generator plugin registered for type: {gen_type}")
            continue
        plugin = gen_cls(config)
        prompt = custom_prompts.get(gen_type)

        items = plugin.generate(text, images=images, prompt_template=prompt)
        if isinstance(items, dict):
            results[gen_type].append(items)
        else:
            results[gen_type].extend(items or [])

    return results


def _media_data_generation(
    text: str,
    images: Optional[List[Any]],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    gen_types = config.get('media_generators', [])
    custom_prompts = config.get('custom_prompts', {})
    results: Dict[str, List[Dict[str, Any]]] = {t: [] for t in gen_types}

    for gen_type in gen_types:
        gen_cls = get_generator(gen_type)
        if not gen_cls:
            print(f"⚠️ No generator plugin registered for type: {gen_type}")
            continue
        plugin = gen_cls(config)
        prompt = custom_prompts.get(gen_type)

        items = plugin.generate(text, images=images, prompt_template=prompt)
        if isinstance(items, dict):
            results[gen_type].append(items)
        else:
            results[gen_type].extend(items or [])

    return results


