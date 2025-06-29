# outputs/generator.py

from doc2train.core.registries.generator_registry import get_generator
from typing import Dict, List, Any, Optional
import random
from doc2train.core.extractor import chunk_text
from doc2train.core.registries.generator_registry import get_generator
import ipdb
from typing import Any, Dict, List, Optional

def generate_data(
    text: str,
    images: Optional[List[Any]],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Chunk input, route to plugins, merge results.

    Parameters:
      - text: the raw text to process
      - images: Optional list of vision inputs
      - config: global configuration dict, expected keys:
          • 'mode': 'extract', 'generate', or 'full'
          • 'text': {'generators': List[str]}
          • 'media': {'generators': List[str]}
          • 'custom_prompts': {generator_name: prompt_template}
    """
    # Determine which generators to run based on mode
    if config.get('mode') in ['generate', 'full']:
        gen_types = config.get('text', {}).get('generators', [])
    else:
        gen_types = config.get('media', {}).get('generators', [])

    custom_prompts = config.get('custom_prompts', {})
    results: Dict[str, List[Dict[str, Any]]] = {t: [] for t in gen_types}

    # Break the text into chunks
    chunks = chunk_text(text, config=config)

    for gen_type in gen_types:
        gen_cls = get_generator(gen_type)
        if not gen_cls:
            print(f"⚠️ No generator plugin registered for type: {gen_type}")
            continue

        plugin = gen_cls(config)
        prompt = custom_prompts.get(gen_type)

        for chunk in chunks:
            # Each plugin.generate returns List[Dict] or dict
            items = plugin.generate(chunk, images=images, prompt_template=prompt)
            if isinstance(items, dict):
                results[gen_type].append(items)
            else:
                results[gen_type].extend(items or [])

    return results


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
