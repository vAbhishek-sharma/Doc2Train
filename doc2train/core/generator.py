# outputs/generator.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from doc2train.core.registries.generator_registry import get_generator
from typing import Dict, List, Any, Optional
import random
from doc2train.core.extractor import chunk_text
from doc2train.utils.resource_manager import resource_manager


def generate_data(
        *,
    text: str,
    images: Optional[List[Any]],
    audio: Optional[Any],
    video: Optional[Any],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Route to plugins for processing based on mode: 'extract', 'generate', or 'full'.
    """
    mode = config.get('mode')
    if mode == 'full':
        return _full_data_generation_mode(text, images, audio, video, config)
    elif mode == 'generate':
        return _generate_data_generation_mode(text, images, audio, video, config)
    else:
        return _media_only_data_generation_mode(text, images, audio, video, config)


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


# -------------------
# Mode handlers
# -------------------

def _full_data_generation_mode(
    text: str,
    images: Optional[List[Any]],
    audio: Optional[Any],
    video: Optional[Any],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    # Generate both text and media data
    text_results = _text_data_generation(text, images, audio, video, config)
    media_results = _media_data_generation(text, images, audio, video, config)
    # Merge dictionaries
    return {**text_results, **media_results}

def _media_only_data_generation_mode(
    text: str,
    images: Optional[List[Any]],
    audio: Optional[Any],
    video: Optional[Any],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    # Generate only media data
    return _media_data_generation(text, images, audio, video, config)

def _generate_data_generation_mode(
    text: str,
    images: Optional[List[Any]],
    audio: Optional[Any],
    video: Optional[Any],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    # Generate only text data
    return _text_data_generation(text, images, audio, video, config)


# -------------------
# Internal generators
# -------------------

def _text_data_generation(
    text: str,
    images: Any,
    audio: Any,
    video: Any,
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Choose sequential vs thread‑pool based on config['llm']['use_async'].
    """
    llm_cfg  = config.get('llm', {})
    use_async= llm_cfg.get('use_async', True)

    if use_async:
        return _threadpool_text_generation(text, images, audio, video, config)
    else:
        return _sequential_text_generation(text, images, audio, video, config)

def _media_data_generation(
    text: str,
    images: Optional[List[Any]],
    audio: Optional[Any],
    video: Optional[Any],
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

        items = plugin.generate(
            text=text,
            input_type='vision',
            images=images,
            audio=audio,
            video=video,
            prompt_template=prompt
        )
        if isinstance(items, dict):
            results[gen_type].append(items)
        else:
            results[gen_type].extend(items or [])

    return results


def _generate_single_chunk(
    chunk: str,
    gen_type: str,
    prompt: str,
    images: Any,
    audio: Any,
    video: Any,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Instantiate plugin and call generate() once."""
    gen_cls = get_generator(gen_type)
    if not gen_cls:
        raise ValueError(f"No generator plugin for: {gen_type}")
    plugin = gen_cls(config)
    items = plugin.generate(
        text=chunk,
        input_type='text',
        images=images,
        audio=audio,
        video=video,
        prompt_template=prompt
    )
    if isinstance(items, dict):
        return [items]
    return items or []


def _sequential_text_generation(
    text: str,
    images: Any,
    audio: Any,
    video: Any,
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    gens   = config.get('text_generators', [])
    prompts= config.get('custom_prompts', {})
    chunks = chunk_text(text)
    results = {g: [] for g in gens}

    for gen_type in gens:
        prompt = prompts.get(gen_type, '')
        for idx, chunk in enumerate(chunks):
            try:
                items = _generate_single_chunk(chunk, gen_type, prompt, images, audio, video, config)
                results[gen_type].extend(items)
            except Exception as e:
                print(f"⚠️ [seq] {gen_type}@chunk{idx} failed: {e}")

    return results


def _threadpool_text_generation(
    text: str,
    images: Any,
    audio: Any,
    video: Any,
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    gens   = config.get('text_generators', [])
    prompts= config.get('custom_prompts', {})
    chunks = chunk_text(text)
    results = {g: [] for g in gens}

    total_tasks = max(len(gens) * len(chunks), 1)
    max_workers = resource_manager.get_optimal_workers(total_tasks)

    def task(chunk, gen_type, prompt):
        return gen_type, _generate_single_chunk(chunk, gen_type, prompt, images, audio, video, config)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        for gen_type in gens:
            prompt = prompts.get(gen_type, '')
            for chunk in chunks:
                fut = exe.submit(task, chunk, gen_type, prompt)
                futures[fut] = gen_type

        for fut in as_completed(futures):
            gen_type = futures[fut]
            try:
                _, items = fut.result()
                results[gen_type].extend(items)
            except Exception as e:
                print(f"⚠️ [pool] {gen_type} task failed: {e}")

    return results


