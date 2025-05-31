# core/processor.py
"""
Main processor - coordinates extraction and generation
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple  # Added this line
from concurrent.futures import ThreadPoolExecutor, as_completed

from .extractor import extract_batch, get_supported_files
from .generator import generate_training_data, filter_by_quality
from config.settings import *

def process_files(file_paths: List[str],
                 mode: str = 'extract-only',
                 generators: List[str] = None,
                 include_vision: bool = False,
                 output_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """
    Process multiple files with the specified mode

    Args:
        file_paths: List of file paths to process
        mode: Processing mode ('extract-only', 'generate', 'full')
        generators: List of generators to use
        include_vision: Whether to process images with vision LLMs
        output_dir: Output directory

    Returns:
        Dictionary with processing results
    """

    start_time = time.time()

    # Get supported files
    if len(file_paths) == 1 and Path(file_paths[0]).is_dir():
        supported_files = get_supported_files(file_paths[0])
    else:
        supported_files = [f for f in file_paths if is_supported_format(f)]

    if not supported_files:
        return {
            'success': False,
            'error': 'No supported files found',
            'files_processed': 0
        }

    print(f"📄 Processing {len(supported_files)} files in {mode} mode...")

    # Extract content from all files
    extracted_data = extract_batch(supported_files, use_cache=USE_CACHE)

    results = {
        'mode': mode,
        'files_processed': len(supported_files),
        'successful': 0,
        'failed': 0,
        'output_dir': output_dir,
        'processing_time': 0,
        'stats': {}
    }

    if mode == 'extract-only':
        results.update(_process_extract_only(extracted_data, output_dir))

    elif mode in ['generate', 'full']:
        if not generators:
            generators = DEFAULT_GENERATORS if mode == 'generate' else list(GENERATORS.keys())

        results.update(_process_generate_data(
            extracted_data, generators, include_vision, output_dir
        ))

    results['processing_time'] = time.time() - start_time

    # Save processing summary
    _save_processing_summary(results, output_dir)

    return results

def _process_extract_only(extracted_data: Dict[str, Tuple[str, List]], output_dir: str) -> Dict:
    """Process files in extract-only mode"""

    extract_dir = Path(output_dir) / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    results = {'successful': 0, 'failed': 0, 'stats': {'text_files': 0, 'images': 0}}

    for file_path, (text, images) in extracted_data.items():
        try:
            file_name = Path(file_path).stem

            # Save text content
            if text:
                text_file = extract_dir / f"{file_name}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                results['stats']['text_files'] += 1

            # Save image metadata
            if images:
                images_file = extract_dir / f"{file_name}_images.json"

                # Prepare image data for JSON (remove binary data)
                json_safe_images = []
                for img in images:
                    img_info = {k: v for k, v in img.items() if k != 'data'}
                    img_info['has_data'] = 'data' in img
                    json_safe_images.append(img_info)

                with open(images_file, 'w', encoding='utf-8') as f:
                    json.dump(json_safe_images, f, indent=2, default=str)

                results['stats']['images'] += len(images)

            results['successful'] += 1
            print(f"✅ Extracted: {Path(file_path).name}")

        except Exception as e:
            results['failed'] += 1
            print(f"❌ Failed: {Path(file_path).name} - {e}")

    return results

def _process_generate_data(extracted_data: Dict[str, Tuple[str, List]],
                          generators: List[str],
                          include_vision: bool,
                          output_dir: str) -> Dict:
    """Process files with LLM generation"""

    # Check if we have LLM providers available
    from .llm_client import get_available_providers

    providers = get_available_providers()
    if not providers:
        return {
            'success': False,
            'error': 'No LLM providers configured',
            'successful': 0,
            'failed': len(extracted_data)
        }

    print(f"🤖 Using LLM provider: {providers[0]}")

    all_generated_data = {}
    results = {'successful': 0, 'failed': 0, 'stats': {}}

    # Process each file
    for file_path, (text, images) in extracted_data.items():
        if not text.strip():
            print(f"⚠️  Skipping {Path(file_path).name} - no text content")
            continue

        try:
            print(f"🔄 Processing {Path(file_path).name}...")

            # Generate training data
            generated_data = generate_training_data(
                text,
                generators=generators,
                images=images if include_vision else None
            )

            # Filter by quality
            filtered_data = filter_by_quality(generated_data, QUALITY_THRESHOLD)

            all_generated_data[file_path] = filtered_data
            results['successful'] += 1

        except Exception as e:
            print(f"❌ Error processing {Path(file_path).name}: {e}")
            results['failed'] += 1

    # Save generated data
    if all_generated_data:
        _save_generated_data(all_generated_data, output_dir, results)

    return results

def _save_generated_data(generated_data: Dict[str, Dict], output_dir: str, results: Dict):
    """Save generated training data to files"""

    output_path = Path(output_dir)

    # Save each data type separately
    for data_type in ['conversations', 'embeddings', 'qa_pairs', 'summaries', 'image_descriptions']:

        type_dir = output_path / data_type
        type_dir.mkdir(parents=True, exist_ok=True)

        all_items = []

        # Collect all items of this type
        for file_path, file_data in generated_data.items():
            if data_type in file_data:
                items = file_data[data_type]

                # Add source file info to each item
                for item in items:
                    if isinstance(item, dict):
                        item['source_file'] = Path(file_path).name

                all_items.extend(items)

        if all_items:
            # Save as JSONL
            output_file = type_dir / f"{data_type}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in all_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            results['stats'][data_type] = len(all_items)
            print(f"💾 Saved {len(all_items)} {data_type} to {output_file.name}")

def _save_processing_summary(results: Dict, output_dir: str):
    """Save processing summary to JSON file"""

    summary_file = Path(output_dir) / "summary.json"

    # Add timestamp
    results['timestamp'] = time.time()
    results['readable_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📊 Processing summary saved to {summary_file}")

def get_processing_summary(output_dir: str) -> Dict:
    """Load processing summary from file"""

    summary_file = Path(output_dir) / "summary.json"

    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    return {}

def save_results(data: Dict, output_path: str, format: str = 'jsonl'):
    """
    Save processing results in specified format

    Args:
        data: Data to save
        output_path: Output file path
        format: Output format ('jsonl', 'csv', 'json')
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == 'jsonl':
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(data, list):
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    elif format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    elif format == 'csv':
        import pandas as pd

        if isinstance(data, list) and data:
            # Flatten nested dictionaries for CSV
            flattened_data = []
            for item in data:
                if isinstance(item, dict):
                    flat_item = {}
                    for key, value in item.items():
                        if isinstance(value, (dict, list)):
                            flat_item[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            flat_item[key] = value
                    flattened_data.append(flat_item)
                else:
                    flattened_data.append({'data': item})

            df = pd.DataFrame(flattened_data)
            df.to_csv(output_file, index=False)
        else:
            # Single item or non-list data
            df = pd.DataFrame([data] if isinstance(data, dict) else [{'data': data}])
            df.to_csv(output_file, index=False)

    print(f"💾 Saved results to {output_file}")

# Utility functions for batch processing
def process_directory_parallel(directory: str,
                             max_workers: int = MAX_WORKERS,
                             **kwargs) -> Dict[str, Any]:
    """
    Process all files in a directory using parallel processing

    Args:
        directory: Directory to process
        max_workers: Maximum number of parallel workers
        **kwargs: Additional arguments for process_files

    Returns:
        Combined processing results
    """

    supported_files = get_supported_files(directory)

    if not supported_files:
        return {'error': 'No supported files found', 'files_processed': 0}

    # Split files into batches for parallel processing
    batch_size = max(1, len(supported_files) // max_workers)
    file_batches = [supported_files[i:i + batch_size]
                   for i in range(0, len(supported_files), batch_size)]

    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit processing tasks
        future_to_batch = {
            executor.submit(process_files, batch, **kwargs): batch
            for batch in file_batches
        }

        # Collect results
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"✅ Completed batch of {len(batch)} files")
            except Exception as e:
                print(f"❌ Batch processing failed: {e}")

    # Combine results
    combined_results = {
        'mode': kwargs.get('mode', 'extract-only'),
        'files_processed': sum(r.get('files_processed', 0) for r in all_results),
        'successful': sum(r.get('successful', 0) for r in all_results),
        'failed': sum(r.get('failed', 0) for r in all_results),
        'processing_time': sum(r.get('processing_time', 0) for r in all_results),
        'stats': {}
    }

    # Combine stats
    for result in all_results:
        if 'stats' in result:
            for key, value in result['stats'].items():
                combined_results['stats'][key] = combined_results['stats'].get(key, 0) + value

    return combined_results

if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else 'extract-only'

        print(f"Testing processor with: {input_path}, mode: {mode}")

        results = process_files([input_path], mode=mode)
        print(f"Results: {results}")
    else:
        print("Usage: python core/processor.py <input_path> [mode]")
