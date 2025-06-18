# outputs/generator.py

from doc2train.core.registries.generator_registry import get_generator
from typing import Dict, List, Any

def generate_data(text: str, gen_types: List[str], config: Dict) -> Dict[str, List[Dict]]:
    """
    Chunk input, route to plugins, merge results.
    """
    # Use your existing chunking logic (from extractor, etc.)
    from doc2train.core.extractor import chunk_text
    chunks = chunk_text(text, config=config)
    results = {t: [] for t in gen_types}

    for gen_type in gen_types:
        gen_cls = get_generator(gen_type)
        if not gen_cls:
            print(f"⚠️ No generator plugin for: {gen_type}")
            continue

        plugin = gen_cls(config)
        prompt = (
            config.get("prompts", {})
            .get("custom", {})
            .get(gen_type)
        )

        for chunk in chunks:
            output = plugin.generate(chunk, gen_type, prompt_template=prompt)
            # Expect plugin to return a dict: {gen_type: [items]}
            if output and isinstance(output, dict) and gen_type in output:
                results[gen_type].extend(output[gen_type])
            elif output:
                results[gen_type].extend(output if isinstance(output, list) else [output])

    return results


def generate_training_data(text: str, generators: List[str] = None, images: List[Dict] = None,
                         custom_prompts: Dict[str, str] = None, use_async: bool = True) -> Dict[str, List]:
    """
    Generate training data from extracted content

    Args:
        text: Extracted text content
        generators: List of generators to use (conversations, embeddings, qa_pairs, summaries)
        images: Optional list of images to process
        custom_prompts: Optional custom prompts for generators
        use_async: Use async processing for faster generation

    Returns:
        Dictionary with generated training data by type
    """
    if not generators:
        generators = DEFAULT_GENERATORS

    # Use custom prompts if provided, otherwise use defaults
    prompts = custom_prompts or get_default_prompts()

    results = {}

    # Process text with each requested generator
    if generators:
        if use_async:
            text_results = _generate_text_data_async(text, generators, prompts)
        else:
            text_results = _generate_text_data_sync(text, generators, prompts)

        if text_results:
            results.update(text_results)

    # Process images if provided and requested
    if images and EXTRACT_IMAGES:
        try:
            results['image_descriptions'] = _generate_image_descriptions(images)
            print(f"✅ Generated {len(results['image_descriptions'])} image descriptions")
        except Exception as e:
            print(f"❌ Error processing images: {e}")
            results['image_descriptions'] = []

    return results

def _generate_text_data_async(text: str, generators: List[str], prompts: Dict[str, str]) -> Dict[str, List]:
    """Generate text data using async LLM calls for speed"""
    from core.extractor import chunk_text

    chunks = chunk_text(text)
    if TEST_MODE:
        chunks = chunks[:TEST_MAX_CHUNKS]

    results = {gen: [] for gen in generators}

    # Use ThreadPoolExecutor for parallel LLM calls
    max_workers = min(8, len(generators) * len(chunks))  # Limit concurrent API calls

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all generation tasks
        future_to_task = {}

        for generator_type in generators:
            if generator_type not in GENERATORS:
                print(f"⚠️ Unknown generator: {generator_type}")
                continue

            for i, chunk in enumerate(chunks):
                future = executor.submit(_generate_single_type, chunk, generator_type, prompts.get(generator_type, ''))
                future_to_task[future] = (generator_type, i)

        # Collect results as they complete
        for future in as_completed(future_to_task):
            generator_type, chunk_index = future_to_task[future]

            try:
                generated_items = future.result()
                if generated_items:
                    results[generator_type].extend(generated_items)

            except Exception as e:
                if DEBUG:
                    print(f"Error generating {generator_type} for chunk {chunk_index}: {e}")

    # Print results
    for generator_type in generators:
        if results[generator_type]:
            print(f"✅ Generated {len(results[generator_type])} {generator_type}")

    return results

def _generate_text_data_sync(text: str, generators: List[str], prompts: Dict[str, str]) -> Dict[str, List]:
    """Generate text data using synchronous LLM calls (original method)"""
    from core.extractor import chunk_text

    chunks = chunk_text(text)
    if TEST_MODE:
        chunks = chunks[:TEST_MAX_CHUNKS]

    results = {gen: [] for gen in generators}

    # Process generators sequentially
    for generator_type in generators:
        if generator_type not in GENERATORS:
            print(f"⚠️ Unknown generator: {generator_type}")
            continue

        for i, chunk in enumerate(chunks):
            generated_items = _generate_single_type(chunk, generator_type, prompts.get(generator_type, ''))
            if generated_items:
                results[generator_type].extend(generated_items)

        if results[generator_type]:
            print(f"✅ Generated {len(results[generator_type])} {generator_type}")

    return results

def _generate_single_type(chunk: str, generator_type: str, prompt_template: str) -> List[Dict]:
    """Generate single type of training data for a chunk"""
    from core.llm_client import call_llm

    # Build prompt dynamically
    prompt = _build_prompt(chunk, generator_type, prompt_template)

    try:
        response = call_llm(prompt, task=generator_type)
        parsed = _parse_json_response(response)

        if parsed:
            # Extract the appropriate data based on generator type
            if generator_type == 'conversations' and 'conversations' in parsed:
                return parsed['conversations']
            elif generator_type == 'embeddings' and 'pairs' in parsed:
                return parsed['pairs']
            elif generator_type == 'qa_pairs' and 'qa_pairs' in parsed:
                return parsed['qa_pairs']
            elif generator_type == 'summaries' and 'summary' in parsed:
                return [{'summary': parsed['summary'], 'original_text': chunk[:200] + "..."}]

    except Exception as e:
        if DEBUG:
            print(f"Error generating {generator_type}: {e}")

    return []

def _build_prompt(chunk: str, generator_type: str, prompt_template: str) -> str:
    """Build dynamic prompt with templates"""

    # Default templates if none provided
    if not prompt_template:
        prompt_template = get_default_prompts().get(generator_type, '')

    # Format templates for each generator type
    if generator_type == 'conversations':
        format_example = '''{{"conversations": [
    {{"messages": [
        {{"role": "user", "content": "user question"}},
        {{"role": "assistant", "content": "AI response"}}
    ]}},
    {{"messages": [
        {{"role": "user", "content": "follow-up question"}},
        {{"role": "assistant", "content": "AI response"}}
    ]}}
]}}'''

    elif generator_type == 'qa_pairs':
        format_example = '''{{"qa_pairs": [
    {{"question": "What is...", "answer": "Complete answer based on content"}},
    {{"question": "How does...", "answer": "Detailed explanation"}}
]}}'''

    elif generator_type == 'embeddings':
        format_example = '''{{"pairs": [
    {{"sentence1": "first sentence", "sentence2": "similar meaning sentence", "similarity": 0.9}},
    {{"sentence1": "different sentence", "sentence2": "unrelated sentence", "similarity": 0.1}}
]}}'''

    elif generator_type == 'summaries':
        format_example = '''{{"summary": "Your concise summary here"}}'''

    else:
        format_example = '{{"data": "generated content"}}'

    # Build the complete prompt
    full_prompt = f"""{prompt_template}

Content:
{chunk}

Format your response as JSON:
{format_example}
"""

    return full_prompt

def get_default_prompts() -> Dict[str, str]:
    """Get default prompts (can be overridden)"""
    return {
        'conversations': "Based on this content, create a natural multi-turn conversation between a user and an AI assistant. Make it educational and engaging. Include 3-4 exchanges (user question -> AI response).",

        'qa_pairs': "Create specific questions that can be answered from this content. Make sure answers are complete and accurate.",

        'embeddings': "From this content, create pairs of sentences that have similar meanings but different wording. Also create some pairs with different meanings for contrast.",

        'summaries': "Create a concise summary of this content, highlighting the key points.",

        'image_descriptions': "Describe this image in detail. What do you see? What might this image be used to illustrate or explain? If there's text in the image, include it in your description."
    }

def set_custom_prompts(custom_prompts: Dict[str, str]):
    """Set custom prompts globally"""
    global SYSTEM_PROMPTS
    SYSTEM_PROMPTS.update(custom_prompts)

def _generate_conversations(text: str) -> List[Dict]:
    """Generate conversational training data"""
    from core.llm_client import call_llm
    from core.extractor import chunk_text

    chunks = chunk_text(text)
    conversations = []

    for i, chunk in enumerate(chunks):
        if TEST_MODE and i >= TEST_MAX_CHUNKS:
            break

        prompt = f"""
            {SYSTEM_PROMPTS['conversations']}

            Based on this content, create a natural multi-turn conversation between a user and an AI assistant.
            Make it educational and engaging. Include 3-4 exchanges (user question -> AI response).

            Content:
            {chunk}

            Format your response as JSON:
            {{"conversations": [
                {{"messages": [
                    {{"role": "user", "content": "user question"}},
                    {{"role": "assistant", "content": "AI response"}}
                ]}},
                {{"messages": [
                    {{"role": "user", "content": "follow-up question"}},
                    {{"role": "assistant", "content": "AI response"}}
                ]}}
            ]}}
            """

        try:
            response = call_llm(prompt, task='conversations')
            parsed = _parse_json_response(response)

            if parsed and 'conversations' in parsed:
                conversations.extend(parsed['conversations'])

        except Exception as e:
            if DEBUG:
                print(f"Error generating conversation for chunk {i}: {e}")

    return conversations

def _generate_embeddings(text: str) -> List[Dict]:
    """Generate semantic similarity pairs for embedding training"""
    from core.llm_client import call_llm
    from core.extractor import chunk_text

    chunks = chunk_text(text)
    embedding_pairs = []

    for i, chunk in enumerate(chunks):
        if TEST_MODE and i >= TEST_MAX_CHUNKS:
            break

        prompt = f"""
            {SYSTEM_PROMPTS['embeddings']}

            From this content, create pairs of sentences that have similar meanings but different wording.
            Also create some pairs with different meanings for contrast.

            Content:
            {chunk}

            Format as JSON:
            {{"pairs": [
                {{"sentence1": "first sentence", "sentence2": "similar meaning sentence", "similarity": 0.9}},
                {{"sentence1": "different sentence", "sentence2": "unrelated sentence", "similarity": 0.1}}
            ]}}
            """

        try:
            response = call_llm(prompt, task='embeddings')
            parsed = _parse_json_response(response)

            if parsed and 'pairs' in parsed:
                embedding_pairs.extend(parsed['pairs'])

        except Exception as e:
            if DEBUG:
                print(f"Error generating embeddings for chunk {i}: {e}")

    return embedding_pairs

def _generate_qa_pairs(text: str) -> List[Dict]:
    """Generate question-answer pairs"""
    from core.llm_client import call_llm
    from core.extractor import chunk_text

    chunks = chunk_text(text)
    qa_pairs = []

    for i, chunk in enumerate(chunks):
        if TEST_MODE and i >= TEST_MAX_CHUNKS:
            break

        prompt = f"""
{SYSTEM_PROMPTS['qa_pairs']}

Create specific questions that can be answered from this content.
Make sure answers are complete and accurate.

Content:
{chunk}

Format as JSON:
{{"qa_pairs": [
    {{"question": "What is...", "answer": "Complete answer based on content"}},
    {{"question": "How does...", "answer": "Detailed explanation"}}
]}}
"""

        try:
            response = call_llm(prompt, task='qa_pairs')
            parsed = _parse_json_response(response)

            if parsed and 'qa_pairs' in parsed:
                qa_pairs.extend(parsed['qa_pairs'])

        except Exception as e:
            if DEBUG:
                print(f"Error generating QA pairs for chunk {i}: {e}")

    return qa_pairs

def _generate_summaries(text: str) -> List[Dict]:
    """Generate document summaries"""
    from core.llm_client import call_llm
    from core.extractor import chunk_text

    chunks = chunk_text(text)
    summaries = []

    for i, chunk in enumerate(chunks):
        if TEST_MODE and i >= TEST_MAX_CHUNKS:
            break

        prompt = f"""
{SYSTEM_PROMPTS['summaries']}

Create a concise summary of this content, highlighting the key points.

Content:
{chunk}

Format as JSON:
{{"summary": "Your concise summary here"}}
"""

        try:
            response = call_llm(prompt, task='summaries')
            parsed = _parse_json_response(response)

            if parsed and 'summary' in parsed:
                summaries.append({
                    'original_text': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    'summary': parsed['summary']
                })

        except Exception as e:
            if DEBUG:
                print(f"Error generating summary for chunk {i}: {e}")

    return summaries

def _generate_image_descriptions(images: List[Dict]) -> List[Dict]:
    """Generate descriptions for images using vision models"""
    from core.llm_client import call_vision_llm

    descriptions = []

    for i, image_data in enumerate(images):
        if TEST_MODE and i >= 2:  # Limit in test mode
            break

        try:
            prompt = """
Describe this image in detail. What do you see? What might this image be used to illustrate or explain?
If there's text in the image, include it in your description.
Make your description useful for someone creating training data.
"""

            description = call_vision_llm(prompt, image_data)

            descriptions.append({
                'image_path': image_data.get('path', f'image_{i}'),
                'description': description,
                'context': image_data.get('context', ''),
                'ocr_text': image_data.get('ocr_text', '')
            })

        except Exception as e:
            if DEBUG:
                print(f"Error describing image {i}: {e}")

    return descriptions

def _parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON response from LLM, handling common formatting issues"""
    if not response:
        return None

    # Try direct parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    if '```json' in response:
        try:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end > start:
                json_text = response[start:end].strip()
                return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # Try to extract JSON from the response
    try:
        # Find first { and last }
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_text = response[start:end]
            return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    if DEBUG:
        print(f"Failed to parse JSON response: {response[:200]}...")

    return None

def validate_generated_data(data: Dict[str, List]) -> Dict[str, int]:
    """Validate generated training data quality"""
    metrics = {}

    for data_type, items in data.items():
        valid_items = 0
        total_items = len(items)

        for item in items:
            if _is_valid_item(item, data_type):
                valid_items += 1

        metrics[data_type] = {
            'total': total_items,
            'valid': valid_items,
            'quality_score': valid_items / total_items if total_items > 0 else 0
        }

    return metrics

def _is_valid_item(item: Dict, data_type: str) -> bool:
    """Check if a generated item meets quality requirements"""

    if data_type == 'conversations':
        if 'messages' not in item:
            return False

        messages = item['messages']
        if len(messages) < 2:
            return False

        for msg in messages:
            if not msg.get('content') or len(msg['content'].strip()) < 10:
                return False

        return True

    elif data_type == 'embeddings':
        required_fields = ['sentence1', 'sentence2', 'similarity']
        if not all(field in item for field in required_fields):
            return False

        similarity = item.get('similarity', 0)
        if not isinstance(similarity, (int, float)) or similarity < 0 or similarity > 1:
            return False

        for field in ['sentence1', 'sentence2']:
            if len(item[field].strip()) < 5:
                return False

        return True

    elif data_type == 'qa_pairs':
        if 'question' not in item or 'answer' not in item:
            return False

        if len(item['question'].strip()) < 5 or len(item['answer'].strip()) < 10:
            return False

        return True

    elif data_type == 'summaries':
        if 'summary' not in item:
            return False

        if len(item['summary'].strip()) < 20:
            return False

        return True

    elif data_type == 'image_descriptions':
        if 'description' not in item:
            return False

        if len(item['description'].strip()) < 10:
            return False

        return True

    return False

def filter_by_quality(data: Dict[str, List], threshold: float = QUALITY_THRESHOLD) -> Dict[str, List]:
    """Filter generated data by quality threshold"""
    filtered_data = {}

    for data_type, items in data.items():
        valid_items = []

        for item in items:
            if _is_valid_item(item, data_type):
                valid_items.append(item)

        quality_score = len(valid_items) / len(items) if items else 0

        if quality_score >= threshold:
            filtered_data[data_type] = valid_items
            if VERBOSE:
                print(f"✅ {data_type}: {len(valid_items)}/{len(items)} items passed quality check")
        else:
            if VERBOSE:
                print(f"⚠️ {data_type}: Quality too low ({quality_score:.2f} < {threshold})")

    return filtered_data

def estimate_cost(text: str, generators: List[str], images: List[Dict] = None) -> Dict[str, float]:
    """Estimate processing costs for different providers"""
    from core.extractor import count_tokens_estimate

    # Rough token estimates
    text_tokens = count_tokens_estimate(text)

    # Multiply by generators (each generator processes the text)
    total_text_tokens = text_tokens * len(generators)

    # Add image processing tokens (vision models)
    image_tokens = len(images) * 1000 if images else 0  # ~1000 tokens per image

    # Cost estimates (rough, as of 2024)
    cost_estimates = {
        'openai': {
            'text_cost': total_text_tokens * 0.00015 / 1000,  # GPT-4o-mini pricing
            'image_cost': image_tokens * 0.01 / 1000,  # GPT-4o vision pricing
            'total': 0
        },
        'deepseek': {
            'text_cost': total_text_tokens * 0.00007 / 1000,  # DeepSeek pricing
            'image_cost': 0,  # No vision model
            'total': 0
        }
    }

    # Calculate totals
    for provider in cost_estimates:
        cost_estimates[provider]['total'] = (
            cost_estimates[provider]['text_cost'] +
            cost_estimates[provider]['image_cost']
        )

    return cost_estimates

# Batch processing functions
def generate_batch(file_data: Dict[str, tuple], generators: List[str] = None) -> Dict[str, Dict]:
    """Generate training data for multiple files"""
    results = {}

    for file_path, (text, images) in file_data.items():
        try:
            print(f"🔄 Processing {Path(file_path).name}...")

            # Generate training data
            generated_data = generate_training_data(text, generators, images)

            # Filter by quality
            filtered_data = filter_by_quality(generated_data)

            results[file_path] = filtered_data

        except Exception as e:

            print(f"❌ Error processing {file_path}: {e}")
            results[file_path] = {}

    return results

# Utility functions for working with generated data
def merge_generated_data(data_list: List[Dict]) -> Dict[str, List]:
    """Merge generated data from multiple sources"""
    merged = {}

    for data in data_list:
        for data_type, items in data.items():
            if data_type not in merged:
                merged[data_type] = []
            merged[data_type].extend(items)

    return merged

def sample_data(data: Dict[str, List], sample_size: int = 10) -> Dict[str, List]:
    """Sample a subset of generated data for review"""
    import random

    sampled = {}

    for data_type, items in data.items():
        if len(items) <= sample_size:
            sampled[data_type] = items
        else:
            sampled[data_type] = random.sample(items, sample_size)

    return sampled

def get_data_stats(data: Dict[str, List]) -> Dict[str, Dict]:
    """Get statistics about generated data"""
    stats = {}

    for data_type, items in data.items():
        if not items:
            stats[data_type] = {'count': 0}
            continue

        count = len(items)

        if data_type == 'conversations':
            total_messages = sum(len(item.get('messages', [])) for item in items)
            avg_messages = total_messages / count if count > 0 else 0

            stats[data_type] = {
                'count': count,
                'total_messages': total_messages,
                'avg_messages_per_conversation': round(avg_messages, 1)
            }

        elif data_type == 'embeddings':
            similarities = [item.get('similarity', 0) for item in items]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0

            stats[data_type] = {
                'count': count,
                'avg_similarity': round(avg_similarity, 2),
                'high_similarity': len([s for s in similarities if s > 0.8]),
                'low_similarity': len([s for s in similarities if s < 0.3])
            }

        else:
            stats[data_type] = {'count': count}

    return stats

if __name__ == "__main__":
    # Simple test
    test_text = """
    Machine learning is a subset of artificial intelligence that focuses on creating systems
    that can learn and improve from experience without being explicitly programmed.
    There are three main types: supervised learning, unsupervised learning, and reinforcement learning.
    """

    print("Testing training data generation...")
    try:
        results = generate_training_data(test_text, ['conversations', 'qa_pairs'])

        for data_type, items in results.items():
            print(f"{data_type}: {len(items)} items generated")

        # Show stats
        stats = get_data_stats(results)
        print("\nStatistics:")
        for data_type, stat in stats.items():
            print(f"  {data_type}: {stat}")

    except Exception as e:
        print(f"Test failed: {e}")
