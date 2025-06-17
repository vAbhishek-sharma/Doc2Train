# utils/validation.py
"""
Complete validation system for Doc2Train v2.0 Enhanced
Input validation, content quality assessment, and configuration validation
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import ipdb
#TO RENAME LATER
def validate_input_and_files(config) -> bool:
    """
    Enhanced input validation with detailed error reporting

    Args:
        args: Parsed command line arguments

    Returns:
        True if valid, prints errors and returns False if invalid
    """
    errors = []
    warnings = []

    # Validate input path
    input_path = Path(config.get('input_path'))
    ipdb.set_trace()
    if not input_path.exists():
        errors.append(f"Input path does not exist: {config.get('input_path')}")
    else:
        # Check if it's a file or directory
        if input_path.is_file():
            if not is_supported_file(str(input_path)):
                warnings.append(f"File type may not be supported: {input_path.suffix}")
        elif input_path.is_dir():
            # Check if directory has any supported files
            supported_files = find_supported_files(str(input_path))
            if not supported_files:
                errors.append(f"No supported files found in directory: {config.get('input_path')}")
            elif len(supported_files) > 1000:
                warnings.append(f"Large number of files detected: {len(supported_files)}. Consider using batch processing.")

    # Validate output directory
    output_dir = Path(config.get('output_dir'))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory {config.get('output_dir')}: {e}")

    # Check disk space
    if output_dir.exists():
        free_space_in_gigabytes = get_free_disk_space(str(output_dir)) / (1024**3)
        if free_space_in_gigabytes < 1.0:  # Less than 1GB free
            warnings.append(f"Low disk space: {free_space_in_gigabytes:.1f}GB available")

    # Validate LLM requirements for generation modes
    if config.get('mode') in ['generate', 'full', 'direct_to_llm']:
        llm_errors = validate_llm_configuration(config)
        errors.extend(llm_errors)

    # Show validation results
    if warnings:
        print("⚠️ Validation warnings:")
        for warning in warnings:
            print(f"  • {warning}")

    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  • {error}")
        return False

    print("✅ Input validation passed")
    return True

def validate_llm_configuration(config) -> List[str]:
    """Validate LLM configuration for generation modes"""
    errors = []

    try:
        from core.llm_client import get_available_providers
        providers = get_available_providers()

        if not providers:
            errors.append("No LLM providers configured. Set API keys in .env file.")
        else:
            print(f"✅ Available LLM providers: {', '.join(providers)}")

            # Validate specific provider if requested
        if config.get('provider') and config['provider'] not in providers:
            errors.append(f"Requested provider '{config['provider']}' not available")

    except ImportError:
        errors.append("LLM client not available")

    return errors

def validate_extraction_quality(text: str, images: List[Dict], config: Dict) -> bool:
    """
    Validate extraction quality based on configuration thresholds

    Args:
        text: Extracted text content
        images: List of extracted images
        config: Processing configuration

    Returns:
        True if quality meets thresholds
    """
    # Get quality thresholds
    min_text_length = config.get('min_text_length', 0)
    quality_threshold = config.get('quality_threshold', 0.0)

    # Text quality checks
    text_quality_score = assess_text_quality(text)

    # Check minimum text length
    if len(text.strip()) < min_text_length:
        if config.get('verbose'):
            print(f"   ⚠️ Text too short: {len(text)} < {min_text_length}")
        return False

    # Check text quality score
    if text_quality_score < quality_threshold:
        if config.get('verbose'):
            print(f"   ⚠️ Text quality too low: {text_quality_score:.2f} < {quality_threshold}")
        return False

    # Image quality checks
    valid_images = 0
    for img in images:
        if validate_image_quality(img, config):
            valid_images += 1

    # Require at least some content
    has_sufficient_content = (
        len(text.strip()) > 0 or
        valid_images > 0 or
        config.get('allow_low_quality', False)
    )

    return has_sufficient_content

def assess_text_quality(text: str) -> float:
    """
    Assess text quality and return score between 0.0 and 1.0

    Args:
        text: Text content to assess

    Returns:
        Quality score (0.0 = poor, 1.0 = excellent)
    """
    if not text.strip():
        return 0.0

    scores = []

    # Length score (longer text generally better)
    length_score = min(1.0, len(text) / 1000)  # Normalize to 1000 chars
    scores.append(length_score * 0.2)

    # Word count score
    words = text.split()
    word_count_score = min(1.0, len(words) / 100)  # Normalize to 100 words
    scores.append(word_count_score * 0.2)

    # Average word length (detect OCR garbage)
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
        # Optimal range is 4-8 characters per word
        if 4 <= avg_word_length <= 8:
            word_length_score = 1.0
        elif avg_word_length < 4:
            word_length_score = avg_word_length / 4
        else:
            word_length_score = max(0.1, 8 / avg_word_length)
        scores.append(word_length_score * 0.3)

    # Sentence structure score
    sentences = re.split(r'[.!?]+', text)
    sentence_score = min(1.0, len(sentences) / 10)  # Normalize to 10 sentences
    scores.append(sentence_score * 0.2)

    # Character diversity score (avoid repetitive content)
    unique_chars = len(set(text.lower()))
    char_diversity = min(1.0, unique_chars / 26)  # Normalize to alphabet
    scores.append(char_diversity * 0.1)

    return sum(scores)

def validate_image_quality(image: Dict, config: Dict) -> bool:
    """
    Validate image quality based on configuration

    Args:
        image: Image dictionary with metadata
        config: Processing configuration

    Returns:
        True if image meets quality requirements
    """
    # Size validation
    min_size = config.get('min_image_size', 0)
    if 'dimensions' in image:
        width, height = image['dimensions']
        if width * height < min_size:
            return False

    # Quality score validation
    quality_threshold = config.get('quality_threshold', 0.0)
    if image.get('quality_score', 1.0) < quality_threshold:
        return False

    # Single color validation
    if config.get('skip_single_color_images') and image.get('is_single_color', False):
        return False

    return True

def is_supported_file(file_path: str) -> bool:
    """Check if file type is supported"""
    try:
        from config.settings import SUPPORTED_FORMATS
        extension = Path(file_path).suffix.lower()
        return extension in SUPPORTED_FORMATS
    except ImportError:
        # Fallback to basic check
        supported_exts = {'.pdf', '.txt', '.epub', '.png', '.jpg', '.jpeg'}
        return Path(file_path).suffix.lower() in supported_exts

def find_supported_files(directory: str, max_files: int = None) -> List[str]:
    """
    Find all supported files in directory

    Args:
        directory: Directory to search
        max_files: Maximum number of files to return

    Returns:
        List of supported file paths
    """
    supported_files = []
    directory_path = Path(directory)

    if directory_path.is_file():
        if is_supported_file(str(directory_path)):
            return [str(directory_path)]
        else:
            return []

    if not directory_path.is_dir():
        return []

    try:
        from config.settings import SUPPORTED_FORMATS
        supported_extensions = set(SUPPORTED_FORMATS.keys())
    except ImportError:
        supported_extensions = {'.pdf', '.txt', '.epub', '.png', '.jpg', '.jpeg', '.srt', '.vtt'}

    # Search recursively
    for ext in supported_extensions:
        pattern = f"**/*{ext}"
        files = list(directory_path.glob(pattern))
        supported_files.extend([str(f) for f in files])

        # Limit files if specified
        if max_files and len(supported_files) >= max_files:
            supported_files = supported_files[:max_files]
            break

    return sorted(supported_files)

def get_free_disk_space(path: str) -> int:
    """
    Get free disk space in bytes

    Args:
        path: Path to check

    Returns:
        Free space in bytes
    """
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path),
                ctypes.pointer(free_bytes),
                None,
                None
            )
            return free_bytes.value
        else:  # Unix/Linux/Mac
            statvfs = os.statvfs(path)
            return statvfs.f_frsize * statvfs.f_bavail
    except:
        return 0

def validate_file_size(file_path: str, max_size_bytes: int) -> bool:
    """
    Validate file size is within limits

    Args:
        file_path: Path to file
        max_size_bytes: Maximum allowed size in bytes

    Returns:
        True if file size is acceptable
    """
    try:
        file_size = Path(file_path).stat().st_size
        return file_size <= max_size_bytes
    except:
        return False

def validate_configuration(config: Dict[str, Any]) -> List[str]:
    """
    Validate processing configuration

    Args:
        config: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate required fields
    required_fields = ['mode', 'output_dir']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required configuration: {field}")

    # Validate mode
    valid_modes = ['extract-only', 'generate', 'full', 'resume', 'direct_to_llm']
    if config.get('mode') not in valid_modes:
        errors.append(f"Invalid mode: {config.get('mode')}. Must be one of {valid_modes}")

    # Validate numeric ranges
    numeric_validations = {
        'threads': (1, 64),
        'max_workers': (1, 32),
        'batch_size': (1, 100),
        'chunk_size': (100, 100000),
        'overlap': (0, 1000),
        'min_image_size': (0, 10000000),
        'min_text_length': (0, 100000),
        'quality_threshold': (0.0, 1.0),
        'timeout': (10, 3600)
    }

    for field, (min_val, max_val) in numeric_validations.items():
        if field in config:
            value = config[field]
            if not isinstance(value, (int, float)):
                errors.append(f"{field} must be a number")
            elif not min_val <= value <= max_val:
                errors.append(f"{field} must be between {min_val} and {max_val}, got {value}")

    # Validate page ranges
    start_page = config.get('start_page', 1)
    end_page = config.get('end_page')
    if end_page and start_page > end_page:
        errors.append(f"start_page ({start_page}) cannot be greater than end_page ({end_page})")

    # Validate skip pages
    skip_pages = config.get('skip_pages', [])
    if skip_pages and start_page in skip_pages:
        errors.append(f"Cannot skip start_page {start_page}")

    # Validate output directory
    output_dir = config.get('output_dir')
    if output_dir:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory {output_dir}: {e}")

    # Validate generators for generation modes
    if config.get('mode') in ['generate', 'full']:
        generators = config.get('generators', [])
        valid_generators = ['conversations', 'embeddings', 'qa_pairs', 'summaries']
        for gen in generators:
            if gen not in valid_generators:
                errors.append(f"Invalid generator: {gen}. Must be one of {valid_generators}")

    # Validate file paths
    path_fields = ['plugin_dir', 'config_file', 'resume_from', 'output_template']
    for field in path_fields:
        if config.get(field):
            path = Path(config[field])
            if not path.exists():
                errors.append(f"{field} path does not exist: {config[field]}")

    return errors

def validate_system_requirements() -> Dict[str, Any]:
    """
    Validate system requirements and capabilities

    Returns:
        Dictionary with validation results
    """
    results = {
        'python_version': True,
        'memory': True,
        'disk_space': True,
        'dependencies': [],
        'warnings': [],
        'errors': []
    }

    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version < (3, 8):
        results['python_version'] = False
        results['errors'].append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")

    # Check available memory
    try:
        import psutil
        memory_in_gigabytes = psutil.virtual_memory().total / (1024**3)
        if memory_in_gigabytes < 2:
            results['memory'] = False
            results['errors'].append(f"Insufficient memory: {memory_in_gigabytes:.1f}GB (minimum 2GB)")
        elif memory_in_gigabytes < 4:
            results['warnings'].append(f"Low memory: {memory_in_gigabytes:.1f}GB (recommended 4GB+)")
    except ImportError:
        results['warnings'].append("Cannot check memory (psutil not installed)")

    # Check disk space
    try:
        free_space_in_gigabytes = get_free_disk_space('.') / (1024**3)
        if free_space_in_gigabytes < 1:
            results['disk_space'] = False
            results['errors'].append(f"Insufficient disk space: {free_space_in_gigabytes:.1f}GB (minimum 1GB)")
        elif free_space_in_gigabytes < 5:
            results['warnings'].append(f"Low disk space: {free_space_in_gigabytes:.1f}GB (recommended 5GB+)")
    except:
        results['warnings'].append("Cannot check disk space")

    # Check critical dependencies
    critical_deps = {
        'pathlib': 'pathlib',
        'json': 'json',
        'time': 'time',
        'threading': 'threading'
    }

    for name, module in critical_deps.items():
        try:
            __import__(module)
            results['dependencies'].append({'name': name, 'status': 'available'})
        except ImportError:
            results['dependencies'].append({'name': name, 'status': 'missing'})
            results['errors'].append(f"Missing critical dependency: {name}")

    # Check optional dependencies
    optional_deps = {
        'pytesseract': 'OCR functionality',
        'PIL': 'Image processing',
        'fitz': 'PDF processing',
        'ebooklib': 'EPUB processing',
        'openai': 'OpenAI integration',
        'psutil': 'System monitoring'
    }

    for module, description in optional_deps.items():
        try:
            __import__(module)
            results['dependencies'].append({'name': module, 'status': 'available', 'description': description})
        except ImportError:
            results['dependencies'].append({'name': module, 'status': 'missing', 'description': description})
            results['warnings'].append(f"Optional dependency missing: {module} ({description})")

    return results

def validate_and_report_system() -> bool:
    """
    Validate system and print comprehensive report

    Returns:
        True if system meets minimum requirements
    """
    print("🔍 System Requirements Validation")
    print("=" * 50)

    results = validate_system_requirements()

    # Print Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status = "✅" if results['python_version'] else "❌"
    print(f"{status} Python Version: {python_version}")

    # Print memory info
    try:
        import psutil
        memory_in_gigabytes = psutil.virtual_memory().total / (1024**3)
        status = "✅" if results['memory'] else "❌"
        print(f"{status} Available Memory: {memory_in_gigabytes:.1f} GB")
    except ImportError:
        print("⚠️ Memory: Cannot check (psutil not available)")

    # Print disk space
    try:
        free_space_in_gigabytes = get_free_disk_space('.') / (1024**3)
        status = "✅" if results['disk_space'] else "❌"
        print(f"{status} Free Disk Space: {free_space_in_gigabytes:.1f} GB")
    except:
        print("⚠️ Disk Space: Cannot check")

    # Print dependencies
    print(f"\n📦 Dependencies:")
    for dep in results['dependencies']:
        if dep['status'] == 'available':
            print(f"   ✅ {dep['name']}")
        else:
            desc = dep.get('description', '')
            print(f"   ❌ {dep['name']}" + (f" ({desc})" if desc else ""))

    # Print warnings
    if results['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in results['warnings']:
            print(f"   • {warning}")

    # Print errors
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"   • {error}")
        return False

    print(f"\n✅ System validation passed!")
    return True

def create_validation_report(file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive validation report

    Args:
        file_paths: List of files to validate
        config: Processing configuration

    Returns:
        Validation report dictionary
    """
    report = {
        'timestamp': time.time(),
        'system_validation': validate_system_requirements(),
        'config_validation': {
            'errors': validate_configuration(config),
            'config': config
        },
        'file_validation': {
            'total_files': len(file_paths),
            'supported_files': 0,
            'unsupported_files': 0,
            'large_files': 0,
            'total_size_mb': 0,
            'file_details': []
        }
    }

    # Validate each file
    max_size = config.get('max_file_size', 100 * 1024 * 1024)
    total_size = 0

    for file_path in file_paths:
        try:
            path = Path(file_path)
            size = path.stat().st_size
            total_size += size

            file_info = {
                'path': str(path),
                'name': path.name,
                'size_mb': size / (1024 * 1024),
                'supported': is_supported_file(file_path),
                'too_large': size > max_size,
                'extension': path.suffix.lower()
            }

            report['file_validation']['file_details'].append(file_info)

            if file_info['supported']:
                report['file_validation']['supported_files'] += 1
            else:
                report['file_validation']['unsupported_files'] += 1

            if file_info['too_large']:
                report['file_validation']['large_files'] += 1

        except Exception as e:
            report['file_validation']['file_details'].append({
                'path': file_path,
                'error': str(e),
                'supported': False
            })
            report['file_validation']['unsupported_files'] += 1

    report['file_validation']['total_size_mb'] = total_size / (1024 * 1024)

    return report

# Time utilities
import time

def format_time_elapsed(start_time: float) -> str:
    """Format elapsed time since start_time"""
    elapsed = time.time() - start_time
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    elif elapsed < 3600:
        return f"{elapsed//60:.0f}m {elapsed%60:.0f}s"
    else:
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"
