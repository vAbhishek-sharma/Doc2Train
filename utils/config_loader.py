# utils/config_loader.py
"""
Unified YAML configuration loader for Doc2Train v2.0
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """Load and manage YAML configuration with environment variable support"""

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                print(f"✅ Loaded config from {self.config_file}")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
                self.config = self._get_default_config()
        else:
            print(f"📝 Creating default config file: {self.config_file}")
            self.config = self._get_default_config()
            self.save_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'mode': 'extract-only',
            'output_dir': 'output',
            'output_format': 'jsonl',
            'processing': {
                'use_async': True,
                'threads': 4,
                'max_workers': 4,
                'batch_size': 10,
                'timeout': 300,
                'max_file_size': '100MB',
                'use_cache': True,
                'save_per_file': False
            },
            'pages': {
                'start_page': 1,
                'end_page': None,
                'skip_pages': []
            },
            'quality': {
                'min_image_size': 1000,
                'min_text_length': 100,
                'skip_single_color': False,
                'header_regex': '',
                'quality_threshold': 0.7
            },
            'features': {
                'extract_images': True,
                'use_ocr': True,
                'include_vision': False,
                'smart_pdf_analysis': True
            },
            'llm': {
                'provider': 'openai',
                'model': '',
                'fallback_provider': 'deepseek',
                'max_concurrent_calls': 5,
                'api_keys': {
                    'openai': '',
                    'deepseek': ''
                }
            },
            'generation': {
                'types': ['conversations', 'qa_pairs'],
                'chunk_size': 4000,
                'overlap': 200
            },
            'prompts': {
                'style': 'default',
                'custom': {}
            },
            'debug': {
                'verbose': False,
                'show_progress': True,
                'show_images': False,
                'test_mode': False,
                'dry_run': False,
                'benchmark': False,
                'validate_only': False
            },
            'advanced': {
                'plugin_dir': '',
                'resume_from': '',
                'clear_cache_after': False
            }
        }

    def save_config(self):
        """Save current configuration to YAML file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            print(f"💾 Saved config to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")

    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        Example: get('llm.provider') returns config['llm']['provider']
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value):
        """
        Set configuration value using dot notation
        Example: set('llm.provider', 'openai')
        """
        keys = key_path.split('.')
        config = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the final value
        config[keys[-1]] = value

    def update_from_args(self, args):
        """Update configuration from simplified command line arguments"""
        # Map simplified command line args to config paths
        if hasattr(args, 'mode') and args.mode:
            self.set('mode', args.mode)

        if hasattr(args, 'output_dir') and args.output_dir:
            self.set('output_dir', args.output_dir)

        if hasattr(args, 'threads') and args.threads:
            self.set('processing.threads', args.threads)

        if hasattr(args, 'sync') and args.sync:
            self.set('processing.use_async', False)

        if hasattr(args, 'async_calls') and args.async_calls:
            self.set('llm.max_concurrent_calls', args.async_calls)

        if hasattr(args, 'provider') and args.provider:
            self.set('llm.provider', args.provider)

        if hasattr(args, 'prompt_style') and args.prompt_style:
            self.set('prompts.style', args.prompt_style)

        if hasattr(args, 'test_mode') and args.test_mode:
            self.set('debug.test_mode', True)

        if hasattr(args, 'verbose') and args.verbose:
            self.set('debug.verbose', True)

        if hasattr(args, 'dry_run') and args.dry_run:
            self.set('debug.dry_run', True)

        # Set input path if provided
        if hasattr(args, 'input_path') and args.input_path:
            self.set('input_path', args.input_path)

    def get_processing_config(self) -> Dict[str, Any]:
        """Get flattened config for processing pipeline"""
        return {
            # Basic settings
            'mode': self.get('mode'),
            'output_dir': self.get('output_dir'),
            'output_format': self.get('output_format'),

            # Processing settings
            'use_async': self.get('processing.use_async'),
            'threads': self.get('processing.threads'),
            'max_workers': self.get('processing.max_workers'),
            'batch_size': self.get('processing.batch_size'),
            'timeout': self.get('processing.timeout'),
            'max_file_size': self._parse_file_size(self.get('processing.max_file_size')),
            'use_cache': self.get('processing.use_cache'),
            'save_per_file': self.get('processing.save_per_file'),

            # Page control
            'start_page': self.get('pages.start_page'),
            'end_page': self.get('pages.end_page'),
            'skip_pages': self.get('pages.skip_pages', []),

            # Quality control
            'min_image_size': self.get('quality.min_image_size'),
            'min_text_length': self.get('quality.min_text_length'),
            'skip_single_color': self.get('quality.skip_single_color'),
            'header_regex': self.get('quality.header_regex'),
            'quality_threshold': self.get('quality.quality_threshold'),

            # Features
            'extract_images': self.get('features.extract_images'),
            'use_ocr': self.get('features.use_ocr'),
            'include_vision': self.get('features.include_vision'),
            'use_smart_analysis': self.get('features.smart_pdf_analysis'),

            # LLM settings
            'provider': self.get('llm.provider'),
            'model': self.get('llm.model'),
            'max_concurrent_calls': self.get('llm.max_concurrent_calls'),

            # Generation settings
            'generators': self.get('generation.types'),
            'chunk_size': self.get('generation.chunk_size'),
            'overlap': self.get('generation.overlap'),

            # Custom prompts
            'custom_prompts': self._get_custom_prompts(),

            # Debug settings
            'verbose': self.get('debug.verbose'),
            'show_progress': self.get('debug.show_progress'),
            'show_images': self.get('debug.show_images'),
            'test_mode': self.get('debug.test_mode'),
            'dry_run': self.get('debug.dry_run'),
            'benchmark': self.get('debug.benchmark'),
            'validate_only': self.get('debug.validate_only'),

            # Advanced
            'plugin_dir': self.get('advanced.plugin_dir'),
            'resume_from': self.get('advanced.resume_from'),
            'clear_cache_after_run': self.get('advanced.clear_cache_after')
        }

    def _parse_file_size(self, size_str: str) -> int:
        """Parse file size string to bytes"""
        if not size_str:
            return 100 * 1024 * 1024  # 100MB default

        size_str = str(size_str).upper().strip()

        if size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        else:
            return int(size_str)

    def _get_custom_prompts(self) -> Dict[str, str]:
        """Get custom prompts based on style and overrides"""
        style = self.get('prompts.style', 'default')
        custom = self.get('prompts.custom', {})

        # If we have completely custom prompts, use them
        if custom:
            return custom

        # Otherwise, use style-based prompts
        return self._get_style_prompts(style)

    def _get_style_prompts(self, style: str) -> Dict[str, str]:
        """Get prompts for a specific style"""
        style_prompts = self.get(f'advanced.prompt_styles.{style}', {})

        if style_prompts:
            return style_prompts

        # Fallback to default prompts
        return {
            'conversations': "Create a natural conversation between a user and an AI assistant based on this content.",
            'qa_pairs': "Generate questions and answers based on this content.",
            'summaries': "Create a summary of this content.",
            'embeddings': "Create sentence pairs for embedding training."
        }

# Global config loader instance
_config_loader = None

def get_config_loader(config_file: str = "config.yaml") -> ConfigLoader:
    """Get global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_file)
    return _config_loader

def load_config_from_yaml(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration and return processing config"""
    loader = get_config_loader(config_file)
    return loader.get_processing_config()
