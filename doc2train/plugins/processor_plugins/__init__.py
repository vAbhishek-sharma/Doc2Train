# plugins/processors_plugins/__init__.py

# Expose all Processor plugins here:
from .sample_custom_video_processor import VideoProcessor

__all__ = [
    "VideoProcessor",
]
