# plugins/processors/custom_video_processor.py
from processors.base_processor import BaseProcessor
from core.llm_plugin_manager import process_media_directly

class VideoProcessor(BaseProcessor):
    """Custom video processor using direct LLM analysis"""

    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        self.processor_name = "VideoProcessor"

    def extract_content_impl(self, file_path: str):
        """Extract content from video using LLM"""
        # Option 1: Process directly with LLM
        if self.config.get('use_direct_processing', True):
            analysis = (
                file_path,
                "Analyze this video and extract key information"
            )
            return analysis, []

        # Option 2: Extract frames and process
        frames = self._extract_key_frames(file_path)
        analyses = []

        for frame in frames:
            frame_analysis = process_media_directly(
                frame,
                prompt="Describe what's happening in this video frame"
            )
            analyses.append(frame_analysis)

        return "\n\n".join(analyses), []
