# plugins/processors/custom_video_processor.py
from doc2train.plugins.processor_plugins.base_processor import BaseProcessor

class VideoProcessor(BaseProcessor):
    """Custom video processor using direct LLM analysis"""
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    priority ='10'
    description = ''
    version = '1.0.0'
    author = 'doc2train'
    processor_name = 'VideoProcessor (Sample)'
    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        self.processor_name = "VideoProcessor"

    def extract_content_impl(self, file_path: str):
        from doc2train.core.llm_client import process_media_directly
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

        return {'text': "\n\n".join(analyses), 'videos':[]}
