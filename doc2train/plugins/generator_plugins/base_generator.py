# doc2train\plugins\generator_plugins\base_generator.py

class BaseGenerator:
    generator_name = None  # e.g., "conversations"
    types_supported = []   # e.g., ["conversations"]
    priority = 10          # Lower = higher priority

    def __init__(self, config=None):
        self.config = config or {}

    def generate(self, chunk, gen_type, prompt_template=None):
        """
        Generate output for the given chunk and gen_type.
        Must be overridden in each plugin.
        """
        raise NotImplementedError("Generator plugins must implement the generate method.")
