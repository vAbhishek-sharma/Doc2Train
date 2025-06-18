# plugins/generator_plugins/summaries_generator.py

from doc2train.plugins.generator_plugins.base_generator import BaseGenerator

class SummariesGenerator(BaseGenerator):
    generator_name = "summaries"
    types_supported = ["summaries"]
    priority = 10

    def generate(self, chunk, gen_type, prompt_template=None):
        prompt = prompt_template or self.config.get("prompts", {}).get("custom", {}).get("summaries")
        # Insert actual LLM call here
        return {
            "summaries": [
                {"summary": "Doc2Train is a modular, plugin-based document processing and training data generation tool."}
            ]
        }
