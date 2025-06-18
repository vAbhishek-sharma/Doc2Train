# plugins/generator_plugins/qa_pairs_generator.py

from doc2train.plugins.generator_plugins.base_generator import BaseGenerator

class QAPairsGenerator(BaseGenerator):
    generator_name = "qa_pairs"
    types_supported = ["qa_pairs"]
    priority = 10

    def generate(self, chunk, gen_type, prompt_template=None):
        prompt = prompt_template or self.config.get("prompts", {}).get("custom", {}).get("qa_pairs")
        # Insert actual LLM call here
        return {
            "qa_pairs": [
                {"question": "What is Doc2Train?", "answer": "A modular doc/LLM pipeline."},
                {"question": "Is it plugin-based?", "answer": "Yes, for all major components."}
            ]
        }
