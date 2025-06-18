# plugins/generator_plugins/embeddings_generator.py

from doc2train.plugins.generator_plugins.base_generator import BaseGenerator

class EmbeddingsGenerator(BaseGenerator):
    generator_name = "embeddings"
    types_supported = ["embeddings"]
    priority = 10

    def generate(self, chunk, gen_type, prompt_template=None):
        prompt = prompt_template or self.config.get("prompts", {}).get("custom", {}).get("embeddings")
        # Insert actual LLM call here
        return {
            "embeddings": [
                {"sentence1": "Doc2Train is modular.", "sentence2": "The tool supports plugins.", "similarity": 0.92},
                {"sentence1": "It's not a monolith.", "sentence2": "It's flexible.", "similarity": 0.81}
            ]
        }
